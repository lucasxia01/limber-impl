#!/usr/bin/env python3
"""End-to-end tests of the sweep summarizer, the tuning assembler/applier and the exact
full-size TUNE-1 schedule arithmetic (limber: N = 7 candidate k values), all against the
stub build environment / bench. The three-system session orchestrator lives in the Zinc
repository and drives this runner through the CLI contract; it is exercised there with
`stub_limber_runner.py`-style wrappers."""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from scripts import apply_poseidon_tuning as apply_tuning
from scripts import assemble_poseidon_tuning as assemble
from scripts import poseidon_common as pc
from scripts import poseidon_runner as runner
from scripts import poseidon_tune1 as t1
from scripts import summarize_poseidon_sweep as summarize
from scripts.tests import support
from scripts.tests.test_poseidon_runner import only_dir


# The historical full-size k order (every admissible k, N = 7) keeps the multi-candidate
# schedule arithmetic covered now that the pinned epoch (`pc.K_ORDER`) schedules a single k.
FULL_K_ORDER = (10, 7, 12, 9, 13, 8, 11)
FULL_SIMPLICITY_ORDER = tuple(sorted(FULL_K_ORDER))


class ScheduleArithmeticTests(unittest.TestCase):
    """The full-size TUNE-1 schedule: N = 7, 2N = 14 blocks, 9N = 63 processes per block,
    18N^2 = 882 children, ordinals 1..882, position balance, simplicity order; plus the
    pinned single-candidate schedule (N = 1: 2 blocks x 9 = 18 children)."""

    def test_full_size_counts(self):
        n = len(FULL_K_ORDER)
        self.assertEqual(n, 7)
        self.assertEqual(sorted(FULL_K_ORDER), list(range(pc.K_RANGE[0], pc.K_RANGE[1] + 1)))
        self.assertEqual(FULL_SIMPLICITY_ORDER, (7, 8, 9, 10, 11, 12, 13))
        orders = t1.block_orders(FULL_K_ORDER)
        self.assertEqual(len(orders), 2 * n)
        self.assertEqual(orders[0], list(FULL_K_ORDER))
        self.assertEqual(orders[1], [7, 12, 9, 13, 8, 11, 10])
        self.assertEqual(orders[n], [11, 8, 13, 9, 12, 7, 10])
        self.assertEqual(orders[n + 1], [8, 13, 9, 12, 7, 10, 11])
        # Every candidate occupies every within-block position exactly twice.
        for k in FULL_K_ORDER:
            for pos in range(n):
                self.assertEqual(sum(1 for o in orders if o[pos] == k), 2, (k, pos))
        schedule = t1.schedule(FULL_K_ORDER, pc.TUNING_INSTANCES)
        self.assertEqual(len(schedule), 18 * n * n)
        self.assertEqual(len(schedule), 882)
        per_block = {}
        for (b, o, cand, inst) in schedule:
            per_block.setdefault(b, []).append((o, cand, inst))
        self.assertEqual(sorted(per_block), list(range(14)))
        for b, items in per_block.items():
            self.assertEqual(len(items), 9 * n)
            self.assertEqual([o for (o, _, _) in items], list(range(63)))
            # Within a block: for instance in 1..=9 ascending, for candidate in the block order.
            self.assertEqual([inst for (_, _, inst) in items],
                             [i for i in range(1, 10) for _ in range(n)])
            self.assertEqual([cand for (_, cand, _) in items], orders[b] * 9)
        names = ["children/block-%d/%s" % (b, t1.child_dir_name(o, c, i))
                 for (b, o, c, i) in schedule]
        self.assertEqual(len(set(names)), 882)
        self.assertEqual(names[0], "children/block-0/00-k10-inst1")
        self.assertEqual(orders[13], [10, 11, 8, 13, 9, 12, 7])
        self.assertEqual(names[-1], "children/block-13/62-k7-inst9")
        # The normal-mode child counts and the session-wide count are unchanged.
        self.assertEqual((pc.NORMAL_PRIMARY_CHILDREN, pc.NORMAL_CHILDREN, pc.SESSION_CHILDREN),
                         (12, 13, 39))
        self.assertEqual(len(runner.normal_child_plan()), 13)

    def test_full_size_recheck_arithmetic(self):
        """A synthetic full-size result (882 processes) reproduces through the re-checker."""
        from fractions import Fraction as F
        processes = {}
        for (b, o, cand, inst) in t1.schedule(FULL_K_ORDER, pc.TUNING_INSTANCES):
            point = F(1000 + (cand - 7) * 40 + b)
            processes[(cand, inst, b)] = {"point": point, "lower": point - 5, "upper": point + 5}
        result = t1.tuning_result({"backend": "brakedown"}, {}, list(FULL_K_ORDER),
                                  list(pc.TUNING_INSTANCES), processes, {},
                                  simplicity_order=list(FULL_SIMPLICITY_ORDER))
        self.assertEqual(len(result["process_estimates"]), 882)
        self.assertEqual(len(result["blocks"]), 14)
        self.assertEqual(result["decision"], {"selected": 7,
                                              "selection_basis": "simplest_unique_best"})
        self.assertEqual(sorted(result["aggregate"]["scores"]),
                         sorted(str(k) for k in FULL_K_ORDER))
        t1.recheck_tuning_result(pc.load_canonical_json(pc.canonical_json_bytes(result)))

    def test_pinned_single_candidate_schedule(self):
        """The pinned epoch schedules `pc.K_ORDER` alone: 2N blocks x 9N children (18 for
        N = 1) and the lone candidate is selected as `only_admissible`."""
        from fractions import Fraction as F
        cands = list(pc.K_ORDER)
        n = len(cands)
        self.assertEqual(pc.SIMPLICITY_ORDER, tuple(sorted(cands)))
        orders = t1.block_orders(cands)
        self.assertEqual(len(orders), 2 * n)
        self.assertEqual(orders[0], cands)
        self.assertEqual(orders[n], list(reversed(cands)))
        schedule = t1.schedule(cands, pc.TUNING_INSTANCES)
        self.assertEqual(len(schedule), 18 * n * n)
        names = ["children/block-%d/%s" % (b, t1.child_dir_name(o, c, i))
                 for (b, o, c, i) in schedule]
        self.assertEqual(len(set(names)), len(schedule))
        processes = {}
        for (b, o, cand, inst) in schedule:
            point = F(1000 + b)
            processes[(cand, inst, b)] = {"point": point, "lower": point - 5, "upper": point + 5}
        result = t1.tuning_result({"backend": "hyrax"}, {}, cands, list(pc.TUNING_INSTANCES),
                                  processes, {}, simplicity_order=list(pc.SIMPLICITY_ORDER))
        self.assertEqual(len(result["process_estimates"]), len(schedule))
        self.assertEqual(result["decision"]["selected"], pc.SIMPLICITY_ORDER[0])
        self.assertEqual(result["decision"]["selection_basis"],
                         "only_admissible" if n == 1 else "simplicity_tie_band")
        if n == 1:
            self.assertEqual((len(orders), len(schedule)), (2, 18))
            self.assertEqual(names[0], "children/block-0/00-k%d-inst1" % cands[0])
            self.assertEqual(names[-1], "children/block-1/08-k%d-inst9" % cands[0])
        t1.recheck_tuning_result(pc.load_canonical_json(pc.canonical_json_bytes(result)))


class TuningToolsTests(unittest.TestCase):
    def setUp(self):
        support.guard_bytecode(self)
        support.shrink_sweep(self)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = support.make_repo(self.tmp)
        self.metadata = support.write_metadata(self.tmp, self.repo)
        self.sbe = support.install_stub(self, self.metadata)
        self.env_base = support.runner_env()

    def sweep(self, backend: str) -> str:
        out = os.path.join(self.tmp, "sweeps", "sweep-out-%s" % backend)
        os.makedirs(out)
        env = dict(self.env_base, SWEEP="1", BACKEND=backend, OUTPUT_PARENT=out)
        code = runner.main(["--repo", self.repo], env)
        self.assertEqual(code, 0)
        archive = only_dir(out)
        manifest = support.read_json(os.path.join(archive, "manifest.json"))
        self.assertEqual(manifest["eligible_for"], ["tuning"])
        return archive

    def test_summarize_assemble_apply(self):
        hyrax = self.sweep("hyrax")
        manifest, recomputed, archived, tuning_id = summarize.rederive(hyrax)
        self.assertEqual(recomputed, archived)
        self.assertEqual(pc.sha256_hex(recomputed), tuning_id)
        self.assertEqual(manifest["tuning_id"], tuning_id)
        self.assertEqual(summarize.main([hyrax, "--quiet"]), 0)
        # A tampered estimate is detected by the re-derivation.
        tampered = os.path.join(self.tmp, "tampered")
        shutil.copytree(hyrax, tampered)
        est = None
        for root, _, files in os.walk(os.path.join(tampered, "children")):
            for name in files:
                if name == "estimates.json" and root.endswith("new"):
                    est = os.path.join(root, name)
                    break
            if est:
                break
        import re
        with open(est, "r+") as f:
            text = re.sub(r'"point_estimate": ([0-9]+)', lambda m: '"point_estimate": %d' % (
                int(m.group(1)) + 1), f.read(), count=1)
            f.seek(0)
            f.write(text)
            f.truncate()
        with self.assertRaises(pc.RunnerError):
            summarize.rederive(tampered)
        # Assemble the limber-both epoch (needs both backends); the store must be disjoint
        # from the bundle-output parent, the archives and the source worktree.
        brakedown = self.sweep("brakedown")
        store = os.path.join(self.tmp, "store")
        os.mkdir(store)
        bundles = os.path.join(self.tmp, "bundles")
        os.mkdir(bundles)
        bundle_path = os.path.join(bundles, "bundle.json")
        code = assemble.main(["--group-set", "limber-both", "--store", store, "--out",
                              os.path.join(self.tmp, "overlapping.json"), hyrax, brakedown])
        self.assertEqual(code, pc.ERROR_CODES["PathOverlap"])
        code = assemble.main(["--group-set", "limber-both", "--store", store, "--out",
                              bundle_path + ".partial", hyrax])
        self.assertEqual(code, pc.ERROR_CODES["BundleInvalid"])
        code = assemble.main(["--group-set", "limber-both", "--store", store, "--out",
                              bundle_path, hyrax, brakedown])
        self.assertEqual(code, 0)
        with open(bundle_path, "rb") as f:
            bundle_bytes = f.read()
        bundle = pc.load_canonical_json(bundle_bytes)
        epoch_id = pc.sha256_hex(bundle_bytes)
        self.assertEqual(pc.read_detached_digest(os.path.join(bundles, "bundle.sha256")),
                         epoch_id)
        self.assertEqual(bundle["schema"], "limber/poseidon-tuning-bundle/v1")
        self.assertEqual(bundle["group_set"], "limber-both")
        self.assertEqual([g["backend"] for g in bundle["groups"]], ["hyrax", "brakedown"])
        self.assertEqual(bundle["generated_module_path"], apply_tuning.MODULE_PATH)
        self.assertEqual(bundle["generated_module_path"], "src/poseidon_tuned_defaults.rs")
        self.assertEqual(sorted(bundle["capability"]),
                         ["benchmark_coins_id", "prime_sampler_id", "protocol_wire_id",
                          "transcript_domain_separator"])
        self.assertIn("comparison_schema_id", bundle["shared_ids"])
        cfg = support.read_json(os.path.join(hyrax, "run-config.json"))
        self.assertEqual(bundle["source_snapshot_id"], cfg["source"]["source_snapshot_id"])
        for g in bundle["groups"]:
            self.assertTrue(os.path.isdir(os.path.join(store, g["origin_manifest_sha256"])))
            self.assertEqual(g["selected"], 7)
            self.assertEqual(g["selection_basis"], "simplest_unique_best")
            self.assertEqual(g["dependency_source_id"],
                             cfg["dependency_sources"]["dependency_source_id"])
            self.assertTrue(pc.is_hex64(g["tuning_id"]))
        # Re-assembling reuses the identical store objects; two archives for one backend
        # are rejected.
        self.assertEqual(assemble.main(["--group-set", "limber-both", "--store", store, "--out",
                                        bundle_path + ".2", hyrax, brakedown]), 0)
        self.assertEqual(assemble.main(["--group-set", "limber-both", "--store", store, "--out",
                                        bundle_path + ".3", hyrax, hyrax]),
                         pc.ERROR_CODES["BundleInvalid"])
        # A rename race (another importer wins the no-replace rename) reuses the winner after
        # a full rehash; a corrupted winner is fatal.
        store2 = os.path.join(self.tmp, "store2")
        os.mkdir(store2)
        real_commit = assemble.af.commit_staging

        def race(staging, final):
            shutil.copytree(staging, final)
            raise pc.RunnerError("commit", "RenameRefused", "simulated race")
        with mock.patch.object(assemble.af, "commit_staging", side_effect=race):
            self.assertEqual(assemble.main(["--group-set", "limber-both", "--store", store2,
                                            "--out", bundle_path + ".4", hyrax, brakedown]), 0)
        self.assertEqual(sorted(os.listdir(store2)),
                         sorted(g["origin_manifest_sha256"] for g in bundle["groups"]))
        with open(os.path.join(store2, bundle["groups"][0]["origin_manifest_sha256"],
                               "bench.log"), "ab") as f:
            f.write(b"!")
        self.assertEqual(assemble.main(["--group-set", "limber-both", "--store", store2, "--out",
                                        bundle_path + ".5", hyrax, brakedown]),
                         pc.ERROR_CODES["IntegrityMismatch"])
        self.assertIs(assemble.af.commit_staging, real_commit)
        # Apply renders a deterministic module that --check accepts and that names the epoch.
        repo2 = os.path.join(self.tmp, "repo2")
        os.makedirs(os.path.join(repo2, "src"))
        self.assertEqual(apply_tuning.main(["--bundle", bundle_path, "--repo", repo2]), 0)
        with open(os.path.join(repo2, apply_tuning.MODULE_PATH), "rb") as f:
            module = f.read()
        self.assertIn(epoch_id.encode(), module)
        self.assertIn(b'pub const TUNING_GROUP_SET: Option<&str> = Some("limber-both");', module)
        self.assertIn(b"pub const TUNED_DEFAULTS: &[(&str, usize, &str)] = &[\n", module)
        self.assertIn(b'  ("hyrax", 7, "%s"),\n' % bundle["groups"][0]["tuning_id"].encode(),
                      module)
        self.assertIn(b'  ("brakedown", 7, "%s"),\n' % bundle["groups"][1]["tuning_id"].encode(),
                      module)
        self.assertNotIn(b"rustfmt::skip", module)
        self.assertTrue(all(len(line) <= 100 for line in module.decode().split("\n")))
        self.assertEqual(module, apply_tuning.render(bundle))
        self.assertEqual(apply_tuning.main(["--bundle", bundle_path, "--repo", repo2,
                                            "--check"]), 0)
        with open(os.path.join(repo2, apply_tuning.MODULE_PATH), "ab") as f:
            f.write(b"// drift\n")
        self.assertEqual(apply_tuning.main(["--bundle", bundle_path, "--repo", repo2,
                                            "--check"]), 3)
        # The initial (pre-epoch) rendering round-trips through --check on a fresh
        # repository and is what the fixture repository (and this checkout, before the
        # first epoch) tracks.
        repo3 = os.path.join(self.tmp, "repo3")
        os.makedirs(os.path.join(repo3, "src"))
        self.assertEqual(apply_tuning.main(["--initial", "--repo", repo3, "--render-to",
                                            os.path.join(self.tmp, "initial.rs")]), 0)
        self.assertEqual(apply_tuning.main(["--initial", "--repo", repo3, "--check"]), 0)
        with open(os.path.join(self.tmp, "initial.rs"), "rb") as f:
            initial = f.read()
        self.assertEqual(initial, apply_tuning.render(None))
        self.assertIn(b"pub const TUNING_EPOCH_ID: Option<&str> = None;", initial)
        self.assertIn(b"pub const TUNED_DEFAULTS: &[(&str, usize, &str)] = &[];", initial)
        self.assertIn(b"v9 default `k = 9`", initial)
        with open(os.path.join(self.repo, apply_tuning.MODULE_PATH), "rb") as f:
            self.assertEqual(f.read(), initial)
        # An unrenderable bundle (k outside 7..=13, unknown backend) is rejected.
        bad = dict(bundle, groups=[dict(bundle["groups"][0], selected=14)] + bundle["groups"][1:])
        with self.assertRaises(pc.RunnerError):
            apply_tuning.render(bad)
        bad = dict(bundle, groups=[dict(bundle["groups"][0], backend="zinc")])
        with self.assertRaises(pc.RunnerError):
            apply_tuning.render(bad)


if __name__ == "__main__":
    unittest.main()
