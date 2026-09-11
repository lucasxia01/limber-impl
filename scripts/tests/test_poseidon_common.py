#!/usr/bin/env python3
"""Unit tests for scripts/poseidon_common.py (limber system profile)."""
import json
import os
import shutil
import tempfile
import unittest
from fractions import Fraction

from scripts import poseidon_common as pc
from scripts.tests import support

SAMPLE_FUNC = "hyrax/mixed3/Hpf1-total3/c2^11v2^11/k9/inst0/thr1/primary/blk2"
SAMPLE_DIR = "hyrax_mixed3_Hpf1-total3_c2_11v2_11_k9_inst0_thr1_primary_blk2"
DIAG_FUNC = "brakedown/mixed3/Hpf10-total30/c2^14v2^14/k13/inst9/thr1/diagnostic/blkdiag"
DIAG_DIR = "brakedown_mixed3_Hpf10-total30_c2_14v2_14_k13_inst9_thr1_diagnos"


class CanonicalJsonTests(unittest.TestCase):
    def setUp(self):
        support.guard_bytecode(self)

    def test_sorted_compact_lf(self):
        self.assertEqual(pc.canonical_json_bytes({"b": 1, "a": [True, None, "x"]}),
                         b'{"a":[true,null,"x"],"b":1}\n')

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            pc.canonical_json_bytes({"a": 1.5})

    def test_load_canonical_rejects_whitespace(self):
        with self.assertRaises(pc.RunnerError) as cm:
            pc.load_canonical_json(b'{"a": 1}\n')
        self.assertEqual(cm.exception.error_code, "CanonicalJsonInvalid")
        self.assertEqual(pc.load_canonical_json(b'{"a":1}\n'), {"a": 1})


class DigestTests(unittest.TestCase):
    def test_detached_digest_round_trip(self):
        d = "a" * 64
        self.assertEqual(pc.format_detached_digest(d), (d + "\n").encode())
        self.assertEqual(pc.parse_detached_digest((d + "\n").encode()), d)

    def test_detached_digest_rejects_variants(self):
        for bad in (b"A" * 64 + b"\n", b"a" * 64, b"a" * 64 + b"\r\n", b" " + b"a" * 63 + b"\n",
                    b"a" * 64 + b"\n\n"):
            with self.assertRaises(pc.RunnerError):
                pc.parse_detached_digest(bad)


class KnobTests(unittest.TestCase):
    def test_flag_and_usize(self):
        self.assertFalse(pc.knob_flag({}, "SWEEP"))
        self.assertTrue(pc.knob_flag({"SWEEP": "1"}, "SWEEP"))
        with self.assertRaises(pc.RunnerError):
            pc.knob_flag({"SWEEP": "true"}, "SWEEP")
        self.assertEqual(pc.knob_usize({"THREADS": "8"}, "THREADS"), 8)
        for bad in ("08", "+1", "-1", "1 ", ""):
            with self.assertRaises(pc.RunnerError):
                pc.knob_usize({"THREADS": bad}, "THREADS")

    def test_rejected_even_when_default(self):
        with self.assertRaises(pc.RunnerError) as cm:
            pc.reject_knobs({"THREADS": "1"}, ("THREADS",), "sweep")
        self.assertEqual(cm.exception.error_code, "RejectedKnob")

    def test_mode_conflict(self):
        with self.assertRaises(pc.RunnerError) as cm:
            pc.select_mode({"SWEEP": "1", "PSIZE": "1"})
        self.assertEqual(cm.exception.error_code, "ModeConflict")
        self.assertEqual(pc.select_mode({"SWEEP": "0"}), "normal")
        self.assertEqual(pc.select_mode({"PSIZE": "1"}), "psize")

    def test_backend_enum(self):
        self.assertEqual(pc.knob_enum({"BACKEND": "brakedown"}, "BACKEND", pc.BACKENDS),
                         "brakedown")
        with self.assertRaises(pc.RunnerError):
            pc.knob_enum({"BACKEND": "zinc"}, "BACKEND", pc.BACKENDS)
        self.assertNotIn("VARIANT", pc.KNOB_NAMES)
        self.assertNotIn("FL", pc.KNOB_NAMES)
        self.assertIn("BACKEND", pc.KNOB_NAMES)


class SystemProfileTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(pc.SYSTEM_ROLES, ("hyrax", "brakedown"))
        self.assertEqual(pc.VARIANTS, pc.BACKENDS)
        # Single-candidate TUNE-1 epoch (tuning-protocol-v2 `single_candidate_epoch`): the
        # pinned order is just the persisted default k = 9 inside the admissible range.
        self.assertEqual(pc.K_ORDER, (9,))
        self.assertEqual(pc.K_RANGE, (7, 13))
        self.assertEqual(pc.CANDIDATES, pc.K_ORDER)
        self.assertEqual(pc.SIMPLICITY_ORDER, tuple(sorted(pc.K_ORDER)))
        self.assertEqual(pc.SIMPLICITY_ORDER, (9,))
        self.assertEqual(len(set(pc.K_ORDER)), len(pc.K_ORDER))
        self.assertTrue(all(pc.K_RANGE[0] <= k <= pc.K_RANGE[1] for k in pc.K_ORDER))
        self.assertEqual(pc.TUNING_INSTANCES, tuple(range(1, 10)))
        self.assertEqual(pc.CHECK_MODES, {"shadow": "not_applicable", "timed": "release"})
        self.assertEqual(pc.PRIME_SAMPLER_ID,
                         "limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096")
        self.assertEqual(pc.BENCHMARK_COINS_ID, "poseidon-bench-chacha20-v1")
        self.assertEqual(pc.PROTOCOL_WIRE_ID, "limber/wire/v-p0d")
        self.assertEqual(pc.PRIME_SAMPLER_CAPS, {"t_max": 4096, "mr_rounds": 72,
                                                 "base_draw_max": 160,
                                                 "max_invocations": 4096})
        self.assertEqual(pc.APPROVED_CAPABILITY_TUPLES[0]["variants"], pc.BACKENDS)
        self.assertEqual(pc.CAPABILITY_KEYS, ("protocol_wire_id", "prime_sampler_id",
                                              "benchmark_coins_id"))
        self.assertEqual(pc.SPEC_FILES["tune-corpus-v1.json"][0],
                         "tests/data/tune-corpus-v1.json")
        self.assertIn("dimensions_by_k", pc.METADATA_REQUIRED_KEYS)
        self.assertIn("check_mode", pc.METADATA_REQUIRED_KEYS)
        self.assertNotIn("checked_flag", pc.METADATA_REQUIRED_KEYS)
        self.assertEqual(pc.METADATA_FORBIDDEN_KEYS, ("checked_flag", "unchecked_flag"))
        self.assertEqual(pc.NORMAL_CHILDREN, 13)
        self.assertEqual(pc.SESSION_CHILDREN, 39)

    def test_diagnostic_groups_per_backend(self):
        self.assertEqual(pc.diagnostic_groups_for("hyrax"),
                         ("setup", "advice", "commit_witness", "prove_after_input_commit"))
        self.assertEqual(pc.diagnostic_groups_for("brakedown"),
                         ("setup", "commit_witness", "prove_after_input_commit"))
        with self.assertRaises(pc.RunnerError):
            pc.diagnostic_groups_for("zinc")

    def test_benchmark_coins_policy_from_spec(self):
        with open(os.path.join(support.REAL_REPO, "specs", "poseidon",
                               "tuning-protocol-v2.json"), "rb") as f:
            spec = pc.load_canonical_json(f.read())
        policy = pc.benchmark_coins_policy(spec)
        self.assertEqual(policy["id"], "poseidon-bench-chacha20-v1")
        self.assertEqual(policy["framing"], pc.BENCHMARK_COINS_FRAMING)
        self.assertEqual(spec["systems"]["limber"]["k_order"], list(pc.K_ORDER))
        self.assertEqual(pc.benchmark_coins_policy({})["framing"], pc.BENCHMARK_COINS_FRAMING)

    def test_evidence_rules_overlay(self):
        with open(os.path.join(support.REAL_REPO, "specs", "poseidon",
                               "timing-schema-v2.json"), "rb") as f:
            schema = pc.load_canonical_json(f.read())
        rules = pc.evidence_artifact_rules(schema)
        self.assertEqual(rules["target_name"], "poseidon_modp")
        self.assertEqual(rules["src_path"], "<SOURCE>/benches/poseidon_modp.rs")
        self.assertEqual(rules["target_kind"], ["bench"])
        self.assertEqual(rules["count"], 1)
        # The shared schema is untouched.
        self.assertEqual(schema["evidence"]["compiler_artifact"]["target_name"], "poseidon")


class CommandTests(unittest.TestCase):
    def test_exact_commands(self):
        prefix = "cargo bench --locked --offline --profile bench --color never".split()
        self.assertEqual(pc.bench_prefix([]), prefix + ["--bench", "poseidon_modp", "-vv"])
        self.assertEqual(pc.evidence_command([]),
                         prefix + ["--message-format=json-render-diagnostics", "--bench",
                                   "poseidon_modp", "--no-run", "-vv"])
        self.assertEqual(pc.metadata_command([])[-2:], ["--", "--print-protocol-metadata"])
        self.assertEqual(pc.preflight_command([], "/c", "a" * 64, "/d")[-8:],
                         ["--run-config", "/c", "--config-sha256", "a" * 64, "--attempt",
                          "preflight", "--artifact-dir", "/d"])
        self.assertEqual(pc.child_command([], "/c", "b" * 64, "/d")[-6:],
                         ["--child-config", "/c", "--child-config-sha256", "b" * 64,
                          "--artifact-dir", "/d"])
        self.assertEqual(" ".join(pc.gate_command("kat")),
                         "cargo test --locked --offline --color never --lib -vv -- "
                         "--exact poseidon2::tests::kat_gate")
        self.assertEqual(" ".join(pc.gate_command("tune_corpus")),
                         "cargo test --locked --offline --color never --lib -vv -- "
                         "--exact poseidon2::tests::tuning_corpus_gate")
        for cmd in (pc.bench_prefix([]), pc.evidence_command([]), pc.gate_command("kat")):
            self.assertNotIn("-p", cmd)
            self.assertNotIn("--features", cmd)
        self.assertEqual(pc.features_for_threads(1), [])
        self.assertEqual(pc.features_for_threads(4), [])
        self.assertEqual(pc.bootstrap_feature_tokens([]), [])
        with self.assertRaises(pc.RunnerError):
            pc.bench_prefix(["parallel"])
        with self.assertRaises(pc.RunnerError):
            pc.bootstrap_feature_tokens(["parallel"])
        with self.assertRaises(pc.RunnerError):
            pc.features_for_threads(0)

    def test_metadata_parse_requires_one_canonical_line(self):
        raw = support.metadata_for(support.REAL_REPO)
        good = pc.canonical_json_bytes(raw)
        md = pc.parse_protocol_metadata(good, ["cargo"])
        self.assertEqual(md.compiled_ids(), raw)
        self.assertEqual(md.capability_tuple(),
                         {"protocol_wire_id": "limber/wire/v-p0d",
                          "prime_sampler_id": pc.PRIME_SAMPLER_ID,
                          "benchmark_coins_id": "poseidon-bench-chacha20-v1"})
        for bad in (b"   Compiling\n" + good, good + b"\n", good[:-1],
                    json.dumps(raw).encode() + b"\n"):
            with self.assertRaises(pc.RunnerError) as cm:
                pc.parse_protocol_metadata(bad, ["cargo"])
            self.assertIn(cm.exception.error_code, ("MetadataMalformed", "CanonicalJsonInvalid"))
        for key in ("argv_forms_version", "dimensions_by_k", "check_mode", "benchmark_coins_id"):
            broken = dict(raw)
            del broken[key]
            with self.assertRaises(pc.RunnerError) as cm:
                pc.parse_protocol_metadata(pc.canonical_json_bytes(broken), ["cargo"])
            self.assertEqual(cm.exception.error_code, "MetadataMalformed")

    def test_tuned_default_forms(self):
        raw = support.metadata_for(support.REAL_REPO)
        md = pc.ProtocolMetadata(raw, ["cargo"])
        self.assertEqual(md.tuned_default("hyrax"), {"k": None, "tuning_id": None})
        md.raw["tuned_defaults"] = [{"backend": "hyrax", "k": 9, "tuning_id": "a" * 64},
                                    ["brakedown", 11, "b" * 64]]
        self.assertEqual(md.tuned_default("hyrax"), {"k": 9, "tuning_id": "a" * 64})
        self.assertEqual(md.tuned_default("brakedown"), {"k": 11, "tuning_id": "b" * 64})

    def test_dimensions_by_candidate(self):
        raw = support.metadata_for(support.REAL_REPO)
        md = pc.ProtocolMetadata(raw, ["cargo"])
        # The default is the pinned schedule (K_ORDER); the compiled `dimensions_by_k`
        # covers the whole admissible range, so any explicit k in K_RANGE resolves.
        dims = pc.dimensions_by_candidate(md)
        self.assertEqual(sorted(dims), sorted(str(k) for k in pc.K_ORDER))
        self.assertEqual(dims["9"], {"log_cons": 14, "log_vars": 14})
        lo, hi = pc.K_RANGE
        full = pc.dimensions_by_candidate(md, range(lo, hi + 1))
        self.assertEqual(sorted(full, key=int), [str(k) for k in range(lo, hi + 1)])
        for k in range(lo, hi + 1):
            self.assertEqual(pc.dimensions_by_candidate(md, [k]),
                             {str(k): {"log_cons": 14, "log_vars": 14}})
        # A k outside 7..=13 has no compiled dimensions and is rejected.
        for k in (lo - 1, hi + 1):
            with self.assertRaises(pc.RunnerError) as cm:
                pc.dimensions_by_candidate(md, [k])
            self.assertEqual(cm.exception.error_code, "MetadataMalformed")
        # An admissible k whose entry is missing or malformed is rejected, whether it
        # is requested explicitly or as part of the pinned schedule.
        del md.raw["dimensions_by_k"]["11"]
        with self.assertRaises(pc.RunnerError) as cm:
            pc.dimensions_by_candidate(md, [11])
        self.assertEqual(cm.exception.error_code, "MetadataMalformed")
        md.raw["dimensions_by_k"]["11"] = {"log_cons": 14}
        with self.assertRaises(pc.RunnerError):
            pc.dimensions_by_candidate(md, [11])
        md.raw["dimensions_by_k"]["11"] = {"log_cons": 14, "log_vars": True}
        with self.assertRaises(pc.RunnerError):
            pc.dimensions_by_candidate(md, [11])
        del md.raw["dimensions_by_k"][str(pc.K_ORDER[0])]
        with self.assertRaises(pc.RunnerError) as cm:
            pc.dimensions_by_candidate(md)
        self.assertEqual(cm.exception.error_code, "MetadataMalformed")

    def test_error_code_sections(self):
        self.assertIn("RejectedEnvironment", pc.ERROR_CODES)
        self.assertIn("ExecutableChanged", pc.ERROR_CODES)
        self.assertIn("RustcArgvInvalid", pc.ERROR_CODES)
        self.assertEqual(pc.RunnerError("dependency", "DependencyClosureChanged", "x").exit_code, 3)


class ExactArithmeticTests(unittest.TestCase):
    def test_estimates_parsed_exactly(self):
        path = os.path.join(support.CRIT_SAMPLE, "verify_core", SAMPLE_DIR, "new",
                            "estimates.json")
        with open(path, "rb") as f:
            est = pc.parse_estimates(f.read())
        self.assertEqual(est["median"]["point_estimate"], Fraction("11781495.8"))
        self.assertEqual(est["median"]["lower_bound"], Fraction("11518691.128472222"))
        self.assertEqual(est["mean"]["confidence_level"], Fraction("0.95"))

    def test_decimal_round_trip(self):
        for text in ("11781495.8", "0.95", "12001394.818075396", "100", "-3.25", "0"):
            self.assertEqual(pc.fraction_to_decimal(Fraction(text)), text)
        with self.assertRaises(ValueError):
            pc.fraction_to_decimal(Fraction(1, 3))

    def test_median_and_ratio(self):
        self.assertEqual(pc.exact_median([3, 1, 2]), Fraction(2))
        self.assertEqual(pc.exact_median([Fraction("1.5"), Fraction("2.5")]), Fraction(2))
        self.assertEqual(pc.ratio_pair(Fraction(6, 4)), ["3", "2"])
        self.assertEqual(pc.ratio_from_pair(["3", "2"]), Fraction(3, 2))
        with self.assertRaises(ValueError):
            pc.ratio_from_pair(["6", "4"])


class TimeTests(unittest.TestCase):
    def test_rfc3339_and_compact(self):
        ts = pc.utc_rfc3339_ns(1_700_000_000_123_456_789)
        self.assertEqual(ts, "2023-11-14T22:13:20.123456789Z")
        self.assertEqual(pc.compact_timestamp(ts), "20231114T221320.123456789Z")


class DependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = support.make_repo(self.tmp)

    def test_lock_and_manifest(self):
        deps = pc.dependency_versions(self.repo)
        self.assertEqual(deps["criterion"]["version"], "0.7.0")
        self.assertEqual(deps["criterion"]["features"], ["cargo_bench_support"])
        self.assertEqual(deps["zstd"], {"crate": None, "sys": None})

    def test_wrong_criterion_rejected(self):
        with open(os.path.join(self.repo, "Cargo.lock"), "a") as f:
            f.write('\n[[package]]\nname = "criterion"\nversion = "0.5.1"\n')
        with self.assertRaises(pc.RunnerError) as cm:
            pc.dependency_versions(self.repo)
        self.assertEqual(cm.exception.error_code, "DependencyMismatch")
        with open(os.path.join(self.repo, "Cargo.lock"), "w") as f:
            f.write(support.CARGO_LOCK)
        with open(os.path.join(self.repo, "Cargo.toml"), "w") as f:
            f.write(support.CARGO_TOML.replace('default-features = false, ', ''))
        with self.assertRaises(pc.RunnerError) as cm:
            pc.dependency_versions(self.repo)
        self.assertEqual(cm.exception.error_code, "DependencyMismatch")

    def test_git_helpers(self):
        self.assertFalse(pc.git_is_dirty(self.repo))
        self.assertTrue(pc.git_is_tracked(self.repo, "Cargo.lock"))
        with open(os.path.join(self.repo, "untracked.txt"), "w") as f:
            f.write("x")
        self.assertTrue(pc.git_is_dirty(self.repo))
        self.assertEqual(pc.repo_root(), support.REAL_REPO)


class CriterionGrammarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = os.path.join(self.tmp, "criterion")
        shutil.copytree(os.path.join(support.CRIT_SAMPLE, "verify_core"),
                        os.path.join(self.root, "verify_core"))
        self.expect = {"groups": {"verify_core"}, "backend": "hyrax", "hashes": 1,
                       "instance": 0, "k": 9, "threads": 1, "kind": "primary", "block": 2,
                       "dimensions": {"log_cons": 11, "log_vars": 11}}

    def test_names_match_sample(self):
        with open(os.path.join(self.root, "verify_core", SAMPLE_DIR, "new",
                               "benchmark.json"), "rb") as f:
            bench = json.loads(f.read())
        names = pc.criterion_names(bench["group_id"], bench["function_id"],
                                   bench["value_str"])
        for key in ("full_id", "directory_name", "title"):
            self.assertEqual(names[key], bench[key])
        self.assertEqual(bench["function_id"], SAMPLE_FUNC)
        self.assertEqual(pc.criterion_filename_safe(SAMPLE_FUNC), SAMPLE_DIR)
        # The diagnostic sample is the 64-byte truncation of its `^`/`/`-escaped id.
        self.assertEqual(len(DIAG_DIR.encode()), 64)
        self.assertEqual(pc.criterion_filename_safe(DIAG_FUNC), DIAG_DIR)
        self.assertEqual(pc.function_id("hyrax", 1, {"log_cons": 11, "log_vars": 11}, 9, 0, 1,
                                        "primary", 2), SAMPLE_FUNC)
        self.assertEqual(pc.function_id("brakedown", 10, {"log_cons": 14, "log_vars": 14}, 13,
                                        9, 1, "diagnostic", "diag"), DIAG_FUNC)

    def test_sample_validates(self):
        recs = pc.validate_criterion_tree(self.root, self.expect)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["parsed"]["block"], 2)
        self.assertEqual(recs[0]["parsed"]["k"], 9)
        self.assertEqual(recs[0]["estimates"]["median"]["point_estimate"],
                         Fraction("11781495.8"))
        # Without a dimensions expectation the IDs' log sizes are not checked.
        exp = dict(self.expect)
        del exp["dimensions"]
        self.assertEqual(len(pc.validate_criterion_tree(self.root, exp)), 1)

    def test_diagnostic_sample_validates(self):
        root = os.path.join(self.tmp, "diag")
        shutil.copytree(os.path.join(support.CRIT_SAMPLE, "setup"),
                        os.path.join(root, "setup"))
        exp = {"groups": {"setup"}, "backend": "brakedown", "hashes": 10, "instance": 9,
               "k": 13, "threads": 1, "kind": "diagnostic", "block": "diag",
               "dimensions": {"log_cons": 14, "log_vars": 14}}
        self.assertEqual(len(pc.validate_criterion_tree(root, exp)), 1)
        with self.assertRaises(pc.RunnerError):
            pc.validate_criterion_tree(root, dict(exp, groups={"setup", "advice"}))

    def assertInvalid(self, expect=None):
        with self.assertRaises(pc.RunnerError) as cm:
            pc.validate_criterion_tree(self.root, expect or self.expect)
        self.assertEqual(cm.exception.error_code, "CriterionTreeInvalid")

    def test_unlisted_leaf_rejected(self):
        with open(os.path.join(self.root, "verify_core", SAMPLE_DIR, "new", "raw.csv"),
                  "w") as f:
            f.write("x")
        self.assertInvalid()

    def test_renamed_directory_rejected(self):
        os.rename(os.path.join(self.root, "verify_core", SAMPLE_DIR),
                  os.path.join(self.root, "verify_core", SAMPLE_DIR[:-1]))
        self.assertInvalid()

    def test_base_new_divergence_rejected(self):
        p = os.path.join(self.root, "verify_core", SAMPLE_DIR, "base", "estimates.json")
        with open(p, "a") as f:
            f.write(" ")
        self.assertInvalid()

    def test_wrong_confidence_level_rejected(self):
        for sub in ("base", "new"):
            p = os.path.join(self.root, "verify_core", SAMPLE_DIR, sub, "estimates.json")
            with open(p, "r+") as f:
                text = f.read().replace("0.95", "0.99")
                f.seek(0)
                f.write(text)
                f.truncate()
        self.assertInvalid()

    def test_wrong_expectation_rejected(self):
        self.assertInvalid(dict(self.expect, block=3))
        self.assertInvalid(dict(self.expect, groups={"prove_e2e"}))
        self.assertInvalid(dict(self.expect, k=10))
        self.assertInvalid(dict(self.expect, backend="brakedown"))
        self.assertInvalid(dict(self.expect, dimensions={"log_cons": 12, "log_vars": 11}))

    def test_function_id_parse(self):
        parsed = pc.parse_function_id("brakedown/mixed3/Hpf10-total30/c2^14v2^14/k13/inst9/"
                                      "thr1/primary/blk3")
        self.assertEqual((parsed["backend"], parsed["k"], parsed["block"], parsed["log_cons"],
                          parsed["log_vars"], parsed["instance"], parsed["kind"]),
                         ("brakedown", 13, 3, 14, 14, 9, "primary"))
        diag = pc.parse_function_id(DIAG_FUNC)
        self.assertEqual((diag["kind"], diag["block"]), ("diagnostic", "diag"))
        for bad in ("hyrax/mixed3/Hpf10-total31/c2^14v2^14/k9/inst0/thr1/primary/blk0",
                    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k14/inst0/thr1/primary/blk0",
                    "zincplus/qz/mixed3/Hpf10-total30/nvars10/deg3/wcols72/fl4/flat/inst0/"
                    "thr1/chk0/primary/blk0",
                    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/chk0/primary/blk0"):
            with self.assertRaises(pc.RunnerError):
                pc.parse_function_id(bad)


if __name__ == "__main__":
    unittest.main()


class CriterionTitleTests(unittest.TestCase):
    def test_collision_suffix_accepted(self):
        base = "steps/zincplus/qz/mixed3/Hpf10-total30/nvars10/deg3/wcols72/fl4/square/inst0/thr1/chk0/diagnostic/blkdiag/prover_step_0"
        exp = pc.criterion_names("steps", base[len("steps/"):].rsplit("/", 1)[0], "prover_step_0")["title"]
        self.assertTrue(pc.criterion_title_matches(exp, exp))
        self.assertTrue(pc.criterion_title_matches(exp, exp + " #2"))
        self.assertTrue(pc.criterion_title_matches(exp, exp + " #17"))
        self.assertFalse(pc.criterion_title_matches(exp, exp + " #1"))
        self.assertFalse(pc.criterion_title_matches(exp, exp + " #02"))
        self.assertFalse(pc.criterion_title_matches(exp, exp + "#2"))
        self.assertFalse(pc.criterion_title_matches(exp, exp + " #2x"))
        self.assertFalse(pc.criterion_title_matches(exp, "other"))

