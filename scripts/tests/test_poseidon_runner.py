#!/usr/bin/env python3
"""End-to-end tests for scripts/poseidon_runner.py (limber) against the stub build
environment, stub cargo and stub bench (three argv forms).

Sweeps run with the shrunken TUNE-1 schedule of `support.shrink_sweep` (N = 3 candidates,
2 instances, 36 children); `test_poseidon_tuning_session` pins the full-size arithmetic."""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr

from scripts import apply_poseidon_tuning as apply_tuning
from scripts import artifact_fs as af
from scripts import poseidon_common as pc
from scripts import poseidon_runner as runner
from scripts import poseidon_tune1 as t1
from scripts.tests import support

SESSION_SPEC = {
    "schema": "poseidon-comparison-session-config/v2",
    "nonce_hex": "ab" * 32,
    "created_utc": "2026-09-08T00:00:00.000000000Z",
    "hashes_per_field": 10, "instance": 0, "threads": 1,
    "systems": {"zinc": {"variant": "qz", "fl": 4}, "hyrax": {"backend": "hyrax"},
                "brakedown": {"backend": "brakedown"}},
    "schedule": {"metric_order": list(pc.PRIMARY_GROUPS),
                 "block_orders": [list(o) for o in pc.SESSION_BLOCK_ORDERS],
                 "diagnostic_order": ["zinc", "hyrax", "brakedown"]},
}
CHILD_CONFIG_KEYS = ["coordinate", "order", "ordinal", "parent_config", "parent_config_sha256",
                     "schema", "session_id", "system_role"]
GATE_KEYS = ["argv", "dependency_source_id", "end_utc", "exit", "features", "log_sha256",
             "post", "pre", "profile", "signal", "start_utc", "status", "target_audit_id",
             "test_summary"]
AUDIT_KEYS = ["cargo_config", "dependency", "executable", "native", "source", "toolchain"]
ORDER = '["zinc","hyrax","brakedown"]'


def only_dir(parent):
    names = [n for n in os.listdir(parent) if not n.startswith(".")]
    return os.path.join(parent, names[0]) if len(names) == 1 else None


def reseal(root, **patch):
    """Test surgery: re-seal an archive's manifest with patched fields (index unchanged)."""
    with open(os.path.join(root, "manifest.json"), "rb") as f:
        manifest = json.loads(f.read())
    manifest.update(patch)
    for n in ("manifest.json", "manifest.sha256"):
        os.unlink(os.path.join(root, n))
    return af.seal_manifest(root, manifest)


def null_execution_fields(entry):
    return all(entry[k] is None for k in runner.EXECUTION_FIELDS)


class RunnerTestBase(unittest.TestCase):
    def setUp(self):
        support.guard_bytecode(self)
        support.shrink_sweep(self)
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = support.make_repo(self.tmp)
        self.metadata = support.write_metadata(self.tmp, self.repo)
        self.sbe = support.install_stub(self, self.metadata)
        self.out = os.path.join(self.tmp, "out")
        os.mkdir(self.out)
        self.addCleanup(self.assert_role_dirs_removed)

    def assert_role_dirs_removed(self):
        for path in self.sbe.CREATED_TARGETS:
            self.assertFalse(os.path.exists(path), "role target %s was not removed" % path)

    def stub(self, **settings):
        self.sbe.STUB_SETTINGS.update(settings)

    def run_runner(self, argv, env):
        err = io.StringIO()
        with redirect_stderr(err):
            code = runner.main(["--repo", self.repo] + argv, env)
        return code, err.getvalue()

    def env(self, **knobs):
        return support.runner_env(**knobs)

    def sweep_env(self, backend="hyrax", **extra):
        return self.env(SWEEP="1", BACKEND=backend, OUTPUT_PARENT=self.out, **extra)


class SweepTests(RunnerTestBase):
    def test_sweep_success(self):
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, 0, err)
        final = only_dir(self.out)
        self.assertTrue(os.path.basename(final).startswith("run-"))
        self.assertNotIn(".staging", os.path.basename(final))
        required, globs = runner.expected_payload_set("sweep", "complete")
        manifest, _, status = af.verify_artifact_dir(final, required, globs)
        self.assertEqual(status, "complete")
        self.assertEqual(manifest["mode"], "sweep")
        self.assertEqual(manifest["system_role"], "hyrax")
        self.assertTrue(manifest["artifact_integrity_valid"])
        self.assertTrue(manifest["provenance_valid"])
        self.assertTrue(manifest["execution_valid"])
        self.assertEqual(manifest["failure_reasons"], [])
        self.assertEqual(manifest["eligible_for"], ["tuning"])
        self.assertIsNone(manifest["tuning_epoch_id"])
        self.assertEqual(manifest["config_id"],
                         pc.read_detached_digest(os.path.join(final, "run-config.sha256")))
        self.assertEqual(manifest["ids"]["benchmark_coins_policy"]["id"],
                         "poseidon-bench-chacha20-v1")
        for name in runner.EVIDENCE_FILES:
            self.assertTrue(os.path.isfile(os.path.join(final, name)), name)
        # Run config: contract keys, projected environment, no raw temporary locator.
        with open(os.path.join(final, "run-config.json"), "rb") as f:
            cfg_bytes = f.read()
        cfgj = pc.load_canonical_json(cfg_bytes)
        for key in ("backend", "k", "dimensions", "dimensions_by_candidate", "workload",
                    "sweep", "check_modes", "criterion", "tuning", "comparison_key", "ids",
                    "features", "rustflags", "source", "lockfile", "dependency_sources",
                    "toolchain", "native_tools", "environment", "executable",
                    "cargo_config_audit", "rejected_inputs_audit", "build", "machine",
                    "preflight_requirements"):
            self.assertIn(key, cfgj)
        for key in ("variant", "fl", "shape", "num_vars"):
            self.assertNotIn(key, cfgj)
        self.assertEqual(cfgj["schema"], "limber/poseidon-run-config/v2")
        self.assertEqual((cfgj["system_role"], cfgj["backend"]), ("hyrax", "hyrax"))
        self.assertEqual((cfgj["k"], cfgj["dimensions"], cfgj["tuning_id"],
                          cfgj["comparison_key"]), (None, None, None, None))
        self.assertEqual(cfgj["workload"], {"backend": "hyrax", "k": None,
                                            "hashes_per_field": 10, "threads": 1,
                                            "instance": None, "candidates": [10, 7, 9],
                                            "instances": [1, 2],
                                            "simplicity_order": [7, 9, 10],
                                            "dimensions_by_candidate": {
                                                "10": {"log_cons": 14, "log_vars": 14},
                                                "7": {"log_cons": 14, "log_vars": 14},
                                                "9": {"log_cons": 14, "log_vars": 14}}})
        self.assertEqual(cfgj["sweep"]["candidates"], [10, 7, 9])
        self.assertEqual(cfgj["sweep"]["simplicity_order"], [7, 9, 10])
        self.assertEqual(cfgj["sweep"]["blocks"], 6)
        self.assertEqual(len(cfgj["sweep"]["schedule"]), 36)
        self.assertEqual(sorted(cfgj["dimensions_by_candidate"]), ["10", "7", "9"])
        self.assertEqual(cfgj["features"], [])
        self.assertEqual(cfgj["check_modes"], {"shadow": "not_applicable", "timed": "release"})
        self.assertEqual(cfgj["preflight_requirements"],
                         {"check_mode": "not_applicable",
                          "deterministic_coins": "poseidon-bench-chacha20-v1",
                          "double_construction": True, "p0d_audit": True,
                          "rayon_threads_asserted": 1})
        self.assertEqual(cfgj["criterion"], {"sample_size": 10, "warm_up_time_s": 1,
                                             "measurement_time_s": 20})
        self.assertEqual(cfgj["criterion_policy"]["diagnostic_order"],
                         ["setup", "advice", "commit_witness", "prove_after_input_commit"])
        self.assertEqual(cfgj["environment"]["CARGO_TARGET_DIR"], "<ROLE_TARGET>")
        self.assertEqual(cfgj["environment"]["TMPDIR"], "<ROLE_TEMP>")
        self.assertEqual(cfgj["environment"]["RAYON_NUM_THREADS"], "1")
        self.assertNotIn("STUB_", cfg_bytes.decode())
        for path in self.sbe.CREATED_TARGETS:
            self.assertNotIn(path.encode(), cfg_bytes)
        self.assertEqual(cfgj["ids"]["compiled"], support.read_json(self.metadata))
        self.assertEqual(cfgj["ids"]["benchmark_coins_policy"],
                         {"id": "poseidon-bench-chacha20-v1",
                          "framing": pc.BENCHMARK_COINS_FRAMING})
        self.assertTrue(cfgj["ids"]["comparison_schema"]["match"])
        self.assertEqual(cfgj["build"]["evidence_command"],
                         ["<SYSROOT>/bin/cargo"] + pc.evidence_command([])[1:])
        self.assertEqual(cfgj["build"]["gate_commands"]["kat"][1:], pc.gate_command("kat")[1:])
        self.assertFalse(cfgj["build"]["parallel"])
        self.assertEqual(cfgj["source"]["source_snapshot_id"], manifest["source_snapshot_id"])
        self.assertFalse(cfgj["source"]["dirty"])
        self.assertIsNone(cfgj["source"]["dirty_closure_sha256"])
        self.assertEqual(cfgj["dependency_sources"]["dependency_source_id"],
                         pc.sha256_file(os.path.join(final, "dependency-sources.json")))
        self.assertEqual(cfgj["toolchain"]["sysroot"], "<SYSROOT>")
        self.assertEqual(cfgj["toolchain"]["wrapper_chain"], [])
        self.assertTrue(cfgj["executable"]["relative_path"].startswith("release/deps/"
                                                                       "poseidon_modp-"))
        # Exactly 2N = 6 blocks x 6 children with the counterbalanced order.
        blocks = sorted(os.listdir(os.path.join(final, "children")))
        self.assertEqual(blocks, ["block-%d" % b for b in range(6)])
        b1 = sorted(os.listdir(os.path.join(final, "children", "block-1")))
        self.assertEqual(b1[:3], ["00-k7-inst1", "01-k9-inst1", "02-k10-inst1"])
        self.assertEqual(len(b1), 6)
        child = os.path.join(final, "children", "block-1", "00-k7-inst1")
        self.assertEqual(sorted(os.listdir(child)), ["bench.log", "child-config.json",
                                                     "child-result.json", "criterion"])
        result = support.read_json(os.path.join(child, "child-result.json"))
        self.assertEqual(result["schema"], "limber/poseidon-child-result/v2")
        self.assertEqual(len(result["criterion_files"]), 8)
        self.assertEqual(sorted(result["pre_audit"]), AUDIT_KEYS)
        self.assertEqual(result["pre_audit"], result["post_audit"])
        self.assertIn("/k7/inst1/thr1/primary/blk1/", "/".join(
            f[0] for f in result["criterion_files"]).replace("_", "/") or "")
        cc = support.read_json(os.path.join(child, "child-config.json"))
        self.assertEqual(sorted(cc), CHILD_CONFIG_KEYS)
        self.assertEqual(cc["schema"], "limber/poseidon-child-config/v2")
        self.assertEqual(cc["parent_config_sha256"], manifest["config_id"])
        self.assertEqual(cc["parent_config"], cfgj)
        self.assertIsNone(cc["order"])
        self.assertEqual(cc["coordinate"], {"kind": "sweep", "block": 1, "ordinal": 0,
                                            "candidate": 7, "instance": 1})
        self.assertEqual(cfgj["sweep"]["block_orders"][1], {"block": 1, "order": [7, 9, 10]})
        self.assertEqual(cc["ordinal"], 7)
        self.assertIsNone(cc["session_id"])
        # Tuning result reproduces and hashes to tuning_id; the simplest k wins.
        tid = pc.read_detached_digest(os.path.join(final, "tuning-result.sha256"))
        self.assertEqual(manifest["tuning_id"], tid)
        with open(os.path.join(final, "tuning-result.json"), "rb") as f:
            data = f.read()
        self.assertEqual(pc.sha256_hex(data), tid)
        result = pc.load_canonical_json(data)
        t1.recheck_tuning_result(result)
        self.assertEqual(result["schema"], "limber/poseidon-tuning-result/v2")
        self.assertEqual(result["group"], {"backend": "hyrax"})
        self.assertEqual(result["candidates"], [10, 7, 9])
        self.assertEqual(result["simplicity_order"], [7, 9, 10])
        self.assertEqual(result["decision"], {"selected": 7,
                                              "selection_basis": "simplest_unique_best"})
        self.assertEqual(result["aggregate"]["simplest"], 7)
        self.assertEqual(result["aggregate"]["wins_vs_simplest"], {"10": 0, "7": 0, "9": 0})
        self.assertEqual(result["aggregate"]["ci_parity_count"]["7"], 2)
        self.assertEqual(len(result["process_estimates"]), 36)
        self.assertEqual(result["provenance"]["run_config_sha256"], manifest["config_id"])
        self.assertEqual(result["provenance"]["dependency_source_id"],
                         cfgj["dependency_sources"]["dependency_source_id"])
        # Journal: evidence_metadata, kat, tune_corpus, preflight_matrix, then 36 children.
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        self.assertEqual(journal["schema"], "limber/poseidon-process-journal/v2")
        names = [e["name"] for e in journal["entries"]]
        self.assertEqual(names[:4], ["evidence_metadata", "kat", "tune_corpus",
                                     "preflight_matrix"])
        self.assertEqual([e["kind"] for e in journal["entries"]].count("child"), 36)
        self.assertTrue(all(e["status"] == "ok" for e in journal["entries"]))
        for e in journal["entries"]:
            self.assertEqual(sorted(e["pre_audit"]), AUDIT_KEYS, e["name"])
            self.assertEqual(sorted(e["post_audit"]), AUDIT_KEYS, e["name"])
            self.assertIsNotNone(e["log_sha256"])
        ev = journal["entries"][0]
        self.assertIsNone(ev["pre_audit"]["executable"])  # no executable before the build
        self.assertEqual(ev["post_audit"], journal["entries"][1]["pre_audit"])
        for e in journal["entries"][1:]:
            self.assertEqual(e["pre_audit"], e["post_audit"], e["name"])
        self.assertEqual(sorted(ev["attempt_result"]),
                         ["dependency_audit", "evidence", "metadata", "terminal_recheck"])
        self.assertEqual(ev["attempt_result"]["metadata"]["argv"][-2:],
                         ["--", "--print-protocol-metadata"])
        self.assertEqual(ev["attempt_result"]["metadata"]["argv"][:11],
                         ["<SYSROOT>/bin/cargo"] + pc.bench_prefix([])[1:])
        self.assertEqual(journal["entries"][3]["attempt_result"]["status"], "ok")
        self.assertEqual(journal["entries"][3]["attempt_result"]["schema"],
                         runner.ATTEMPT_RESULT_SCHEMA)
        self.assertEqual(journal["entries"][4]["argv"][-6], "--child-config")
        # Gate results: exactly kat and tune_corpus, the crate's --lib tests without -p.
        gates = support.read_json(os.path.join(final, "gate-results.json"))
        self.assertEqual(list(gates), ["kat", "tune_corpus"])
        for name, g in gates.items():
            self.assertEqual(sorted(g), GATE_KEYS)
            self.assertEqual(g["status"], "ok")
            self.assertEqual(g["argv"], ["<SYSROOT>/bin/cargo"] + pc.gate_command(name)[1:])
            self.assertIn("--lib", g["argv"])
            self.assertEqual(g["argv"][-1], pc.GATE_TEST_NAMES[name])
            self.assertEqual((g["profile"], g["features"], g["exit"]), ("release", [], 0))
            self.assertTrue(g["test_summary"]["passed"])
            self.assertEqual(g["dependency_source_id"],
                             cfgj["dependency_sources"]["dependency_source_id"])
            self.assertEqual(sorted(g["pre"]), AUDIT_KEYS)
            self.assertEqual(g["target_audit_id"], cfgj["build"]["role_target_audit_id"])
        meta = support.read_json(os.path.join(final, "sweep-metadata.json"))
        self.assertEqual(meta["schema"], runner.SWEEP_METADATA_SCHEMA)
        self.assertEqual(meta["config_sha256"], manifest["config_id"])
        self.assertEqual(len(meta["matrix"]), 6)
        self.assertTrue(all(m["status"] == "ok" for m in meta["matrix"]))
        self.assertEqual(meta["matrix"][0]["preflight"]["schema"], runner.PREFLIGHT_SCHEMA)
        self.assertEqual(meta["matrix"][0]["preflight"]["rayon_threads_asserted"], 1)
        self.assertEqual(meta["matrix"][0]["preflight"]["prime_audit"]["prime_sampler_id"],
                         pc.PRIME_SAMPLER_ID)

    def test_brakedown_sweep(self):
        code, err = self.run_runner([], self.sweep_env("brakedown"))
        self.assertEqual(code, 0, err)
        final = only_dir(self.out)
        manifest = support.read_json(os.path.join(final, "manifest.json"))
        self.assertEqual(manifest["system_role"], "brakedown")
        cfg = support.read_json(os.path.join(final, "run-config.json"))
        self.assertEqual(cfg["criterion_policy"]["diagnostic_order"],
                         ["setup", "commit_witness", "prove_after_input_commit"])
        result = support.read_json(os.path.join(final, "tuning-result.json"))
        self.assertEqual(result["group"], {"backend": "brakedown"})

    def test_child_failure_produces_failed_archive(self):
        self.stub(fail_child="9:2:1")
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, pc.ERROR_CODES["ChildFailed"], err)
        final = only_dir(self.out)
        required, globs = runner.expected_payload_set("sweep", "failed")
        manifest, _, status = af.verify_artifact_dir(final, required, globs)
        self.assertEqual(status, "failed")
        self.assertEqual(manifest["error_code"], "ChildFailed")
        self.assertEqual(manifest["eligible_for"], [])
        self.assertFalse(os.path.exists(os.path.join(final, "children")))
        self.assertFalse(os.path.exists(os.path.join(final, "sweep-metadata.json")))
        failure = support.read_json(os.path.join(final, "failure.json"))
        self.assertEqual(failure["schema"], "limber/poseidon-failure/v2")
        self.assertEqual(failure["coordinate"]["candidate"], 9)
        self.assertEqual(failure["coordinate"]["block"], 1)
        self.assertEqual(len(failure["preflight_matrix"]), 6)
        self.assertEqual(failure["preflight_evidence"]["attempt_result"]["status"], "ok")
        self.assertEqual(failure["exit_code"], 101)
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        children = [e for e in journal["entries"] if e["kind"] == "child"]
        statuses = [e["status"] for e in children]
        self.assertEqual(statuses.count("failed"), 1)
        self.assertIn("not_run", statuses)
        self.assertTrue(all(null_execution_fields(e) for e in children
                            if e["status"] == "not_run"))
        self.assertEqual(sorted(os.listdir(final)),
                         sorted(list(required) + ["artifacts.sha256", "manifest.failed.json",
                                                  "manifest.failed.sha256"]))

    def test_preflight_failure_records_complete_matrix(self):
        self.stub(fail_preflight="10:2")
        code, _ = self.run_runner([], self.sweep_env())
        self.assertEqual(code, pc.ERROR_CODES["PreflightFailed"])
        final = only_dir(self.out)
        failure = support.read_json(os.path.join(final, "failure.json"))
        self.assertEqual(len(failure["preflight_matrix"]), 6)
        bad = [m for m in failure["preflight_matrix"] if not m["admissible"]]
        self.assertEqual([[m["candidate"], m["instance"]] for m in bad], [[10, 2]])
        self.assertIn("status", bad[0]["failed_conjuncts"])
        self.assertIn("preflight_double_construction_components_equal",
                      bad[0]["failed_conjuncts"])
        self.assertEqual(failure["preflight_evidence"]["attempt_result"]["status"], "failed")
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        pf = journal["entries"][3]
        self.assertEqual((pf["name"], pf["status"], pf["exit_code"]),
                         ("preflight_matrix", "failed", 3))
        self.assertTrue(all(e["status"] == "not_run" and null_execution_fields(e)
                            for e in journal["entries"][4:]))

    def test_gate_failure(self):
        self.stub(fail_gate="tuning_corpus_gate")
        code, _ = self.run_runner([], self.sweep_env())
        self.assertEqual(code, pc.ERROR_CODES["GateFailed"])
        final = only_dir(self.out)
        gates = support.read_json(os.path.join(final, "gate-results.json"))
        self.assertEqual([gates["kat"]["status"], gates["tune_corpus"]["status"]],
                         ["ok", "failed"])
        self.assertTrue(gates["tune_corpus"]["test_summary"]["result_line"].startswith(
            "test result: FAILED"))
        self.assertEqual(gates["tune_corpus"]["exit"], 101)
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        self.assertEqual([e["status"] for e in journal["entries"][:4]],
                         ["ok", "ok", "failed", "not_run"])
        self.assertTrue(null_execution_fields(journal["entries"][3]))
        manifest = support.read_json(os.path.join(final, "manifest.failed.json"))
        self.assertIn("gate_failed:\"tune_corpus\"", manifest["failure_reasons"])

    def test_dirty_worktree(self):
        with open(os.path.join(self.repo, "scratch.txt"), "w") as f:
            f.write("x")
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, pc.ERROR_CODES["DirtyWorktree"])
        self.assertIn("DirtyWorktree", err)
        self.assertEqual(os.listdir(self.out), [])
        code, _ = self.run_runner([], self.sweep_env(POSEIDON_ALLOW_DIRTY="1"))
        self.assertEqual(code, 0)
        final = only_dir(self.out)
        manifest = support.read_json(os.path.join(final, "manifest.json"))
        self.assertFalse(manifest["provenance_valid"])
        self.assertEqual(manifest["eligible_for"], [])
        self.assertIn("dirty_worktree_allowed", manifest["failure_reasons"])
        self.assertIn("dirty_override_active", manifest["failure_reasons"])
        cfg = support.read_json(os.path.join(final, "run-config.json"))
        self.assertTrue(cfg["source"]["dirty"])
        self.assertTrue(pc.is_hex64(cfg["source"]["dirty_closure_sha256"]))
        # The override forces role-ineligibility even on a clean tree; provenance stays valid.
        os.unlink(os.path.join(self.repo, "scratch.txt"))
        out2 = os.path.join(self.tmp, "out-clean-override")
        os.mkdir(out2)
        code, _ = self.run_runner([], self.env(SWEEP="1", BACKEND="hyrax", OUTPUT_PARENT=out2,
                                               POSEIDON_ALLOW_DIRTY="1"))
        self.assertEqual(code, 0)
        manifest = support.read_json(os.path.join(only_dir(out2), "manifest.json"))
        self.assertTrue(manifest["provenance_valid"])
        self.assertEqual(manifest["eligible_for"], [])
        self.assertEqual(manifest["failure_reasons"], ["dirty_override_active"])

    def test_source_change_during_run_is_fatal(self):
        real = self.sbe.run_cargo_subprocess
        calls = []

        def mutate_then_run(argv, env, cwd, bundle, sink):
            calls.append(argv)
            if len(calls) == 3:  # after metadata and the KAT gate: the corpus gate
                with open(os.path.join(self.repo, "mutation.txt"), "w") as f:
                    f.write("x")
            return real(argv, env, cwd, bundle, sink)
        self.sbe.run_cargo_subprocess = mutate_then_run
        try:
            code, err = self.run_runner([], self.sweep_env())
        finally:
            self.sbe.run_cargo_subprocess = real
        self.assertEqual(code, pc.ERROR_CODES["SourceChanged"], err)
        final = only_dir(self.out)
        manifest = support.read_json(os.path.join(final, "manifest.failed.json"))
        self.assertFalse(manifest["provenance_valid"])
        self.assertEqual(manifest["error_code"], "SourceChanged")
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        self.assertEqual(journal["entries"][2]["status"], "failed")
        self.assertEqual(journal["entries"][2]["error"]["error_code"], "SourceChanged")

    def test_rejected_and_invalid_knobs(self):
        code, err = self.run_runner([], self.sweep_env(THREADS="1"))
        self.assertEqual((code, "RejectedKnob" in err), (2, True))
        code, err = self.run_runner([], self.sweep_env(PSIZE="1"))
        self.assertEqual((code, "ModeConflict" in err), (2, True))
        code, err = self.run_runner([], self.env(SWEEP="1", OUTPUT_PARENT=self.out))
        self.assertEqual((code, "BACKEND" in err), (2, True))
        code, err = self.run_runner([], self.env(SWEEP="1", BACKEND="zinc",
                                                 OUTPUT_PARENT=self.out))
        self.assertEqual((code, "InvalidKnob" in err), (2, True))
        code, err = self.run_runner([], self.sweep_env(HASHES="10"))
        self.assertEqual((code, "RejectedKnob" in err), (2, True))
        code, err = self.run_runner([], self.env(SWEEP="1", BACKEND="hyrax",
                                                 OUTPUT_PARENT=os.path.join(self.tmp, "missing")))
        self.assertEqual(code, pc.ERROR_CODES["OutputParentInvalid"])
        code, err = self.run_runner([], self.env(SWEEP="1", BACKEND="hyrax",
                                                 OUTPUT_PARENT=self.repo))
        self.assertEqual(code, pc.ERROR_CODES["PathOverlap"])
        code, err = self.run_runner([], self.env(BACKEND="hyrax"))
        self.assertEqual((code, "UsageError" in err), (2, True))
        for name in ("PYTHONPYCACHEPREFIX", "RUSTFLAGS", "CARGO_PROFILE_BENCH_LTO",
                     "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"):
            code, err = self.run_runner([], self.sweep_env(**{name: "x"}))
            self.assertEqual((code, "RejectedEnvironment" in err), (2, True), name)
        self.assertEqual(os.listdir(self.out), [])

    def test_null_compiled_id_is_recorded_failure(self):
        md = support.write_metadata(self.tmp, self.repo, timing_schema_id=None)
        self.stub(metadata_json=md)
        code, _ = self.run_runner([], self.sweep_env())
        self.assertEqual(code, 0)
        final = only_dir(self.out)
        manifest = support.read_json(os.path.join(final, "manifest.json"))
        self.assertIn('spec_gate_failed:{"file": "timing-schema-v2.json", "reason": '
                      '"compiled_id_null"}', manifest["failure_reasons"])
        self.assertFalse(manifest["execution_valid"])
        cfg = support.read_json(os.path.join(final, "run-config.json"))
        check = cfg["ids"]["spec_checks"]["timing-schema-v2.json"]
        self.assertEqual((check["compiled_id"], check["match"]), (None, False))
        self.assertEqual(list(support.read_json(os.path.join(final, "gate-results.json"))),
                         ["kat", "tune_corpus"])

    def test_metadata_value_and_key_requirements(self):
        md = support.write_metadata(self.tmp, self.repo, overflow_checks=True, checked_flag=True,
                                    comparison_schema_id="0" * 64,
                                    diagnostic_groups=["setup"])
        self.stub(metadata_json=md)
        code, _ = self.run_runner([], self.sweep_env())
        self.assertEqual(code, 0)
        manifest = support.read_json(os.path.join(only_dir(self.out), "manifest.json"))
        self.assertIn("metadata_overflow_checks_invalid:true", manifest["failure_reasons"])
        self.assertIn("metadata_checked_flag_not_applicable:true", manifest["failure_reasons"])
        self.assertIn('metadata_diagnostic_groups_invalid:["setup"]',
                      manifest["failure_reasons"])
        self.assertIn('spec_gate_failed:{"file": "comparison-schema-v2.json", "reason": '
                      '"mismatch"}', manifest["failure_reasons"])
        self.assertEqual(manifest["eligible_for"], [])
        raw = support.metadata_for(self.repo)
        del raw["dimensions_by_k"]
        with open(md, "w") as f:
            json.dump(raw, f)
        out2 = os.path.join(self.tmp, "out2")
        os.mkdir(out2)
        code, err = self.run_runner([], self.env(SWEEP="1", BACKEND="hyrax",
                                                 OUTPUT_PARENT=out2))
        self.assertEqual(code, pc.ERROR_CODES["MetadataMalformed"], err)
        self.assertEqual(os.listdir(out2), [])
        # A candidate of the (shrunken) TUNE-1 list without dimensions is fatal too.
        raw = support.metadata_for(self.repo)
        del raw["dimensions_by_k"]["9"]
        with open(md, "w") as f:
            json.dump(raw, f)
        code, err = self.run_runner([], self.env(SWEEP="1", BACKEND="hyrax",
                                                 OUTPUT_PARENT=out2))
        self.assertEqual(code, pc.ERROR_CODES["MetadataMalformed"], err)
        self.assertIn("k = 9", err)
        self.assertEqual(os.listdir(out2), [])

    def test_capability_tuple_checks(self):
        md = support.write_metadata(self.tmp, self.repo, protocol_wire_id="limber/wire/v9",
                                    common_lift_binding_id="exact-common-lift-v1")
        self.stub(metadata_json=md)
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, 0, err)
        manifest = support.read_json(os.path.join(only_dir(self.out), "manifest.json"))
        codes = [r.split(":")[0] for r in manifest["failure_reasons"]]
        self.assertIn("capability_tuple_not_approved", codes)
        self.assertIn("common_lift_not_applicable", codes)
        self.assertEqual(manifest["eligible_for"], [])
        shutil.rmtree(only_dir(self.out))
        md = support.write_metadata(self.tmp, self.repo, prime_sampler_id="limber-prime-v0")
        self.stub(metadata_json=md)
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, 0, err)
        manifest = support.read_json(os.path.join(only_dir(self.out), "manifest.json"))
        codes = [r.split(":")[0] for r in manifest["failure_reasons"]]
        self.assertIn("sampler_not_p0d", codes)
        self.assertIn("capability_tuple_not_approved", codes)

    def test_missing_security_spec_is_hard_error(self):
        os.unlink(os.path.join(self.repo, "specs", "poseidon", "security-accounting-v1.json"))
        support.git(self.repo, "commit", "-qam", "drop spec")
        code, err = self.run_runner([], self.sweep_env())
        self.assertEqual(code, pc.ERROR_CODES["SpecMissing"])
        self.assertIn("security-accounting-v1.json is absent", err)
        self.assertEqual(os.listdir(self.out), [])


class EligibilityTests(unittest.TestCase):
    def test_pure_role_gates(self):
        ctx = runner.Context("sweep", {}, "/nonexistent", "hyrax")
        ctx.config = {"workload": {"hashes_per_field": 10, "threads": 1, "instance": None},
                      "build": {"parallel": False}, "features": []}
        v = runner.evaluate(ctx, "complete", 0)
        self.assertEqual(v["eligible_for"], ["tuning"])
        ctx.reason("compiled_metadata_differs", "execution")
        self.assertEqual(runner.evaluate(ctx, "complete", 0)["eligible_for"], [])
        ctx.reasons = []
        ctx.config["workload"]["threads"] = 8
        self.assertEqual(runner.evaluate(ctx, "complete", 0)["eligible_for"], [])
        ctx.config["workload"]["threads"] = 1
        self.assertEqual(runner.evaluate(ctx, "failed", 9)["eligible_for"], [])
        ctx.mode = "normal"
        ctx.config["workload"]["instance"] = 0
        ctx.config["comparison_key"] = "k"
        self.assertEqual(runner.evaluate(ctx, "complete", 0)["eligible_for"],
                         ["timing_table", "size_table"])
        ctx.reason("size_fields_differ_from_normal", "size")
        self.assertEqual(runner.evaluate(ctx, "complete", 0)["eligible_for"], ["timing_table"])
        ctx.reasons = []
        ctx.reason("SourceChanged", "provenance")
        self.assertFalse(runner.evaluate(ctx, "complete", 0)["provenance_valid"])


class PreflightValidationTests(unittest.TestCase):
    """The limber preflight/proof-size predicates (contract section 4)."""

    def preflight(self, **patch):
        records = [{"purpose": "RuntimeP", "width_bits": 128, "candidates": 40,
                    "bases_accepted": 72, "bases_rejected": 0, "mr_rounds_completed": 72,
                    "rolling_digest": "a" * 64, "outcome": {"success": "01" * 16}}]
        obj = {"schema": runner.PREFLIGHT_SCHEMA, "config_sha256": "c" * 64, "status": "ok",
               "error": None, "rayon_threads_asserted": 1, "backend": "hyrax", "k": 9,
               "hashes_per_field": 10, "instance": 0,
               "prime_audit": {"complete": True, "prime_sampler_id": pc.PRIME_SAMPLER_ID,
                               "invocations": 1, "records": records},
               "double_construction": {"commitments_equal": True, "components_equal": True,
                                       "remainder_equal": True, "audits_equal": True},
               "digests": ["1" * 64, "2" * 64, "3" * 64], "canonical_io_ok": True,
               "shape_satisfiable": True}
        obj.update(patch)
        return obj

    def test_preflight_conjuncts(self):
        expected = {"backend": "hyrax", "k": 9, "hashes_per_field": 10, "instance": 0}
        ok = self.preflight()
        self.assertEqual(runner.validate_preflight(ok, expected, "c" * 64, 1,
                                                   pc.PRIME_SAMPLER_ID), [])
        self.assertEqual(runner.validate_preflight({"schema": "x"}, expected, "c" * 64, 1,
                                                   pc.PRIME_SAMPLER_ID), ["schema"])
        cases = {
            "rayon_threads_asserted": self.preflight(rayon_threads_asserted=2),
            "prime_audit_complete": self.preflight(prime_audit=dict(
                ok["prime_audit"], complete=False)),
            "prime_audit_sampler_id": self.preflight(prime_audit=dict(
                ok["prime_audit"], prime_sampler_id="prime-v2")),
            "prime_audit_invocations": self.preflight(prime_audit=dict(
                ok["prime_audit"], invocations=2)),
            "prime_audit_records": self.preflight(prime_audit=dict(
                ok["prime_audit"], records=[], invocations=0)),
            "double_construction_audits_equal": self.preflight(double_construction=dict(
                ok["double_construction"], audits_equal=False)),
            "canonical_io_ok": self.preflight(canonical_io_ok=False),
            "shape_satisfiable": self.preflight(shape_satisfiable=None),
            "digests": self.preflight(digests=["1" * 64]),
            "expected_k": self.preflight(k=10),
            "status": self.preflight(status="failed"),
            "config_sha256": self.preflight(config_sha256="d" * 64),
        }
        for conjunct, obj in cases.items():
            failed = runner.validate_preflight(obj, expected, "c" * 64, 1, pc.PRIME_SAMPLER_ID)
            self.assertIn(conjunct, failed, conjunct)
        self.assertIn("prime_audit_sampler_id", runner.validate_preflight(
            ok, expected, "c" * 64, 1, "other-sampler"))
        self.assertIn("rayon_threads_asserted", runner.validate_preflight(
            ok, expected, "c" * 64, 4, pc.PRIME_SAMPLER_ID))
        broken = self.preflight()
        del broken["prime_audit"]["records"][0]["rolling_digest"]
        self.assertIn("prime_audit_record_0", runner.validate_preflight(
            broken, expected, "c" * 64, 1, pc.PRIME_SAMPLER_ID))

    def test_proof_size_conjuncts(self):
        with open(os.path.join(support.FIXTURES, "proof-size-sample.json"), "rb") as f:
            sample = pc.load_canonical_json(f.read())
        expected = {"backend": "hyrax", "k": 9, "hashes_per_field": 10, "instance": 0}
        pre = {"config_sha256": "0" * 64}
        self.assertEqual(runner.validate_proof_size(sample, expected, "1" * 64, pre), [])
        self.assertIn("comparison_key", runner.validate_proof_size(sample, expected, "2" * 64,
                                                                   pre))
        self.assertIn("metric_kind", runner.validate_proof_size(
            dict(sample, metric_kind="wire_bytes"), expected, "1" * 64, pre))
        self.assertIn("verified", runner.validate_proof_size(
            dict(sample, verified=False), expected, "1" * 64, pre))
        self.assertIn("components_commitments_bytes", runner.validate_proof_size(
            dict(sample, components=dict(sample["components"], commitments_bytes="12")),
            expected, "1" * 64, pre))
        self.assertIn("components", runner.validate_proof_size(
            dict(sample, components={}), expected, "1" * 64, pre))
        self.assertIn("analytical_remainder", runner.validate_proof_size(
            dict(sample, analytical_remainder=None), expected, "1" * 64, pre))
        self.assertIn("expected_backend", runner.validate_proof_size(
            sample, dict(expected, backend="brakedown"), "1" * 64, pre))
        self.assertIn("differs_from_preflight_config", runner.validate_proof_size(
            sample, expected, "1" * 64, {"config_sha256": "9" * 64}))


class Tune1Tests(unittest.TestCase):
    """TUNE-1 with the limber orders: schedule order (10, 7, 9), simplicity order (7, 9, 10)."""

    CANDS = [10, 7, 9]
    SIMPLE = [7, 9, 10]

    def processes(self, ratio_9, ratio_10, blocks=6):
        procs = {}
        for i in range(1, 10):
            for b in range(blocks):
                procs[(7, i, b)] = {"point": 1000 + b, "lower": 990, "upper": 1010}
                r9 = ratio_9[i - 1] if isinstance(ratio_9, list) else ratio_9
                r10 = ratio_10[i - 1] if isinstance(ratio_10, list) else ratio_10
                procs[(9, i, b)] = {"point": (1000 + b) * r9, "lower": 990 * r9,
                                    "upper": 1010 * r9}
                procs[(10, i, b)] = {"point": (1000 + b) * r10, "lower": 990 * r10,
                                     "upper": 1010 * r10}
        return procs

    def test_cases(self):
        from fractions import Fraction as F
        insts = range(1, 10)
        d = t1.decide(self.CANDS, insts, self.processes(F("1.05"), F("1.08")), self.SIMPLE)
        self.assertEqual((d["simplest"], d["selection_basis"], d["selected"]),
                         (7, "simplest_unique_best", 7))
        d = t1.decide(self.CANDS, insts, self.processes(F("1.02"), F("1.08")), self.SIMPLE)
        self.assertEqual((d["selection_basis"], d["selected"], d["band"]),
                         ("simplicity_tie_band", 7, [7, 9]))
        self.assertTrue(d["tie_band_non_singleton"])
        d = t1.decide(self.CANDS, insts, self.processes(F("0.9"), F("0.95")), self.SIMPLE)
        self.assertEqual((d["selection_basis"], d["selected"]), ("performance", 9))
        self.assertEqual(d["wins_vs_simplest"][9], 9)
        # The band's simplest member is chosen even when a larger k is marginally faster.
        d = t1.decide(self.CANDS, insts, self.processes(F("0.9"), F("0.89")), self.SIMPLE)
        self.assertEqual((d["selection_basis"], d["selected"], d["band"]),
                         ("performance", 9, [9, 10]))
        d = t1.decide(self.CANDS, insts, self.processes([F("0.5")] * 6 + [F("3")] * 3,
                                                        F("1.5")), self.SIMPLE)
        self.assertEqual(d["selection_basis"], "simplicity_insufficient_wins")
        d = t1.decide([9], insts, {(9, i, b): {"point": 1, "lower": 1, "upper": 1}
                                   for i in insts for b in range(2)})
        self.assertEqual((d["selection_basis"], d["selected"]), ("only_admissible", 9))
        # Band edge: exactly 1.03 is inside the band.
        d = t1.decide(self.CANDS, insts, self.processes(F("1.03"), F("1.5")), self.SIMPLE)
        self.assertEqual(d["band"], [7, 9])
        # The simplicity order must be a permutation of the candidates.
        with self.assertRaises(pc.RunnerError):
            t1.decide(self.CANDS, insts, self.processes(F(1), F(1)), [7, 9])
        # Without an explicit simplicity order the candidate order is the simplicity order
        # (s0 = 10; k = 7 is then the unique band member and wins on performance).
        d = t1.decide(self.CANDS, insts, self.processes(F("1.05"), F("1.08")))
        self.assertEqual((d["simplest"], d["selected"], d["selection_basis"]),
                         (10, 7, "performance"))

    def test_result_round_trip_and_child_names(self):
        from fractions import Fraction as F
        insts = list(range(1, 10))
        result = t1.tuning_result({"backend": "hyrax"}, {"x": 1}, self.CANDS, insts,
                                  self.processes(F("1.05"), F("1.08")), {"p": 1},
                                  simplicity_order=self.SIMPLE)
        data = pc.canonical_json_bytes(result)
        t1.recheck_tuning_result(pc.load_canonical_json(data))
        self.assertEqual(result["candidate_parameter"], "k")
        self.assertEqual(result["blocks"][1]["order"], [7, 9, 10])
        self.assertEqual(result["blocks"][3]["order"], [9, 7, 10])
        tampered = pc.load_canonical_json(data)
        tampered["simplicity_order"] = [10, 9, 7]
        with self.assertRaises(pc.RunnerError):
            t1.recheck_tuning_result(tampered)
        self.assertEqual(t1.child_dir_name(0, 7, 1), "00-k7-inst1")
        self.assertEqual(t1.child_dir_name(62, 13, 9), "62-k13-inst9")


class NormalAndPsizeTests(RunnerTestBase):
    def make_store_from_sweep(self, backend="hyrax"):
        code, err = self.run_runner([], self.sweep_env(backend))
        self.assertEqual(code, 0, err)
        sweep = only_dir(self.out)
        store = os.path.join(self.tmp, "store")
        os.mkdir(store)
        sha = pc.read_detached_digest(os.path.join(sweep, "manifest.sha256"))
        shutil.copytree(sweep, os.path.join(store, sha))
        shutil.rmtree(sweep)
        with open(os.path.join(store, sha, "run-config.json"), "rb") as f:
            cfg = pc.load_canonical_json(f.read())
        tuning_id = pc.read_detached_digest(os.path.join(store, sha, "tuning-result.sha256"))
        bundle = {"schema": "limber/poseidon-tuning-bundle/v1",
                  "source_sha": cfg["source"]["git_sha"],
                  "source_snapshot_id": cfg["source"]["source_snapshot_id"],
                  "lockfile_sha256": cfg["lockfile"]["sha256"],
                  "generated_module_path": support.GENERATED_MODULE, "group_set": "limber-both",
                  "groups": [{"backend": backend, "tuning_id": tuning_id,
                              "origin_manifest_sha256": sha, "selected": 7,
                              "dependency_source_id":
                                  cfg["dependency_sources"]["dependency_source_id"]}]}
        bundle_path = os.path.join(self.tmp, "tuning-bundle.json")
        pc.write_canonical_json(bundle_path, bundle)
        # The epoch commit: exactly the generated defaults module rendered from the bundle.
        pc.write_bytes(os.path.join(self.repo, support.GENERATED_MODULE),
                       apply_tuning.render(bundle))
        support.git(self.repo, "commit", "-qam", "apply tuning epoch")
        # From here on the (stub) bench is compiled with this epoch's generated defaults.
        self.stub(metadata_json=support.tuned_metadata(self.tmp, self.repo, bundle_path,
                                                       name="tuned-metadata.json"))
        return store, bundle_path, sha, tuning_id

    def write_session_files(self, envelope, obj, name):
        data = pc.write_canonical_json(os.path.join(envelope, name + ".json"), obj)
        pc.write_detached_digest(os.path.join(envelope, name + ".sha256"), pc.sha256_hex(data))
        return pc.sha256_hex(data)

    def normal_env(self, envelope, store, bundle_path, backend="hyrax", **extra):
        return self.env(BACKEND=backend, TUNING_BUNDLE=bundle_path, TUNING_STORE=store,
                        SESSION_SPEC=os.path.join(envelope, "comparison-session-config.json"),
                        **extra)

    def preconfigure(self, work, backend="hyrax", **knobs):
        env = self.env(BACKEND=backend, **knobs)
        code, err = self.run_runner(["normal", "preconfigure", "--staging", work], env)
        self.assertEqual(code, 0, err)
        with open(os.path.join(work, "preconfigure.json"), "rb") as f:
            return pc.load_canonical_json(f.read())

    def test_normal_flow_then_psize(self):
        store, bundle_path, origin_sha, tuning_id = self.make_store_from_sweep()
        sessions = os.path.join(self.tmp, "sessions")
        os.mkdir(sessions)
        envelope = os.path.join(sessions, "session-20260908T000000.000000000Z-abcdefabcdef"
                                ".staging")
        os.makedirs(os.path.join(envelope, "systems"))
        session_id = self.write_session_files(envelope, SESSION_SPEC,
                                              "comparison-session-config")
        staging = os.path.join(envelope, "systems", "hyrax")
        work = os.path.join(self.tmp, "work", "hyrax")
        os.mkdir(os.path.dirname(work))
        # Pre-configuration happens before the session config exists.
        pre = self.preconfigure(work)
        self.assertEqual(sorted(os.listdir(work)),
                         ["build-profile.json", "build-profile.log", "dependency-sources.json",
                          "preconfigure.json", "state.json"])
        for key in ("source_snapshot_id", "dependency_source_id", "lockfile_sha256",
                    "compiled", "executable", "role_target", "features", "threads", "dirty",
                    "git_sha"):
            self.assertIn(key, pre)
        self.assertEqual((pre["system_role"], pre["backend"], pre["repository"]),
                         ("hyrax", "hyrax", "limber-impl"))
        self.assertEqual(pre["schema"], "limber/poseidon-preconfigure/v2")
        self.assertTrue(os.path.isdir(pre["role_target"]))
        # SESSION_SPEC is a required environment input of `normal prepare`.
        code, err = self.run_runner(["normal", "prepare", "--staging", staging,
                                     "--preconfigured", work],
                                    self.env(BACKEND="hyrax", TUNING_BUNDLE=bundle_path,
                                             TUNING_STORE=store))
        self.assertEqual((code, "SESSION_SPEC" in err), (2, True))
        # A different BACKEND than the pre-configured one is refused.
        code, err = self.run_runner(["normal", "prepare", "--staging", staging,
                                     "--preconfigured", work],
                                    self.normal_env(envelope, store, bundle_path, "brakedown"))
        self.assertEqual(code, pc.ERROR_CODES["OutputParentInvalid"], err)
        env = self.normal_env(envelope, store, bundle_path)
        # The runner renders the generated module from the bundle itself (no
        # --rendered-defaults from the orchestrator for limber).
        code, err = self.run_runner(["normal", "prepare", "--staging", staging,
                                     "--preconfigured", work], env)
        self.assertEqual(code, 0, err)
        cfg = support.read_json(os.path.join(staging, "run-config.json"))
        self.assertEqual(cfg["session_id"], session_id)
        self.assertEqual((cfg["tuning_id"], cfg["k"], cfg["backend"]), (tuning_id, 7, "hyrax"))
        self.assertEqual(cfg["workload"], {"backend": "hyrax", "k": 7, "hashes_per_field": 10,
                                           "instance": 0, "threads": 1})
        self.assertEqual(cfg["dimensions"], {"log_cons": 14, "log_vars": 14})
        self.assertEqual(cfg["tuning"], {"tuning_id": tuning_id,
                                         "tuning_epoch_id": cfg["tuning_epoch_id"]})
        self.assertTrue(cfg["tuning_origin"]["generated_defaults_reproduced"])
        self.assertEqual(cfg["tuning_origin"]["origins_validated"], ["hyrax"])
        self.assertEqual(cfg["tuning_origin"]["source_diff_from_epoch"],
                         [support.GENERATED_MODULE])
        self.assertEqual(cfg["tuning_origin"]["group_set"], "limber-both")
        self.assertEqual(cfg["source"]["source_snapshot_id"], pre["source_snapshot_id"])
        self.assertEqual(cfg["session_schedule"]["diagnostic_order"],
                         list(pc.diagnostic_groups_for("hyrax")))
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        self.assertEqual([e["name"] for e in journal["entries"][:4]],
                         ["evidence_metadata", "kat", "tune_corpus", "preflight"])
        self.assertEqual([e["status"] for e in journal["entries"]],
                         ["ok"] + ["not_run"] * 16)
        self.assertEqual(journal["failure_reasons"], [])
        gates = support.read_json(os.path.join(staging, "gate-results.json"))
        self.assertEqual([gates[n]["status"] for n in ("kat", "tune_corpus")],
                         ["not_run", "not_run"])
        self.assertIsNone(gates["kat"]["exit"])
        self.assertEqual(sorted(gates["kat"]), GATE_KEYS)
        # Every later step needs the same BACKEND; children before the preflight are
        # refused; the preflight before the gates is refused.
        code, _ = self.run_runner(["normal", "gate", "--staging", staging, "--gate", "kat"],
                                  self.normal_env(envelope, store, bundle_path, "brakedown"))
        self.assertEqual(code, pc.ERROR_CODES["OutputParentInvalid"])
        code, _ = self.run_runner(["normal", "child", "--staging", staging, "--kind",
                                   "diagnostic", "--order", ORDER, "--ordinal", "38"], env)
        self.assertEqual(code, pc.ERROR_CODES["ChildFailed"])
        code, _ = self.run_runner(["normal", "preflight", "--staging", staging], env)
        self.assertEqual(code, pc.ERROR_CODES["PreflightFailed"])
        for gate in ("kat", "tune_corpus"):
            code, err = self.run_runner(["normal", "gate", "--staging", staging, "--gate", gate],
                                        env)
            self.assertEqual(code, 0, err)
        code, err = self.run_runner(["normal", "preflight", "--staging", staging], env)
        self.assertEqual(code, 0, err)
        size = support.read_json(os.path.join(staging, "proof-size.json"))
        self.assertEqual(size["schema"], "limber/poseidon-proof-size/v2")
        self.assertEqual(size["metric_kind"], "mixed_payload_estimate")
        self.assertEqual(size["comparison_key"], cfg["comparison_key"])
        self.assertTrue(size["verified"])
        pre_json = support.read_json(os.path.join(staging, "preflight.json"))
        self.assertEqual(pre_json["config_sha256"], pc.read_detached_digest(
            os.path.join(staging, "run-config.sha256")))
        self.assertEqual(pre_json["rayon_threads_asserted"], 1)
        self.assertEqual(pre_json["prime_audit"]["invocations"],
                         len(pre_json["prime_audit"]["records"]))
        # Direct normal invocation without the orchestrator is rejected.
        code, err = self.run_runner([], self.env(BACKEND="hyrax"))
        self.assertEqual(code, 2)
        # 12 primary children + 1 diagnostic, in session order.
        ordinal = 2
        for metric in pc.PRIMARY_GROUPS:
            for b, order in enumerate(pc.SESSION_BLOCK_ORDERS):
                code, err = self.run_runner(["normal", "child", "--staging", staging, "--kind",
                                             "primary", "--metric", metric, "--block", str(b),
                                             "--order", json.dumps(list(order)), "--ordinal",
                                             str(ordinal)], env)
                self.assertEqual(code, 0, err)
                ordinal += 3
        code, err = self.run_runner(["normal", "child", "--staging", staging, "--kind",
                                     "diagnostic", "--order", ORDER, "--ordinal", "38"], env)
        self.assertEqual(code, 0, err)
        # Re-running an imported coordinate is refused.
        code, err = self.run_runner(["normal", "child", "--staging", staging, "--kind",
                                     "diagnostic", "--order", ORDER, "--ordinal", "38"], env)
        self.assertEqual(code, pc.ERROR_CODES["ChildFailed"])
        diag = os.path.join(staging, "children", "diagnostic")
        result = support.read_json(os.path.join(diag, "child-result.json"))
        self.assertEqual(len(result["criterion_files"]), 8 * 4)  # hyrax: four groups
        groups = sorted({f[0].split("/")[1] for f in result["criterion_files"]})
        self.assertEqual(groups, sorted(pc.diagnostic_groups_for("hyrax")))
        cc = support.read_json(os.path.join(diag, "child-config.json"))
        self.assertEqual(sorted(cc), CHILD_CONFIG_KEYS)
        self.assertEqual((cc["session_id"], cc["ordinal"]), (session_id, 13))
        self.assertEqual(cc["order"], ["zinc", "hyrax", "brakedown"])
        self.assertTrue(os.path.isdir(os.path.join(staging, "children", "primary",
                                                   "verify_core", "block-5")))
        prim = support.read_json(os.path.join(staging, "children", "primary", "prove_e2e",
                                              "block-0", "child-result.json"))
        self.assertEqual(len(prim["criterion_files"]), 8)
        self.assertIn("k7_inst0_thr1_primary_blk0", prim["criterion_files"][0][0])
        # Seal after the orchestrator writes the session result.
        entries = [{"system": s, "status": "ok"} for s in ["zinc", "hyrax", "brakedown"]
                   for _ in range(pc.NORMAL_CHILDREN)]
        result_id = self.write_session_files(envelope, {"session_id": session_id,
                                                        "entries": entries},
                                             "comparison-session-result")
        code, err = self.run_runner(["normal", "seal", "--staging", staging,
                                     "--session-result",
                                     os.path.join(envelope, "comparison-session-result.json")],
                                    env)
        self.assertEqual(code, 0, err)
        self.assertFalse(os.path.exists(pre["role_target"]))
        required, globs = runner.expected_payload_set("normal", "complete")
        manifest, hyrax_sha, status = af.verify_artifact_dir(staging, required, globs)
        self.assertEqual(status, "complete")
        self.assertEqual(manifest["schema"], "limber/poseidon-run-manifest/v2")
        self.assertEqual(manifest["system_role"], "hyrax")
        self.assertEqual(manifest["failure_reasons"], [])
        self.assertEqual(manifest["eligible_for"], ["timing_table", "size_table"])
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        self.assertEqual(len(journal["entries"]), 17)
        self.assertTrue(all(e["status"] == "ok" for e in journal["entries"]))
        self.assertEqual(journal["entries"][4]["session_ordinal"], 2)
        # Build the enclosing session root (orchestrator's job) around the nested archive.
        by_role = {"hyrax": hyrax_sha}
        for role in ("zinc", "brakedown"):
            d = os.path.join(envelope, "systems", role)
            os.mkdir(d)
            pc.write_bytes(os.path.join(d, "artifacts.sha256"), b"")
            by_role[role] = af.seal_manifest(d, {"status": "complete", "artifact_index_sha256":
                                                 pc.sha256_hex(b"")})
        lines = []
        for rel in sorted(list(runner.SESSION_FILES) + ["systems/%s/%s" % (r, n)
                                                         for r in by_role
                                                         for n in ("artifacts.sha256",
                                                                   "manifest.json",
                                                                   "manifest.sha256")]):
            full = os.path.join(envelope, rel)
            lines.append(af.format_index_line(pc.sha256_file(full), os.path.getsize(full), rel))
        index = "".join(lines).encode()
        pc.write_bytes(os.path.join(envelope, "session-artifacts.sha256"), index)
        sm = {"status": "complete", "session_id": session_id, "session_result_id": result_id,
              "artifact_index_sha256": pc.sha256_hex(index),
              "parent_manifest_sha256_by_role": by_role}
        self.write_session_files(envelope, sm, "session-manifest")
        final_session = envelope[:-len(".staging")]
        af.commit_staging(envelope, final_session)
        normal_run = os.path.join(final_session, "systems", "hyrax")
        # psize reproduction against the linked normal archive (BACKEND is rejected: the
        # role comes from the linked archive).
        out2 = os.path.join(self.tmp, "out2")
        os.mkdir(out2)
        env = self.env(PSIZE="1", NORMAL_RUN=normal_run, TUNING_STORE=store, OUTPUT_PARENT=out2,
                       BACKEND="hyrax")
        code, err = self.run_runner([], env)
        self.assertEqual((code, "RejectedKnob" in err), (2, True))
        env = self.env(PSIZE="1", NORMAL_RUN=normal_run, TUNING_STORE=store, OUTPUT_PARENT=out2)
        code, err = self.run_runner([], env)
        self.assertEqual(code, 0, err)
        final = only_dir(out2)
        required, globs = runner.expected_payload_set("psize", "complete")
        manifest, _, status = af.verify_artifact_dir(final, required, globs)
        self.assertEqual(status, "complete")
        self.assertEqual(manifest["system_role"], "hyrax")
        self.assertEqual(manifest["failure_reasons"], [])
        self.assertEqual(manifest["eligible_for"], ["size_table"])
        link = support.read_json(os.path.join(final, "normal-link.json"))
        self.assertEqual(sorted(link), ["comparison_key", "normal_manifest_sha256",
                                        "session_id", "session_manifest_sha256",
                                        "session_result_id", "system_role"])
        self.assertEqual(link["normal_manifest_sha256"], hyrax_sha)
        self.assertEqual(link["system_role"], "hyrax")
        self.assertEqual(link["session_id"], session_id)
        self.assertEqual(link["session_result_id"], result_id)
        self.assertEqual(link["comparison_key"], cfg["comparison_key"])
        self.assertNotIn("/", json.dumps(link))
        journal = support.read_json(os.path.join(final, "process-journal.json"))
        self.assertEqual([e["name"] for e in journal["entries"]],
                         ["evidence_metadata", "kat", "tune_corpus", "preflight"])
        psize_size = support.read_json(os.path.join(final, "proof-size.json"))
        normal_size = support.read_json(os.path.join(normal_run, "proof-size.json"))
        self.assertNotEqual(psize_size.pop("config_sha256"), normal_size.pop("config_sha256"))
        self.assertEqual(psize_size, normal_size)
        # psize with a mismatched/unqualified normal link is rejected before any output.
        reseal(normal_run, eligible_for=[])
        out3 = os.path.join(self.tmp, "out3")
        os.mkdir(out3)
        env = self.env(PSIZE="1", NORMAL_RUN=normal_run, TUNING_STORE=store, OUTPUT_PARENT=out3)
        code, err = self.run_runner([], env)
        self.assertEqual(code, pc.ERROR_CODES["LinkInvalid"])
        self.assertEqual(os.listdir(out3), [])
        env = self.env(PSIZE="1", NORMAL_RUN=os.path.join(final_session, "systems", "zinc"),
                       TUNING_STORE=store, OUTPUT_PARENT=out3)
        code, err = self.run_runner([], env)
        self.assertEqual((code, "LinkInvalid" in err), (pc.ERROR_CODES["LinkInvalid"], True))

    def prepared_parent(self, store, bundle_path, spec=SESSION_SPEC, backend="hyrax", **knobs):
        envelope = os.path.join(self.tmp, "session-x.staging")
        os.makedirs(os.path.join(envelope, "systems"))
        session_id = self.write_session_files(envelope, spec, "comparison-session-config")
        staging = os.path.join(envelope, "systems", backend)
        work = os.path.join(self.tmp, "work-x")
        self.preconfigure(work, backend, **knobs)
        env = self.normal_env(envelope, store, bundle_path, backend, **knobs)
        code, err = self.run_runner(["normal", "prepare", "--staging", staging,
                                     "--preconfigured", work], env)
        self.assertEqual(code, 0, err)
        return envelope, staging, env, session_id

    def failed_seal(self, envelope, staging, env, session_id, stage, code_name):
        result_id = self.write_session_files(envelope, {"session_id": session_id,
                                                        "entries": []},
                                             "comparison-session-result")
        failure = os.path.join(self.tmp, "failure-info.json")
        pc.write_canonical_json(failure, {"stage": stage, "error_code": code_name,
                                          "message": "hyrax %s" % stage,
                                          "coordinate": {"kind": stage, "system": "hyrax"}})
        code, err = self.run_runner(["normal", "seal", "--staging", staging,
                                     "--session-result",
                                     os.path.join(envelope, "comparison-session-result.json"),
                                     "--failure", failure], env)
        self.assertEqual(code, 0, err)
        required, globs = runner.expected_payload_set("normal", "failed")
        manifest, _, status = af.verify_artifact_dir(staging, required, globs,
                                                     confirm_durability=False)
        self.assertEqual((status, manifest["error_code"]), ("failed", code_name))
        self.assertEqual(sorted(os.listdir(staging)),
                         sorted(list(required) + ["artifacts.sha256", "manifest.failed.json",
                                                  "manifest.failed.sha256"]))
        self.assertEqual(pc.read_detached_digest(os.path.join(
            staging, "comparison-session-result.sha256")), result_id)
        return support.read_json(os.path.join(staging, "failure.json"))

    def test_normal_preflight_failure_then_failed_seal(self):
        store, bundle_path, _, _ = self.make_store_from_sweep()
        self.stub(fail_preflight="7:0")  # frozen into the closed environment
        envelope, staging, env, session_id = self.prepared_parent(store, bundle_path)
        for gate in ("kat", "tune_corpus"):
            self.assertEqual(self.run_runner(["normal", "gate", "--staging", staging, "--gate",
                                              gate], env)[0], 0)
        code, err = self.run_runner(["normal", "preflight", "--staging", staging], env)
        self.assertEqual(code, pc.ERROR_CODES["PreflightFailed"], err)
        self.assertIn("double_construction_components_equal", err)
        self.assertFalse(os.path.exists(os.path.join(staging, "proof-size.json")))
        failure = self.failed_seal(envelope, staging, env, session_id, "preflight",
                                   "PreflightFailed")
        self.assertEqual(failure["preflight_evidence"]["preflight"]["status"], "failed")
        self.assertEqual(failure["preflight_evidence"]["attempt_result"]["status"], "failed")
        self.assertIsNone(failure["preflight_evidence"]["proof_size"])
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        self.assertEqual([e["status"] for e in journal["entries"][:4]],
                         ["ok", "ok", "ok", "failed"])
        self.assertTrue(all(null_execution_fields(e) for e in journal["entries"][4:]))

    def test_normal_gate_failure_then_failed_seal(self):
        store, bundle_path, _, _ = self.make_store_from_sweep()
        self.stub(fail_gate="kat_gate")  # frozen into the closed environment
        envelope, staging, env, session_id = self.prepared_parent(store, bundle_path)
        code, err = self.run_runner(["normal", "gate", "--staging", staging, "--gate", "kat"],
                                    env)
        self.assertEqual(code, pc.ERROR_CODES["GateFailed"], err)
        self.failed_seal(envelope, staging, env, session_id, "gates", "GateFailed")
        gates = support.read_json(os.path.join(staging, "gate-results.json"))
        self.assertEqual((gates["kat"]["status"], gates["tune_corpus"]["status"]),
                         ("failed", "not_run"))
        self.assertIsNone(gates["tune_corpus"]["pre"])
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        self.assertTrue(null_execution_fields(journal["entries"][3]))

    def test_exploratory_threads_are_ineligible_without_features(self):
        store, bundle_path, _, _ = self.make_store_from_sweep()
        spec = dict(SESSION_SPEC, threads=8)
        envelope, staging, env, _ = self.prepared_parent(store, bundle_path, spec=spec,
                                                         THREADS="8")
        cfg = support.read_json(os.path.join(staging, "run-config.json"))
        self.assertEqual((cfg["features"], cfg["build"]["parallel"]), ([], False))
        self.assertEqual(cfg["environment"]["RAYON_NUM_THREADS"], "8")
        self.assertEqual(cfg["preflight_requirements"]["rayon_threads_asserted"], 8)
        self.assertNotIn("--features", cfg["build"]["evidence_command"])
        self.assertNotIn("--features", cfg["build"]["gate_commands"]["kat"])
        for gate in ("kat", "tune_corpus"):
            self.assertEqual(self.run_runner(["normal", "gate", "--staging", staging, "--gate",
                                              gate], env)[0], 0)
        self.assertEqual(self.run_runner(["normal", "preflight", "--staging", staging],
                                         env)[0], 0)
        pre = support.read_json(os.path.join(staging, "preflight.json"))
        self.assertEqual(pre["rayon_threads_asserted"], 8)
        code, err = self.run_runner(["normal", "child", "--staging", staging, "--kind",
                                     "primary", "--metric", "prove_e2e", "--block", "0",
                                     "--order", ORDER, "--ordinal", "2"], env)
        self.assertEqual(code, 0, err)
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        argv = journal["entries"][4]["argv"]
        self.assertNotIn("--features", argv)
        child = support.read_json(os.path.join(staging, "children", "primary", "prove_e2e",
                                               "block-0", "child-result.json"))
        self.assertIn("_thr8_primary_blk0", child["criterion_files"][0][0])
        codes = [r["code"] for r in journal["failure_reasons"]]
        self.assertIn("threads_not_one", codes)
        self.assertNotIn("compiled_features_mismatch", codes)
        self.assertNotIn("compiled_tuning_mismatch", codes)
        self.run_runner(["normal", "cleanup", "--staging", staging], env)

    def test_compiled_tuning_mismatch_is_recorded(self):
        store, bundle_path, _, _ = self.make_store_from_sweep()
        self.stub(metadata_json=self.metadata)  # pre-epoch bench: no tuned defaults
        envelope, staging, env, session_id = self.prepared_parent(store, bundle_path)
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        self.assertIn("compiled_tuning_mismatch", [r["code"] for r in journal["failure_reasons"]])
        for gate in ("kat", "tune_corpus"):
            self.assertEqual(self.run_runner(["normal", "gate", "--staging", staging, "--gate",
                                              gate], env)[0], 0)
        code, err = self.run_runner(["normal", "preflight", "--staging", staging], env)
        self.assertEqual(code, pc.ERROR_CODES["AttemptResultInvalid"], err)
        failure = self.failed_seal(envelope, staging, env, session_id, "preflight",
                                   "AttemptResultInvalid")
        self.assertIsNone(failure["preflight_evidence"]["attempt_result"])

    def test_generated_module_drift_is_recorded(self):
        store, bundle_path, _, _ = self.make_store_from_sweep()
        with open(os.path.join(self.repo, support.GENERATED_MODULE), "ab") as f:
            f.write(b"// drift\n")
        support.git(self.repo, "commit", "-qam", "drift")
        envelope, staging, env, _ = self.prepared_parent(store, bundle_path)
        journal = support.read_json(os.path.join(staging, "process-journal.json"))
        codes = [r["code"] for r in journal["failure_reasons"]]
        self.assertIn("generated_defaults_not_reproduced", codes)
        self.assertNotIn("non_generated_source_diff", codes)
        cfg = support.read_json(os.path.join(staging, "run-config.json"))
        self.assertFalse(cfg["tuning_origin"]["generated_defaults_reproduced"])
        self.run_runner(["normal", "cleanup", "--staging", staging], env)

    def test_bundle_with_wrong_origin_rejected(self):
        store, bundle_path, origin_sha, _ = self.make_store_from_sweep()
        with open(os.path.join(store, origin_sha, "sweep-metadata.json"), "ab") as f:
            f.write(b" ")
        envelope = os.path.join(self.tmp, "session-y.staging")
        os.makedirs(os.path.join(envelope, "systems"))
        self.write_session_files(envelope, SESSION_SPEC, "comparison-session-config")
        work = os.path.join(self.tmp, "work-y")
        self.preconfigure(work)
        env = self.normal_env(envelope, store, bundle_path)
        code, err = self.run_runner(["normal", "prepare", "--staging",
                                     os.path.join(envelope, "systems", "hyrax"),
                                     "--preconfigured", work], env)
        self.assertEqual(code, pc.ERROR_CODES["IntegrityMismatch"])
        self.assertFalse(os.path.exists(os.path.join(envelope, "systems", "hyrax")))
        self.run_runner(["normal", "cleanup", "--staging", work], env)

    def test_bundle_without_this_backend_rejected(self):
        store, bundle_path, origin_sha, _ = self.make_store_from_sweep()
        envelope = os.path.join(self.tmp, "session-z.staging")
        os.makedirs(os.path.join(envelope, "systems"))
        self.write_session_files(envelope, SESSION_SPEC, "comparison-session-config")
        work = os.path.join(self.tmp, "work-z")
        self.preconfigure(work, "brakedown")
        env = self.normal_env(envelope, store, bundle_path, "brakedown")
        code, err = self.run_runner(["normal", "prepare", "--staging",
                                     os.path.join(envelope, "systems", "brakedown"),
                                     "--preconfigured", work], env)
        self.assertEqual((code, "brakedown" in err), (pc.ERROR_CODES["BundleInvalid"], True))
        self.run_runner(["normal", "cleanup", "--staging", work], env)


if __name__ == "__main__":
    unittest.main()
