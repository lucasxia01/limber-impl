#!/usr/bin/env python3
"""limber Poseidon2 benchmark runner (Zinc plan v10, section 9; limber contract sections
1, 4, 5, 6: run modes, artifacts, predicate).

This is the limber adaptation of the Zinc+ `scripts/poseidon_runner.py`; the runner CLI
contract, the archive layout and every public function name are the same so that the Zinc
session orchestrator and publisher drive/consume it unchanged. The system-profile
differences (backends instead of variants, k instead of shapes, limber commands, metadata,
preflight and proof-size records, Criterion grammar) live in `poseidon_common`.

Single-shot modes (invoked through `scripts/run_poseidon_bench.sh`, which passes `"$@"`):

    SWEEP=1 BACKEND=hyrax OUTPUT_PARENT=/abs/out scripts/run_poseidon_bench.sh
    PSIZE=1 NORMAL_RUN=/abs/session-.../systems/hyrax TUNING_STORE=/abs/store \\
        OUTPUT_PARENT=/abs/out scripts/run_poseidon_bench.sh

Both run the pre-configuration phase (input/environment validation -> source snapshot ->
dependency-source audit and metadata replay in a disposable metadata target -> fresh role
target -> evidence build -> compiled metadata from that target -> rechecks), only then build
the immutable `run-config.json`, create `OUTPUT_PARENT/run-<time>-<config12>.staging`, copy
the evidence, run the two gates, the one preflight attempt (sweep: the complete
candidate x instance matrix in one bench attempt), the children, seal, and commit with one
no-replace rename. A failure before the config pair exists creates no archive; a later
failure is committed with the exact failed payload set.

Normal mode is driven by the Zinc session orchestrator through subcommands that share one
nested staging directory (`<envelope>.staging/systems/<role>`; the runner creates the leaf
exclusively and never renames it). `--repo` names the repository checkout (default: this
one); the role is the `BACKEND` knob (`hyrax` | `brakedown`):

    poseidon_runner.py --repo <abs> normal preconfigure --staging <work dir>
        env: BACKEND [THREADS] [TUNING_BUNDLE TUNING_STORE] [POSEIDON_ALLOW_DIRTY];
        rejected: NORMAL_RUN OUTPUT_PARENT SESSION_SPEC SWEEP=1 PSIZE=1.
        Creates <work dir> exclusively, performs the pre-configuration phase, keeps the
        role target/temp directories alive, writes `dependency-sources.json`,
        `build-profile.json`, `build-profile.log`, the raw `state.json` and
        `preconfigure.json` (system identity: source_snapshot_id, git sha/tree, lockfile
        sha256, dependency_source_id, toolchain/native closure hashes, compiled metadata,
        evidenced executable, features, threads, role target/temp locators) and prints
        `preconfigure <work dir>/preconfigure.json`.
    poseidon_runner.py --repo <abs> normal prepare --staging <nested> --preconfigured <work dir>
        [--rendered-defaults <path>]
        env: BACKEND TUNING_BUNDLE TUNING_STORE SESSION_SPEC [THREADS]
             [POSEIDON_ALLOW_DIRTY]; rejected: NORMAL_RUN OUTPUT_PARENT SWEEP=1 PSIZE=1;
             HASHES/INSTANCE may only be present as 10/0.
        Rechecks the snapshot/closures, validates the session spec (adjacent `.sha256`
        digest = session_id), the bundle and every origin archive, renders the generated
        defaults module from the bundle (or takes `--rendered-defaults`) and compares it with
        the tracked `src/poseidon_tuned_defaults.rs`, creates the nested staging, copies the
        evidence, lockfile, spec copies, bundle and session config, and writes the immutable
        `run-config.json`/`run-config.sha256` pair last, with the journal (four preparatory
        attempts + 13 `not_run` children) and `gate-results.json`.
    poseidon_runner.py --repo <abs> normal gate --staging <nested> --gate kat|tune_corpus
    poseidon_runner.py --repo <abs> normal preflight --staging <nested>
    poseidon_runner.py --repo <abs> normal child --staging <nested> --kind primary
        --metric <prove_e2e|verify_core> --block <0..5> --order '["zinc","hyrax","brakedown"]'
        --ordinal <session ordinal>
    poseidon_runner.py --repo <abs> normal child --staging <nested> --kind diagnostic
        --order '["zinc","hyrax","brakedown"]' --ordinal <session ordinal>
    poseidon_runner.py --repo <abs> normal seal --staging <nested> --session-result <path>
        [--failure <json path>]
        Copies the session result pair, writes the journal, artifact index and
        `manifest.json`/`manifest.sha256` (or, with `--failure`, removes every imported
        child tree and success payload, writes `failure.json` and the failed pair), then
        removes the role target/temp directories.
    poseidon_runner.py --repo <abs> normal cleanup --staging <work dir | nested>
        Removes the role target/temp directories recorded in that state (idempotent).

Bench contract (limber contract section 4, NOTES-for-rust.md): the bench takes no
environment inputs except `RAYON_NUM_THREADS`. Form 1 prints the compiled metadata; form 2
(`--run-config <abs> --config-sha256 <hex> --attempt preflight --artifact-dir <abs empty>`)
writes the preflight outputs plus `attempt-result.json`; form 3 (`--child-config <abs>
--child-config-sha256 <hex> --artifact-dir <abs empty>`) writes the Criterion tree under
`<dir>/criterion`. Every subprocess runs in the closed environment built by
`poseidon_build_env` with `RAYON_NUM_THREADS = T`; limber has no parallel feature, so
`T > 1` changes nothing but that variable and makes the run exploratory (ineligible).

Exit codes: 0 success; otherwise `poseidon_common.ERROR_CODES[error_code]`. Every error is
printed as `poseidon_runner: error [<stage>/<error_code>]: <message>` on stderr.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

try:
    from scripts import apply_poseidon_tuning as apply_tuning
    from scripts import artifact_fs as af
    from scripts import poseidon_common as pc
    from scripts import poseidon_tune1 as t1
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import apply_poseidon_tuning as apply_tuning  # noqa: E402
    import artifact_fs as af  # noqa: E402
    import poseidon_common as pc  # noqa: E402
    import poseidon_tune1 as t1  # noqa: E402

try:
    from scripts import poseidon_build_env as be
except ImportError:
    try:
        import poseidon_build_env as be  # noqa: F401
    except ImportError:
        be = None  # tests inject scripts/tests/stub_build_env.py; production requires C

# Schema names of every file the runner writes or the bench writes for it. The limber bench
# validates `RUN_CONFIG_SCHEMA` / `CHILD_CONFIG_SCHEMA` and writes the preflight, proof-size,
# sweep-metadata and attempt-result schemas (contract section 4).
RUN_CONFIG_SCHEMA = "limber/poseidon-run-config/v2"
JOURNAL_SCHEMA = "limber/poseidon-process-journal/v2"
MANIFEST_SCHEMA = "limber/poseidon-run-manifest/v2"
FAILURE_SCHEMA = "limber/poseidon-failure/v2"
CHILD_CONFIG_SCHEMA = "limber/poseidon-child-config/v2"
CHILD_RESULT_SCHEMA = "limber/poseidon-child-result/v2"
SWEEP_METADATA_SCHEMA = "limber/poseidon-sweep-metadata/v2"
PROOF_SIZE_SCHEMA = "limber/poseidon-proof-size/v2"
PREFLIGHT_SCHEMA = "limber/poseidon-preflight/v2"
ATTEMPT_RESULT_SCHEMA = "limber/poseidon-attempt-result/v2"
PRECONFIGURE_SCHEMA = "limber/poseidon-preconfigure/v2"
STATE_SCHEMA = "limber/poseidon-runner-state/v2"
PROOF_SIZE_METRIC_KIND = "mixed_payload_estimate"
PRIME_AUDIT_RECORD_KEYS = ("purpose", "width_bits", "candidates", "bases_accepted",
                           "bases_rejected", "mr_rounds_completed", "rolling_digest", "outcome")
DOUBLE_CONSTRUCTION_KEYS = ("commitments_equal", "components_equal", "remainder_equal",
                            "audits_equal")
PROOF_SIZE_COMPONENT_KEYS = ("commitments_bytes", "eval_arg_bytes", "commitments_sha256",
                             "eval_arg_sha256")
TMP_DIR = ".tmp"
STATE_FILE = "state.json"
SESSION_FILES = ("comparison-session-config.json", "comparison-session-config.sha256",
                 "comparison-session-result.json", "comparison-session-result.sha256")
EVIDENCE_FILES = ("dependency-sources.json", "build-profile.json", "build-profile.log")
COMMON_PAYLOADS = (("run-config.json", "run-config.sha256", "Cargo.lock.archived")
                   + EVIDENCE_FILES + ("bench.log", "process-journal.json", "gate-results.json")
                   + pc.SPEC_ORDER)
CHILD_FILES = ("child-config.json", "child-result.json", "bench.log")
MODE_SUCCESS = {
    "sweep": (("sweep-metadata.json", "tuning-result.json", "tuning-result.sha256"),
              tuple("children/block-*/*/" + f for f in CHILD_FILES)
              + ("children/block-*/*/criterion/*",)),
    "psize": (("preflight.json", "proof-size.json", "normal-link.json"), ()),
    "normal": (SESSION_FILES + ("preflight.json", "tuning-bundle.json", "proof-size.json"),
               tuple("children/primary/*/block-*/" + f for f in CHILD_FILES)
               + ("children/primary/*/block-*/criterion/*",)
               + tuple("children/diagnostic/" + f for f in CHILD_FILES)
               + ("children/diagnostic/criterion/*",)),
}
MODE_FAILED = {"sweep": (("failure.json",), ()), "psize": (("failure.json",), ()),
               "normal": (("failure.json",) + SESSION_FILES, ())}
PREFLIGHT_IMPORTS = {"normal": ("preflight.json", "proof-size.json"),
                     "psize": ("preflight.json", "proof-size.json"),
                     "sweep": ("sweep-metadata.json",)}
EXECUTION_FIELDS = ("argv", "pid", "exit_code", "signal", "started_utc", "ended_utc",
                    "monotonic_start_ns", "monotonic_end_ns", "loadavg_before", "loadavg_after",
                    "pre_audit", "post_audit", "log_sha256", "import_status",
                    "child_config_sha256", "child_result_sha256", "session_ordinal",
                    "attempt_result", "error")


def expected_payload_set(mode: str, status: str):
    extra, globs = (MODE_SUCCESS if status == "complete" else MODE_FAILED)[mode]
    return list(COMMON_PAYLOADS) + list(extra), list(globs)


def _be():
    if be is None:
        raise pc.RunnerError("toolchain", "ToolchainUnavailable",
                             "scripts/poseidon_build_env.py is not available")
    return be


def _overlaps(a: str, b: str) -> bool:
    a, b = a.rstrip("/"), b.rstrip("/")
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def _now_ns() -> int:
    return time.time_ns()


def _loadavg():
    try:
        return ["%.2f" % v for v in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _rmtree(path) -> bool:
    if path and os.path.lexists(path) and os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
        return True
    return False


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def fresh_temp_dir(prefix: str, disjoint_from) -> str:
    """`mktemp -d` outside the worktree/output roots, symlink-free, absolute (plan 1847)."""
    path = os.path.realpath(tempfile.mkdtemp(prefix=prefix))
    for other in disjoint_from:
        if other and _overlaps(path, os.path.realpath(other)):
            shutil.rmtree(path)
            raise pc.RunnerError("env", "TempDirInvalid",
                                 "temporary directory %s overlaps %s" % (path, other))
    if os.listdir(path):
        raise pc.RunnerError("env", "TempDirInvalid", "%s is not empty" % path)
    return path


# ---------------------------------------------------------------------------
# Context: everything one attempt knows, persisted through the payload files.
# ---------------------------------------------------------------------------

class Context:
    def __init__(self, mode: str, env, repo: str, role=None):
        self.mode = mode
        self.env = env
        self.repo = repo
        self.role = role
        self.knobs = {}
        self.threads = 1
        self.features = []
        self.timing_schema = None
        self.tuning_protocol = None
        self.rejected_inputs = None
        self.snapshot = None
        self.lock_bytes = None
        self.lock_sha = None
        self.toolchain = None
        self.native = None
        self.cargo_config = None
        self.home = None
        self.cargo_home = None
        self.role_temp = None
        self.role_target = None
        self.metadata_target = None
        self.env_closed = None
        self.dep_audit = None
        self.dep_bytes = None
        self.target_audit = None
        self.evidence = None
        self.evidence_log = None
        self.executable = None
        self.metadata = None
        self.dimensions = None
        self.comparison_schema = None
        self.audit_initial = None
        self.evidence_attempt = None
        self.build_profile = None
        self.deps = None
        self.machine = None
        self.specs = {}
        self.config = None
        self.config_bytes = None
        self.config_id = None
        self.staging = None
        self.journal = []
        self.gates = {}
        self.reasons = []
        self.session = None
        self.link = None
        self.matrix = None
        self.preflight_evidence = None
        self.tuning_id = None

    # -- reasons -------------------------------------------------------------
    def reason(self, code: str, scope: str, detail=None) -> None:
        rec = {"code": code, "scope": scope, "detail": detail}
        if rec not in self.reasons:
            self.reasons.append(rec)

    def has_scope(self, scope: str) -> bool:
        return any(r["scope"] == scope for r in self.reasons)

    # -- persisted paths -----------------------------------------------------
    def path(self, *parts) -> str:
        return os.path.join(self.staging, *parts)

    def log(self, data) -> None:
        if isinstance(data, str):
            data = data.encode("utf-8", "replace")
        with open(self.path("bench.log"), "ab") as f:
            f.write(data)

    def write_journal(self) -> None:
        pc.write_canonical_json(self.path("process-journal.json"), {
            "schema": JOURNAL_SCHEMA, "mode": self.mode, "system_role": self.role,
            "config_id": self.config_id, "entries": self.journal,
            "failure_reasons": self.reasons,
        })

    def write_gates(self) -> None:
        pc.write_canonical_json(self.path("gate-results.json"),
                                {n: self.gates[n] for n in pc.GATE_ORDER})

    def entry(self, name: str) -> dict:
        for e in self.journal:
            if e["name"] == name:
                return e
        raise pc.RunnerError("internal", "InternalError", "journal lacks %s" % name)

    # -- identity projection roots -------------------------------------------
    def roots(self) -> dict:
        return {"<SOURCE>": self.repo, "<CARGO_HOME>": self.cargo_home, "<HOME>": self.home,
                "<SYSROOT>": (self.toolchain or {}).get("sysroot"),
                "<METADATA_TARGET>": self.metadata_target, "<ROLE_TARGET>": self.role_target,
                "<ROLE_TEMP>": self.role_temp, "<STAGING>": self.staging}

    def project(self, value):
        """Replace audited root prefixes in any string (recursively) by their tokens."""
        if isinstance(value, str):
            for token, root in sorted(((t, r) for t, r in self.roots().items() if r),
                                      key=lambda tr: -len(tr[1])):
                root = root.rstrip("/")
                if value == root or value.startswith(root + "/"):
                    return token + value[len(root):]
            return value
        if isinstance(value, list):
            return [self.project(v) for v in value]
        if isinstance(value, dict):
            return {k: self.project(v) for k, v in value.items()}
        return value

    def normalize_argv(self, argv: list) -> list:
        out = list(argv)
        if out and (out[0] == "cargo" or
                    (self.toolchain and out[0] == self.toolchain.get("cargo_path"))):
            out[0] = "<SYSROOT>/bin/cargo"
        return self.project(out)

    def cargo(self) -> str:
        return self.toolchain["cargo_path"]

    def audit_bundle(self) -> dict:
        return {"repo": self.repo, "snapshot": self.snapshot, "dependency": self.dep_audit,
                "cargo_config": self.cargo_config, "toolchain": self.toolchain,
                "native": self.native, "target_dir": self.role_target,
                "executable": self.executable, "timing_schema": self.timing_schema}

    # -- state persistence (normal mode runs as several processes) ------------
    STATE_KEYS = ("mode", "role", "repo", "threads", "features", "rejected_inputs", "snapshot",
                  "lock_sha", "toolchain", "native", "cargo_config", "home", "cargo_home",
                  "role_temp", "role_target", "env_closed", "dep_audit", "target_audit",
                  "evidence", "executable", "audit_initial", "evidence_attempt",
                  "build_profile", "deps", "machine", "reasons", "dimensions",
                  "comparison_schema")

    def state(self) -> dict:
        st = {"schema": STATE_SCHEMA}
        for key in self.STATE_KEYS:
            st[key] = getattr(self, key)
        st["metadata"] = {"raw": self.metadata.raw, "command": self.metadata.command}
        st["knobs"] = self.knobs
        return st

    def write_state(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f, sort_keys=True)
            f.write("\n")

    def load_state(self, path: str) -> None:
        with open(path, "rb") as f:
            st = json.loads(f.read().decode("utf-8"))
        if st.get("schema") != STATE_SCHEMA or st.get("role") not in pc.SYSTEM_ROLES or \
                (self.role is not None and st.get("role") != self.role):
            raise pc.RunnerError("staging", "OutputParentInvalid",
                                 "%s is not this runner's (%s) state" % (path, self.role))
        for key in self.STATE_KEYS:
            setattr(self, key, st[key])
        self.metadata = pc.ProtocolMetadata(st["metadata"]["raw"], st["metadata"]["command"])
        self.knobs = st["knobs"]

    def cleanup_role_dirs(self) -> dict:
        out = {"role_target_removed": _rmtree(self.role_target),
               "role_temp_removed": _rmtree(self.role_temp),
               "metadata_target_removed": _rmtree(self.metadata_target)}
        return out


def journal_entry(ordinal: int, kind: str, name: str, coordinate) -> dict:
    e = {"ordinal": ordinal, "kind": kind, "name": name, "coordinate": coordinate,
         "status": "not_run", "path": None}
    for key in EXECUTION_FIELDS:
        e[key] = None
    return e


def gate_record(ctx: Context, name: str) -> dict:
    """`gate-results.json` entry (timing-schema `config_schemas.gate_results` entry keys);
    execution fields are null until the gate runs."""
    return {"status": "not_run", "argv": ctx.normalize_argv([ctx.cargo()] +
                                                             pc.gate_command(name)[1:]),
            "profile": "dev", "features": [],
            "target_audit_id": ctx.target_audit["audit_id"],
            "dependency_source_id": ctx.dep_audit["dependency_source_id"],
            "start_utc": None, "end_utc": None, "exit": None, "signal": None,
            "pre": None, "post": None, "log_sha256": None, "test_summary": None}


# ---------------------------------------------------------------------------
# Knob tables.
# ---------------------------------------------------------------------------

def parse_knobs(mode: str, env, session_spec_required: bool = True) -> dict:
    knobs = {"allow_dirty": pc.knob_flag(env, "POSEIDON_ALLOW_DIRTY")}
    if mode == "sweep":
        pc.reject_knobs(env, ("PSIZE", "HASHES", "INSTANCE", "THREADS", "NORMAL_RUN",
                              "TUNING_BUNDLE", "TUNING_STORE", "SESSION_SPEC"), mode)
        knobs["backend"] = pc.knob_enum(env, "BACKEND", pc.BACKENDS)
        knobs["output_parent"] = pc.knob_string(env, "OUTPUT_PARENT", required=True)
        if knobs["backend"] is None:
            raise pc.RunnerError("env", "InvalidKnob", "sweep requires BACKEND")
        knobs["threads"] = 1
        knobs["hashes"] = pc.CANONICAL_HASHES
    elif mode == "psize":
        pc.reject_knobs(env, ("SWEEP", "BACKEND", "HASHES", "INSTANCE", "THREADS",
                              "TUNING_BUNDLE", "SESSION_SPEC"), mode)
        knobs["normal_run"] = pc.knob_string(env, "NORMAL_RUN", required=True)
        knobs["tuning_store"] = pc.knob_string(env, "TUNING_STORE", required=True)
        knobs["output_parent"] = pc.knob_string(env, "OUTPUT_PARENT", required=True)
    elif mode == "normal":
        pc.reject_knobs(env, ("NORMAL_RUN", "OUTPUT_PARENT"), mode)
        if pc.knob_flag(env, "SWEEP") or pc.knob_flag(env, "PSIZE"):
            raise pc.RunnerError("env", "ModeConflict", "SWEEP/PSIZE cannot be 1 in normal mode")
        knobs["backend"] = pc.knob_enum(env, "BACKEND", pc.BACKENDS)
        if knobs["backend"] is None:
            raise pc.RunnerError("env", "InvalidKnob", "normal requires BACKEND")
        if session_spec_required:
            knobs["tuning_bundle"] = pc.knob_string(env, "TUNING_BUNDLE", required=True)
            knobs["tuning_store"] = pc.knob_string(env, "TUNING_STORE", required=True)
            knobs["session_spec"] = pc.knob_string(env, "SESSION_SPEC", required=True)
        else:
            pc.reject_knobs(env, ("SESSION_SPEC",), "normal preconfigure")
            knobs["tuning_bundle"] = pc.knob_string(env, "TUNING_BUNDLE")
            knobs["tuning_store"] = pc.knob_string(env, "TUNING_STORE")
        for name, fixed in (("HASHES", pc.CANONICAL_HASHES), ("INSTANCE", pc.HELD_OUT_INSTANCE)):
            value = pc.knob_usize(env, name)
            if value is not None and value != fixed:
                raise pc.RunnerError("env", "InvalidKnob", "%s must be %d in normal mode" %
                                     (name, fixed))
        knobs["threads"] = pc.knob_usize(env, "THREADS", 1)
        if knobs["threads"] == 0:
            raise pc.RunnerError("env", "InvalidKnob", "THREADS must be >= 1")
        knobs["hashes"] = pc.CANONICAL_HASHES
        knobs["instance"] = pc.HELD_OUT_INSTANCE
    else:
        raise pc.RunnerError("internal", "InternalError", "unknown mode %s" % mode)
    return knobs


def check_output_parent(path: str, disjoint_from) -> str:
    if not os.path.isabs(path):
        raise pc.RunnerError("env", "OutputParentInvalid", "OUTPUT_PARENT must be absolute")
    real = os.path.realpath(path)
    if real != path.rstrip("/") or not os.path.isdir(path) or os.path.islink(path):
        raise pc.RunnerError("env", "OutputParentInvalid",
                             "OUTPUT_PARENT must be an existing, normalized, symlink-free "
                             "directory (%s)" % path)
    if not os.access(path, os.W_OK | os.X_OK):
        raise pc.RunnerError("env", "OutputParentInvalid", "OUTPUT_PARENT is not writable")
    for other in disjoint_from:
        if other and _overlaps(real, os.path.realpath(other)):
            raise pc.RunnerError("env", "PathOverlap",
                                 "OUTPUT_PARENT %s overlaps %s" % (real, other))
    return real


def check_input_dir(path: str, what: str) -> str:
    if not os.path.isabs(path):
        raise pc.RunnerError("env", "LinkInvalid", "%s must be absolute" % what)
    if os.path.realpath(path) != path.rstrip("/") or not os.path.isdir(path):
        raise pc.RunnerError("env", "LinkInvalid",
                             "%s must be an existing normalized symlink-free directory" % what)
    return path.rstrip("/")


def check_repo(path: str) -> str:
    if not os.path.isabs(path) or not os.path.isdir(path) or os.path.islink(path):
        raise pc.RunnerError("env", "OutputParentInvalid",
                             "--repo must be an absolute existing directory (%s)" % path)
    return path.rstrip("/")


# ---------------------------------------------------------------------------
# Pre-configuration phase (plan 2220-2236; no archive is created here).
# ---------------------------------------------------------------------------

def load_specs(ctx: Context) -> None:
    for name in pc.SPEC_ORDER:
        data, sha = pc.load_spec(ctx.repo, name)
        ctx.specs[name] = {"bytes": data, "sha256": sha, "metadata_key": pc.SPEC_FILES[name][1],
                           "compiled_id": None, "match": None}
    ctx.timing_schema = pc.load_canonical_json(ctx.specs["timing-schema-v2.json"]["bytes"],
                                               "timing-schema-v2.json")
    ctx.tuning_protocol = pc.load_canonical_json(
        ctx.specs["tuning-protocol-v2.json"]["bytes"], "tuning-protocol-v2.json")
    # The comparison schema is tracked but not archived: its digest is compared with the
    # compiled `comparison_schema_id` (a mismatch or an absent file is a recorded failure).
    path = os.path.join(ctx.repo, pc.COMPARISON_SCHEMA_PATH)
    try:
        with open(path, "rb") as f:
            ctx.comparison_schema = {"path": pc.COMPARISON_SCHEMA_PATH,
                                     "sha256": pc.sha256_hex(f.read()), "present": True}
    except OSError:
        ctx.comparison_schema = {"path": pc.COMPARISON_SCHEMA_PATH, "sha256": None,
                                 "present": False}


def bind_spec_checks(ctx: Context) -> None:
    for name in pc.SPEC_ORDER:
        spec = ctx.specs[name]
        compiled = ctx.metadata.get(spec["metadata_key"])
        spec["compiled_id"] = compiled
        spec["match"] = compiled == spec["sha256"]
        if compiled is None:
            ctx.reason("spec_gate_failed", "execution", {"file": name,
                                                         "reason": "compiled_id_null"})
        elif not spec["match"]:
            ctx.reason("spec_gate_failed", "execution", {"file": name, "reason": "mismatch"})
    compiled = ctx.metadata.get("comparison_schema_id")
    cs = ctx.comparison_schema
    cs["compiled_id"] = compiled
    cs["match"] = cs["present"] and compiled == cs["sha256"]
    if not cs["present"]:
        ctx.reason("spec_gate_failed", "execution", {"file": "comparison-schema-v2.json",
                                                     "reason": "file_missing"})
    elif compiled is None:
        ctx.reason("spec_gate_failed", "execution", {"file": "comparison-schema-v2.json",
                                                     "reason": "compiled_id_null"})
    elif not cs["match"]:
        ctx.reason("spec_gate_failed", "execution", {"file": "comparison-schema-v2.json",
                                                     "reason": "mismatch"})


class LogSink:
    """`log_sink` for `be.run_cargo_subprocess`: accepts `sink(stream, bytes)` (preferred:
    `stream` is "stdout" or "stderr") and file-like `write(bytes)` of a combined stream."""

    def __init__(self):
        self.streams = {}
        self.combined = bytearray()

    def __call__(self, *args):
        if len(args) == 2:
            self.streams[args[0]] = bytes(args[1])
            self.combined += bytes(args[1])
        elif len(args) == 1:
            self.combined += bytes(args[0])

    def write(self, data) -> int:
        self.combined += bytes(data)
        return len(data)

    def flush(self) -> None:
        return None

    def stdout(self) -> bytes:
        return self.streams.get("stdout", bytes(self.combined))

    def log_bytes(self) -> bytes:
        return bytes(self.combined)


def run_subprocess(ctx: Context, entry: dict, argv: list, env: dict, cwd=None) -> tuple:
    """One Cargo subprocess through `be.run_cargo_subprocess` (pre/post rehash around it);
    fills the journal entry's execution fields. Returns `(record, sink)`."""
    sink = LogSink()
    entry["argv"] = ctx.normalize_argv(argv)
    entry["loadavg_before"] = _loadavg()
    if ctx.staging:
        ctx.log("==== [%d] %s: %s\n" % (entry["ordinal"], entry["name"], " ".join(argv)))
    try:
        rec = _be().run_cargo_subprocess(argv, env, cwd or ctx.repo, ctx.audit_bundle(), sink)
    except pc.RunnerError as exc:
        entry["status"] = "failed"
        entry["error"] = exc.to_json()
        entry["loadavg_after"] = _loadavg()
        entry["log_sha256"] = pc.sha256_hex(sink.log_bytes())
        if ctx.staging:
            ctx.log(sink.log_bytes() + ("==== [%d] error %s\n" % (entry["ordinal"],
                                                                exc)).encode())
        if exc.stage in ("source", "dependency", "toolchain", "evidence"):
            ctx.reason(exc.error_code, "provenance", exc.message)
        raise
    entry["pid"] = rec.get("pid")
    entry["exit_code"] = rec.get("exit")
    entry["signal"] = rec.get("signal")
    entry["started_utc"] = rec.get("start_utc")
    entry["ended_utc"] = rec.get("end_utc")
    entry["monotonic_start_ns"] = int(rec["start_mono"]) if rec.get("start_mono") is not None \
        else None
    entry["monotonic_end_ns"] = int(rec["end_mono"]) if rec.get("end_mono") is not None \
        else None
    entry["pre_audit"] = rec.get("pre")
    entry["post_audit"] = rec.get("post")
    entry["loadavg_after"] = _loadavg()
    entry["log_sha256"] = pc.sha256_hex(sink.log_bytes())
    if ctx.staging:
        ctx.log(sink.log_bytes() + ("==== [%d] exit %s signal %s\n" % (
            entry["ordinal"], rec.get("exit"), rec.get("signal"))).encode())
    return rec, sink


def full_recheck(ctx: Context, label: str) -> dict:
    """Recompute every frozen audit (source/lock snapshot, dependency closure, Cargo config,
    toolchain/native closures, evidenced executable) and require equality."""
    b = _be()
    try:
        b.recheck_source_snapshot(ctx.repo, ctx.snapshot)
        closure = b.rehash_dependency_closure(ctx.dep_audit)
        if closure != ctx.dep_audit["closure_sha256"]:
            raise pc.RunnerError("dependency", "DependencyClosureChanged",
                                 "dependency closure %s != frozen %s" %
                                 (closure, ctx.dep_audit["closure_sha256"]))
        b.recheck_cargo_config(ctx.cargo_config)
        b.recheck_toolchain(ctx.toolchain)
        b.recheck_native_tools(ctx.native)
        if ctx.executable is not None:
            b.rehash_executable(ctx.role_target, ctx.executable)
        hashes = b.rehash_all(ctx.audit_bundle())
        if ctx.audit_initial is not None and hashes != ctx.audit_initial:
            raise pc.RunnerError("source", "SourceChanged",
                                 "audit hashes at %s differ from the frozen set" % label)
    except pc.RunnerError as exc:
        ctx.reason(exc.error_code, "provenance", "%s: %s" % (label, exc.message))
        raise
    return hashes


def preconfigure(ctx: Context, threads: int) -> None:
    """Validate the environment, snapshot the source, audit the dependency sources, build
    the evidence in a fresh role target and extract the compiled metadata from it."""
    b = _be()
    env, repo = ctx.env, ctx.repo
    ctx.threads = threads
    ctx.features = pc.features_for_threads(threads)
    ctx.rejected_inputs = b.reject_inherited_environment(env)
    load_specs(ctx)
    external_roots = [p for p in (ctx.knobs.get("output_parent"), ctx.knobs.get("tuning_store"),
                                  ctx.knobs.get("normal_run")) if p]
    ctx.snapshot = b.source_snapshot(repo, ctx.knobs["allow_dirty"],
                                     external_roots=external_roots)
    if ctx.knobs["allow_dirty"]:
        # The override is accepted in every mode but forces eligible_for = [] (plan 2458),
        # even when the tree happens to be clean; an actually dirty tree additionally
        # invalidates provenance.
        ctx.reason("dirty_override_active", "role")
    if ctx.snapshot["dirty"]:
        ctx.reason("dirty_worktree_allowed", "provenance")
    if not pc.git_is_tracked(repo, "Cargo.lock"):
        raise pc.RunnerError("source", "LockfileUntracked", "Cargo.lock is not tracked by git")
    with open(os.path.join(repo, "Cargo.lock"), "rb") as f:
        ctx.lock_bytes = f.read()
    ctx.lock_sha = pc.sha256_hex(ctx.lock_bytes)
    if ctx.lock_sha != ctx.snapshot["lock_sha256"]:
        raise pc.RunnerError("source", "SourceSnapshotInvalid",
                             "Cargo.lock bytes differ from the snapshot's lock hash")
    ctx.deps = pc.dependency_versions(repo)
    ctx.toolchain = b.resolve_toolchain(env, ctx.timing_schema, cwd=repo)
    ctx.native = b.resolve_native_tools(ctx.toolchain, ctx.timing_schema)
    ctx.home = env.get("HOME")
    if not ctx.home or not os.path.isabs(ctx.home):
        raise pc.RunnerError("env", "InvalidKnob", "HOME must be an absolute path")
    ctx.cargo_home = env.get("CARGO_HOME") or os.path.join(ctx.home, ".cargo")
    ctx.cargo_config = b.cargo_config_audit(repo, ctx.cargo_home, ctx.timing_schema, repo=repo,
                                            rustflags=pc.RUSTFLAGS)
    disjoint = [repo, ctx.knobs.get("output_parent"), ctx.knobs.get("tuning_store"),
                ctx.knobs.get("normal_run")]
    ctx.role_temp = fresh_temp_dir("poseidon-%s-temp-" % ctx.role, disjoint)
    role = {"name": ctx.role, "system": pc.SYSTEM, "mode": ctx.mode, "features": ctx.features,
            "rustflags": pc.RUSTFLAGS}
    # Dependency-source audit and metadata replay in a disposable metadata target.
    ctx.metadata_target = fresh_temp_dir("poseidon-%s-metadata-" % ctx.role, disjoint)
    env_meta = b.closed_environment(role, ctx.toolchain, ctx.native, ctx.cargo_home, ctx.home,
                                    ctx.metadata_target, ctx.role_temp, threads,
                                    ctx.timing_schema)
    audit = b.dependency_source_audit(repo, ctx.snapshot, env_meta, ctx.toolchain,
                                      ctx.cargo_home, ctx.metadata_target, ctx.features,
                                      ctx.timing_schema)
    # The bootstrap target now holds cargo's rustc-info cache; the replay uses
    # its own fresh empty disposable target under the role temp directory.
    b.replay_metadata(audit, env_meta)
    ctx.dep_bytes = audit["bytes"]
    if pc.sha256_hex(ctx.dep_bytes) != audit["dependency_source_id"]:
        raise pc.RunnerError("dependency", "DependencySourceInvalid",
                             "dependency_source_id is not the SHA-256 of the document")
    # The frozen audit (document + raw locators) is what every later rehash recomputes
    # against; the bytes live in dependency-sources.json.
    ctx.dep_audit = {k: v for k, v in audit.items() if k != "bytes"}
    shutil.rmtree(ctx.metadata_target)
    ctx.metadata_target = None
    # Fresh role target, closed environment, evidence build.
    ctx.role_target = fresh_temp_dir("poseidon-%s-target-" % ctx.role, disjoint)
    ctx.env_closed = b.closed_environment(role, ctx.toolchain, ctx.native, ctx.cargo_home,
                                          ctx.home, ctx.role_target, ctx.role_temp, threads,
                                          ctx.timing_schema)
    if ctx.env_closed.get("RAYON_NUM_THREADS") != str(threads):
        raise pc.RunnerError("env", "InvalidKnob", "closed environment RAYON_NUM_THREADS != T")
    ctx.target_audit = {"token": "<ROLE_TARGET>", "initially_empty": True, "entries": 0,
                        "created_utc": pc.utc_rfc3339_ns(_now_ns())}
    ctx.target_audit["audit_id"] = pc.sha256_hex(pc.canonical_json_bytes(
        {"role": ctx.role, "token": "<ROLE_TARGET>", "initially_empty": True, "entries": 0}))
    ev_entry = journal_entry(0, "evidence_metadata", "evidence_metadata",
                             {"kind": "evidence_metadata"})
    pre = b.rehash_all(ctx.audit_bundle())
    ev_start = pc.utc_rfc3339_ns(_now_ns())
    ev_mono = time.monotonic_ns()
    evidence = b.evidence_build(repo, ctx.env_closed, ctx.role_target, ctx.features, ctx.role,
                                ctx.timing_schema, dependency_audit=audit)
    ctx.evidence_log = evidence["log_bytes"]
    ctx.evidence = {k: v for k, v in evidence.items() if k != "log_bytes"}
    ctx.executable = dict(evidence["executable"])
    post = b.rehash_all(ctx.audit_bundle())
    ctx.audit_initial = post
    evidence_rec = {"argv": ctx.normalize_argv(pc.evidence_command(ctx.features)),
                    "started_utc": ev_start, "ended_utc": pc.utc_rfc3339_ns(_now_ns()),
                    "monotonic_start_ns": ev_mono, "monotonic_end_ns": time.monotonic_ns(),
                    "exit_code": evidence.get("exit"), "log_sha256": evidence["log_sha256"],
                    "pre_audit": pre, "post_audit": post, "executable": ctx.executable}
    # Compiled metadata from the same target (exact command; whole stdout = one JSON line).
    argv = [ctx.cargo()] + pc.metadata_command(ctx.features)[1:]
    md_entry = journal_entry(0, "evidence_metadata", "metadata", None)
    rec, sink = run_subprocess(ctx, md_entry, argv, ctx.env_closed)
    if rec.get("exit") != 0:
        raise pc.RunnerError("metadata", "MetadataUnavailable", "%s exited %s: %s" % (
            " ".join(argv), rec.get("exit"), sink.log_bytes().decode("utf-8", "replace")[-2000:]))
    ctx.metadata = pc.parse_protocol_metadata(sink.stdout(), ctx.normalize_argv(argv))
    # `dimensions_by_k` (H = 10) must name every TUNE-1 candidate: the run config carries
    # it as `dimensions_by_candidate` and the Criterion IDs are checked against it.
    ctx.dimensions = pc.dimensions_by_candidate(ctx.metadata)
    md_rec = {k: md_entry[k] for k in ("argv", "pid", "exit_code", "signal", "started_utc",
                                       "ended_utc", "monotonic_start_ns", "monotonic_end_ns",
                                       "pre_audit", "post_audit", "log_sha256")}
    md_rec["stdout_sha256"] = rec.get("stdout_sha256")
    ctx.audit_initial = full_recheck(ctx, "after metadata")
    ev_entry.update({"status": "ok", "argv": [evidence_rec["argv"], md_rec["argv"]],
                     "pid": md_rec["pid"], "exit_code": 0, "signal": None,
                     "started_utc": evidence_rec["started_utc"],
                     "ended_utc": md_rec["ended_utc"],
                     "monotonic_start_ns": evidence_rec["monotonic_start_ns"],
                     "monotonic_end_ns": md_rec["monotonic_end_ns"],
                     "loadavg_before": md_entry["loadavg_before"],
                     "loadavg_after": md_entry["loadavg_after"],
                     "pre_audit": pre, "post_audit": ctx.audit_initial,
                     "log_sha256": pc.sha256_hex(ctx.evidence_log + sink.log_bytes()),
                     "import_status": "not_applicable",
                     "attempt_result": {"evidence": evidence_rec, "metadata": md_rec,
                                        "dependency_audit": {
                                            "dependency_source_id":
                                                ctx.dep_audit["dependency_source_id"],
                                            "closure_sha256": ctx.dep_audit["closure_sha256"],
                                            "metadata_normalized_sha256":
                                                ctx.dep_audit["metadata_normalized_sha256"],
                                            "replay": "identical"}}})
    ctx.evidence_attempt = ev_entry
    ctx.machine = pc.machine_fingerprint()
    bind_spec_checks(ctx)
    metadata_value_check(ctx)
    ctx.build_profile = b.build_profile_document(
        evidence=evidence, snapshot=ctx.snapshot, dependency=audit,
        cargo_config=ctx.cargo_config, toolchain=ctx.toolchain, native=ctx.native,
        env_raw=ctx.env_closed, roots=ctx.roots(), rejected_audit=ctx.rejected_inputs,
        compiled_metadata=ctx.metadata.raw, timing_schema=ctx.timing_schema,
        features=ctx.features, role=ctx.role)


def metadata_value_check(ctx: Context) -> None:
    md = ctx.metadata
    for key, expected in pc.METADATA_REQUIRED_VALUES.items():
        if md.get(key) is not expected and md.get(key) != expected:
            ctx.reason("metadata_%s_invalid" % key, "execution", md.get(key))
    for key in pc.METADATA_FORBIDDEN_KEYS:
        if key in md.raw:
            ctx.reason("metadata_%s_not_applicable" % key, "execution", md.get(key))
    if md.get("criterion_version") != pc.CRITERION_SETTINGS["version"]:
        ctx.reason("criterion_version_mismatch", "execution", md.get("criterion_version"))
    # The bench requires run-config `features` to equal its compiled feature list.
    if "features" in md.raw and md.get("features") != list(ctx.features):
        ctx.reason("compiled_features_mismatch", "execution",
                   {"compiled": md.get("features"), "requested": list(ctx.features)})


def compiled_tuning_check(ctx: Context, backend: str, tuning_id, epoch_id, k) -> None:
    """The bench requires `tuning.tuning_id`/`tuning_epoch_id` to equal its compiled tuned
    default for `backend` and its compiled `TUNING_EPOCH_ID`, and the workload k to be the
    compiled default k; record a mismatch."""
    md = ctx.metadata
    compiled = md.tuned_default(backend)
    if md.get("tuning_epoch_id") != epoch_id or compiled["tuning_id"] != tuning_id or \
            (compiled["k"] is not None and compiled["k"] != k):
        ctx.reason("compiled_tuning_mismatch", "execution",
                   {"compiled_epoch": md.get("tuning_epoch_id"), "epoch": epoch_id,
                    "compiled_default": compiled, "tuning_id": tuning_id, "k": k})


def capability_check(ctx: Context, backend: str) -> None:
    md = ctx.metadata
    tuple_ = md.capability_tuple()
    approved = False
    for cand in pc.APPROVED_CAPABILITY_TUPLES:
        if all(cand[k] == tuple_[k] for k in cand if k != "variants") and \
                backend in cand["variants"]:
            approved = True
    if not approved:
        ctx.reason("capability_tuple_not_approved", "execution", tuple_)
    if md.get("common_lift_binding_id") is not None:
        ctx.reason("common_lift_not_applicable", "execution", md.get("common_lift_binding_id"))
    if md.get("prime_sampler_id") != pc.PRIME_SAMPLER_ID:
        ctx.reason("sampler_not_p0d", "execution", md.get("prime_sampler_id"))


def dependency_sources_document(ctx: Context) -> dict:
    return {"dependency_source_id": ctx.dep_audit["dependency_source_id"],
            "dependency_sources_sha256": ctx.dep_audit["dependency_source_id"],
            "closure_sha256": ctx.dep_audit["closure_sha256"],
            "metadata_normalized_sha256": ctx.dep_audit["metadata_normalized_sha256"],
            "replay": "identical", "file": "dependency-sources.json"}


def build_run_config(ctx: Context, workload: dict, extra: dict) -> None:
    b = _be()
    md = ctx.metadata
    backend = workload["backend"]
    k = workload["k"]
    cfg = {
        "schema": RUN_CONFIG_SCHEMA,
        "plan_revision": "v10",
        "created_utc": pc.utc_rfc3339_ns(_now_ns()),
        "mode": ctx.mode,
        "system_role": ctx.role,
        "backend": backend,
        "k": k,
        "dimensions": ctx.dimensions[str(k)] if k is not None else None,
        "dimensions_by_candidate": dict(ctx.dimensions),
        "workload": workload,
        "sweep": None,
        "check_modes": dict(pc.CHECK_MODES),
        "criterion": {"sample_size": pc.CRITERION_SETTINGS["sample_size"],
                      "warm_up_time_s": pc.CRITERION_SETTINGS["warm_up_time_s"],
                      "measurement_time_s": pc.CRITERION_SETTINGS["measurement_time_s"]},
        "criterion_policy": {
            "version": pc.CRITERION_SETTINGS["version"],
            "default_features": False, "features": pc.CRITERION_SETTINGS["features"],
            "one_group_per_primary_process": True,
            "primary_order": list(pc.PRIMARY_GROUPS),
            "diagnostic_order": list(pc.diagnostic_groups_for(backend)),
        },
        "tuning": {"tuning_id": extra.get("tuning_id"),
                   "tuning_epoch_id": extra.get("tuning_epoch_id")},
        "tuning_id": None, "tuning_epoch_id": None, "comparison_key": None,
        "ids": {
            "compiled": md.compiled_ids(),
            "tracked_specs": {n: ctx.specs[n]["sha256"] for n in pc.SPEC_ORDER},
            "spec_checks": {n: {"file_sha256": ctx.specs[n]["sha256"],
                                "metadata_key": ctx.specs[n]["metadata_key"],
                                "compiled_id": ctx.specs[n]["compiled_id"],
                                "match": ctx.specs[n]["match"]} for n in pc.SPEC_ORDER},
            "comparison_schema": dict(ctx.comparison_schema),
            "benchmark_coins_policy": pc.benchmark_coins_policy(ctx.tuning_protocol),
        },
        "features": list(ctx.features),
        "rustflags": pc.RUSTFLAGS,
        "source": {"git_sha": ctx.snapshot["commit"], "tree": ctx.snapshot["tree"],
                   "source_snapshot_id": ctx.snapshot["source_snapshot_id"],
                   "closure_sha256": ctx.snapshot.get("closure_sha256"),
                   "dirty": ctx.snapshot["dirty"],
                   "dirty_closure_sha256": ctx.snapshot.get("dirty_closure_sha256"),
                   "allow_dirty": ctx.knobs["allow_dirty"]},
        "lockfile": {"path": "Cargo.lock", "sha256": ctx.lock_sha, "tracked": True,
                     "blob": ctx.snapshot.get("lock_blob"), "archived_as": "Cargo.lock.archived"},
        "dependency_sources": dependency_sources_document(ctx),
        "toolchain": ctx.project({k_: v for k_, v in ctx.toolchain.items()
                                  if k_ != "toolchain_closure"}),
        "native_tools": ctx.project(ctx.native),
        "environment": b.identity_projection(ctx.env_closed, ctx.roots()),
        "executable": ctx.executable,
        "cargo_config_audit": {
            "audit_id": ctx.cargo_config.get("audit_id"),
            "candidates": ctx.project([c.get("path") if isinstance(c, dict) else c
                                       for c in ctx.cargo_config.get("candidates", [])]),
            "discovered": ctx.project(ctx.cargo_config.get("discovered")),
            "parsed": ctx.cargo_config.get("parsed"),
        },
        "rejected_inputs_audit": ctx.rejected_inputs,
        "build": {"profile": "bench", "allocator": "system", "features": list(ctx.features),
                  "parallel": False, "locked": True, "offline": True,
                  "rustflags": pc.RUSTFLAGS, "rayon_num_threads": workload["threads"],
                  "wrapper_chain": [], "role_target": "<ROLE_TARGET>",
                  "role_target_audit_id": ctx.target_audit["audit_id"],
                  "evidence_command": ctx.normalize_argv(pc.evidence_command(ctx.features)),
                  "metadata_command": ctx.normalize_argv(pc.metadata_command(ctx.features)),
                  "bench_command_prefix": ctx.normalize_argv(pc.bench_prefix(ctx.features)),
                  "gate_commands": {n: ctx.normalize_argv(pc.gate_command(n))
                                    for n in pc.GATE_ORDER},
                  "build_profile": "build-profile.json", "build_profile_log":
                  "build-profile.log"},
        "dependencies": ctx.deps,
        "machine": ctx.machine,
        "preflight_requirements": {
            "check_mode": pc.CHECK_MODE_METADATA,
            "deterministic_coins": pc.BENCHMARK_COINS_ID,
            "double_construction": True, "p0d_audit": True,
            "rayon_threads_asserted": workload["threads"],
        },
    }
    cfg.update(extra)
    cfg["tuning"] = {"tuning_id": cfg["tuning_id"], "tuning_epoch_id": cfg["tuning_epoch_id"]}
    ctx.config = cfg
    ctx.config_bytes = pc.canonical_json_bytes(cfg)
    ctx.config_id = pc.sha256_hex(ctx.config_bytes)


def comparison_key(ctx: Context, workload: dict, tuning_id, tuning_epoch_id) -> str:
    """Hash of every result-affecting field shared by normal and psize (plan 2036-2043)."""
    md = ctx.metadata
    fields = {
        "source_snapshot_id": ctx.snapshot["source_snapshot_id"], "lockfile_sha256": ctx.lock_sha,
        "dependency_source_id": ctx.dep_audit["dependency_source_id"],
        "toolchain_closure_sha256": ctx.toolchain["toolchain_closure_sha256"],
        "native_closure_sha256": ctx.native["native_closure_sha256"],
        "rustflags": pc.RUSTFLAGS, "features": list(ctx.features),
        "backend": workload["backend"], "k": workload["k"],
        "hashes_per_field": workload["hashes_per_field"],
        "dimensions": ctx.dimensions[str(workload["k"])],
        "instance": workload["instance"], "threads": workload["threads"],
        "check_modes": dict(pc.CHECK_MODES),
        "kat_fixture_sha256": md.get("kat_fixture_sha256"),
        "tuning_corpus_id": md.get("tuning_corpus_id"),
        "tuning_protocol_id": md.get("tuning_protocol_id"),
        "timing_schema_id": md.get("timing_schema_id"),
        "security_accounting_id": md.get("security_accounting_id"),
        "comparison_schema_id": md.get("comparison_schema_id"),
        "protocol_wire_id": md.get("protocol_wire_id"),
        "transcript_domain_separator": md.get("transcript_domain_separator"),
        "prime_sampler_id": md.get("prime_sampler_id"),
        "benchmark_coins_id": md.get("benchmark_coins_id"),
        "benchmark_coins_framing": pc.benchmark_coins_framing(ctx.tuning_protocol),
        "tuning_id": tuning_id, "tuning_epoch_id": tuning_epoch_id,
    }
    return pc.sha256_hex(pc.canonical_json_bytes(fields))


# ---------------------------------------------------------------------------
# Staging initialisation (evidence first, config pair last), gates.
# ---------------------------------------------------------------------------

def write_evidence(ctx: Context, root: str) -> None:
    pc.write_bytes(os.path.join(root, "dependency-sources.json"), ctx.dep_bytes)
    pc.write_canonical_json(os.path.join(root, "build-profile.json"), ctx.build_profile)
    pc.write_bytes(os.path.join(root, "build-profile.log"), ctx.evidence_log)


def read_evidence(ctx: Context, root: str) -> None:
    with open(os.path.join(root, "dependency-sources.json"), "rb") as f:
        ctx.dep_bytes = f.read()
    if pc.sha256_hex(ctx.dep_bytes) != ctx.dep_audit["dependency_source_id"]:
        raise pc.RunnerError("dependency", "DependencySourceInvalid",
                             "dependency-sources.json does not hash to DEPENDENCY_SOURCE_ID")
    with open(os.path.join(root, "build-profile.log"), "rb") as f:
        ctx.evidence_log = f.read()
    if pc.sha256_hex(ctx.evidence_log) != ctx.evidence["log_sha256"]:
        raise pc.RunnerError("evidence", "EvidenceBuildFailed",
                             "build-profile.log does not hash to the evidence log hash")


def planned_journal(ctx: Context, child_entries: list, fourth_name: str) -> None:
    prep = [ctx.evidence_attempt,
            journal_entry(1, "gate", "kat", {"kind": "gate", "gate": "kat"}),
            journal_entry(2, "gate", "tune_corpus", {"kind": "gate", "gate": "tune_corpus"}),
            journal_entry(3, "preflight", fourth_name, {"kind": fourth_name})]
    ctx.journal = prep + child_entries


def init_staging(ctx: Context, child_entries: list, fourth_name: str, before_config=()) -> None:
    """Evidence, lockfile, spec copies and extra files first; the config pair last."""
    st = ctx.staging
    write_evidence(ctx, st)
    pc.write_bytes(os.path.join(st, "Cargo.lock.archived"), ctx.lock_bytes)
    for name in pc.SPEC_ORDER:
        pc.write_bytes(os.path.join(st, name), ctx.specs[name]["bytes"])
    pc.write_bytes(os.path.join(st, "bench.log"), b"")
    for name, data in before_config:
        pc.write_bytes(os.path.join(st, name), data)
    ctx.gates = {n: gate_record(ctx, n) for n in pc.GATE_ORDER}
    planned_journal(ctx, child_entries, fourth_name)
    ctx.write_gates()
    ctx.write_journal()
    pc.write_bytes(os.path.join(st, "run-config.json"), ctx.config_bytes)
    pc.write_detached_digest(os.path.join(st, "run-config.sha256"), ctx.config_id)


def run_gate(ctx: Context, name: str) -> bool:
    """One exact release-test subprocess in the closed environment and role target."""
    entry = ctx.entry(name)
    if entry["status"] != "not_run":
        raise pc.RunnerError("gates", "GateFailed", "gate %s already ran" % name)
    argv = [ctx.cargo()] + pc.gate_command(name)[1:]
    entry["import_status"] = "not_applicable"
    rec = ctx.gates[name]
    try:
        result, sink = run_subprocess(ctx, entry, argv, ctx.env_closed)
    except pc.RunnerError:
        rec.update({"status": "failed", "start_utc": entry["started_utc"],
                    "test_summary": {"error": entry["error"]}})
        ctx.write_gates()
        ctx.write_journal()
        raise
    stdout = sink.stdout().decode("utf-8", "replace")
    result_line = None
    for line in stdout.split("\n"):
        if line.startswith("test result:"):
            result_line = line
    test_name = pc.GATE_TEST_NAMES[name]
    passed = (result.get("exit") == 0 and result_line is not None
              and result_line.startswith(pc.GATE_OK_PREFIX)
              and ("test %s ... ok" % test_name) in stdout)
    summary = {"test": test_name, "result_line": result_line, "expected_prefix": pc.GATE_OK_PREFIX,
               "passed": passed, "stdout_sha256": result.get("stdout_sha256"),
               "stderr_sha256": result.get("stderr_sha256")}
    rec.update({"status": "ok" if passed else "failed",
                "start_utc": entry["started_utc"], "end_utc": entry["ended_utc"],
                "exit": entry["exit_code"], "signal": entry["signal"],
                "pre": entry["pre_audit"], "post": entry["post_audit"],
                "log_sha256": entry["log_sha256"], "test_summary": summary})
    entry["status"] = "ok" if passed else "failed"
    if not passed:
        entry["error"] = {"stage": "gates", "error_code": "GateFailed",
                          "message": "%s did not report %r" % (name, pc.GATE_OK_PREFIX)}
        ctx.reason("gate_failed", "execution", name)
    ctx.write_gates()
    ctx.write_journal()
    return passed


def run_all_gates(ctx: Context) -> None:
    for name in pc.GATE_ORDER:
        if not run_gate(ctx, name):
            raise pc.RunnerError("gates", "GateFailed", "gate %s failed" % name)


# ---------------------------------------------------------------------------
# Preflight attempt (form 2) and its imported files.
# ---------------------------------------------------------------------------

def validate_preflight(obj, expected: dict, config_id: str, threads: int,
                       compiled_sampler_id) -> list:
    """Return the list of failed preflight conjuncts (empty = valid), schema
    `limber/poseidon-preflight/v2` (contract section 4): status ok, the asserted rayon
    thread count, the complete P0-D prime audit under the compiled sampler id with
    `invocations == len(records)`, the four double-construction flags, canonical IO and
    shape satisfiability, three digests, and the workload fields of the config."""
    failed = []
    if not isinstance(obj, dict) or obj.get("schema") != PREFLIGHT_SCHEMA:
        return ["schema"]
    if obj.get("config_sha256") != config_id:
        failed.append("config_sha256")
    if obj.get("status") != "ok":
        failed.append("status")
    if obj.get("error") is not None:
        failed.append("error")
    if obj.get("rayon_threads_asserted") != threads or \
            not _is_int(obj.get("rayon_threads_asserted")):
        failed.append("rayon_threads_asserted")
    audit = obj.get("prime_audit")
    if not isinstance(audit, dict):
        failed.append("prime_audit")
    else:
        if audit.get("complete") is not True:
            failed.append("prime_audit_complete")
        if audit.get("prime_sampler_id") != compiled_sampler_id or \
                not isinstance(audit.get("prime_sampler_id"), str):
            failed.append("prime_audit_sampler_id")
        records = audit.get("records")
        count = audit.get("invocations")
        if not isinstance(records, list) or not records:
            failed.append("prime_audit_records")
        else:
            for i, rec in enumerate(records):
                if not isinstance(rec, dict) or any(k not in rec for k in
                                                    PRIME_AUDIT_RECORD_KEYS):
                    failed.append("prime_audit_record_%d" % i)
                    break
        if not _is_int(count) or count < 1 or not isinstance(records, list) or \
                count != len(records):
            failed.append("prime_audit_invocations")
        if _is_int(count) and count > pc.PRIME_SAMPLER_CAPS["max_invocations"]:
            failed.append("prime_audit_invocations_cap")
    dc = obj.get("double_construction")
    if not isinstance(dc, dict):
        failed.append("double_construction")
    else:
        for key in DOUBLE_CONSTRUCTION_KEYS:
            if dc.get(key) is not True:
                failed.append("double_construction_" + key)
    for key in ("canonical_io_ok", "shape_satisfiable"):
        if obj.get(key) is not True:
            failed.append(key)
    digests = obj.get("digests")
    if not isinstance(digests, list) or len(digests) != 3 or \
            not all(pc.is_hex64(d) for d in digests):
        failed.append("digests")
    for key, value in expected.items():
        if key in obj and obj[key] != value:
            failed.append("expected_" + key)
    return failed


def validate_proof_size(obj, expected: dict, key: str, preflight) -> list:
    """`limber/poseidon-proof-size/v2`: a verified `mixed_payload_estimate` record bound to
    the config's comparison key with the commitment/eval-arg component sizes and digests and
    the structured analytical remainder."""
    failed = []
    if not isinstance(obj, dict) or obj.get("schema") != PROOF_SIZE_SCHEMA:
        return ["schema"]
    if obj.get("verified") is not True:
        failed.append("verified")
    if obj.get("metric_kind") != PROOF_SIZE_METRIC_KIND:
        failed.append("metric_kind")
    if obj.get("comparison_key") != key:
        failed.append("comparison_key")
    for name in ("config_sha256", "components", "analytical_remainder"):
        if name not in obj:
            failed.append("missing_" + name)
    components = obj.get("components")
    if not isinstance(components, dict) or any(k not in components for k in
                                               PROOF_SIZE_COMPONENT_KEYS):
        failed.append("components")
    else:
        for name in ("commitments_bytes", "eval_arg_bytes"):
            if not _is_int(components[name]) or components[name] < 0:
                failed.append("components_" + name)
        for name in ("commitments_sha256", "eval_arg_sha256"):
            if not pc.is_hex64(components[name]):
                failed.append("components_" + name)
    if not isinstance(obj.get("analytical_remainder"), dict):
        failed.append("analytical_remainder")
    if isinstance(preflight, dict) and preflight.get("config_sha256") is not None and \
            obj.get("config_sha256") != preflight.get("config_sha256"):
        failed.append("differs_from_preflight_config")
    for name, value in expected.items():
        if obj.get(name) != value:
            failed.append("expected_" + name)
    return failed


def import_attempt_dir(ctx: Context, artifact_dir: str, rc) -> dict:
    """Validate and import the allowlisted files of one form-2 attempt, then remove the
    temporary directory. Returns `{"attempt_result", "files": {name: (bytes, obj|None)},
    "conjuncts": [...]}`."""
    conjuncts = []
    files = {}
    try:
        names = sorted(os.listdir(artifact_dir))
        for name in names:
            full = os.path.join(artifact_dir, name)
            if os.path.islink(full) or not os.path.isfile(full):
                raise pc.RunnerError("preflight", "AttemptResultInvalid",
                                     "attempt directory has a non-regular entry %s" % name)
            with open(full, "rb") as f:
                files[name] = f.read()
        if "attempt-result.json" not in files:
            raise pc.RunnerError("preflight", "AttemptResultInvalid",
                                 "the bench wrote no attempt-result.json (exit %s)" % rc)
        result = pc.load_canonical_json(files["attempt-result.json"], "attempt-result.json")
        if not isinstance(result, dict) or result.get("schema") != ATTEMPT_RESULT_SCHEMA or \
                result.get("attempt") != "preflight" or result.get("mode") != ctx.mode or \
                result.get("config_sha256") != ctx.config_id:
            raise pc.RunnerError("preflight", "AttemptResultInvalid",
                                 "attempt-result.json does not describe this attempt")
        expected_outputs = [{"name": n, "size": len(files[n]), "sha256": pc.sha256_hex(files[n])}
                            for n in names if n != "attempt-result.json"]
        if result.get("outputs") != expected_outputs:
            raise pc.RunnerError("preflight", "AttemptResultInvalid",
                                 "attempt-result.json outputs do not list the written files")
        if result.get("status") not in ("ok", "failed"):
            raise pc.RunnerError("preflight", "AttemptResultInvalid", "attempt status invalid")
        if result.get("compiled") != ctx.metadata.raw:
            ctx.reason("compiled_metadata_differs", "execution")
            conjuncts.append("attempt_compiled_metadata_differs")
        allowed = set(PREFLIGHT_IMPORTS[ctx.mode])
        unexpected = [n for n in names if n != "attempt-result.json" and n not in allowed]
        if unexpected:
            raise pc.RunnerError("preflight", "AttemptResultInvalid",
                                 "attempt wrote unlisted files %s" % unexpected)
        parsed = {}
        for name in allowed:
            if name in files:
                try:
                    parsed[name] = (files[name], pc.load_canonical_json(files[name], name))
                except pc.RunnerError as exc:
                    parsed[name] = (files[name], None)
                    conjuncts.append("%s_not_canonical:%s" % (name, exc.message))
            else:
                parsed[name] = (None, None)
                conjuncts.append("%s_missing" % name)
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
    if rc != 0:
        conjuncts.append("exit_code:%s" % rc)
    if result.get("status") != "ok":
        conjuncts.append("attempt_status:%s" % result.get("error"))
    return {"attempt_result": result, "files": parsed, "conjuncts": conjuncts}


def run_preflight_attempt(ctx: Context, entry: dict) -> dict:
    """The one form-2 bench attempt for this parent config."""
    if entry["status"] != "not_run":
        raise pc.RunnerError("preflight", "PreflightFailed", "preflight already ran")
    tmp_root = ctx.path(TMP_DIR)
    os.makedirs(tmp_root, exist_ok=True)
    artifact_dir = os.path.join(tmp_root, "preflight-%d" % entry["ordinal"])
    _rmtree(artifact_dir)
    os.mkdir(artifact_dir)
    config_path = ctx.path("run-config.json")
    if pc.sha256_file(config_path) != ctx.config_id:
        raise pc.RunnerError("seal", "IntegrityMismatch", "run-config.json was modified")
    argv = [ctx.cargo()] + pc.preflight_command(ctx.features, os.path.realpath(config_path),
                                                ctx.config_id,
                                                os.path.realpath(artifact_dir))[1:]
    entry["import_status"] = "not_applicable"
    try:
        rec, _ = run_subprocess(ctx, entry, argv, ctx.env_closed)
    except pc.RunnerError:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        ctx.write_journal()
        raise
    try:
        imported = import_attempt_dir(ctx, artifact_dir, rec.get("exit"))
    except pc.RunnerError as exc:
        entry["status"] = "failed"
        entry["error"] = exc.to_json()
        ctx.reason("preflight_failed", "execution", entry["name"])
        ctx.write_journal()
        raise
    entry["attempt_result"] = imported["attempt_result"]
    return imported


def preflight_expectation(ctx: Context, workload: dict) -> dict:
    """The workload fields a preflight/proof-size record must repeat."""
    return {"backend": workload["backend"], "k": workload["k"],
            "hashes_per_field": workload["hashes_per_field"], "instance": workload["instance"]}


def evaluate_single_preflight(ctx: Context, entry: dict, imported: dict, expected: dict,
                              key: str) -> dict:
    pre_bytes, pre = imported["files"]["preflight.json"]
    size_bytes, size = imported["files"]["proof-size.json"]
    conjuncts = list(imported["conjuncts"])
    threads = ctx.config["workload"]["threads"]
    sampler = ctx.metadata.get("prime_sampler_id")
    if pre is not None:
        conjuncts.extend("preflight_" + c for c in validate_preflight(pre, expected, ctx.config_id,
                                                                      threads, sampler))
    if size is not None:
        conjuncts.extend("size_" + c for c in validate_proof_size(size, expected, key, pre))
    record = {"preflight": pre, "proof_size": size, "attempt_result": imported["attempt_result"],
              "failed_conjuncts": conjuncts, "admissible": not conjuncts, "error": None,
              "preflight_bytes": pre_bytes, "proof_size_bytes": size_bytes}
    ctx.preflight_evidence = {k: record[k] for k in ("preflight", "proof_size",
                                                       "attempt_result")}
    if record["admissible"]:
        entry["status"] = "ok"
    else:
        msg = "preflight failed: %s" % ", ".join(conjuncts)
        record["error"] = {"stage": "preflight", "error_code": "PreflightFailed", "message": msg}
        entry["status"] = "failed"
        entry["error"] = record["error"]
        entry["attempt_result"] = {"attempt_result": imported["attempt_result"],
                                   "preflight": pre, "proof_size": size}
        ctx.reason("preflight_failed", "execution", entry["name"])
    ctx.write_journal()
    return record


def evaluate_sweep_matrix(ctx: Context, entry: dict, imported: dict, workload: dict) -> dict:
    meta_bytes, meta = imported["files"]["sweep-metadata.json"]
    conjuncts = list(imported["conjuncts"])
    matrix = []
    threads = workload["threads"]
    sampler = ctx.metadata.get("prime_sampler_id")
    if not isinstance(meta, dict) or meta.get("schema") != SWEEP_METADATA_SCHEMA:
        conjuncts.append("sweep_metadata_schema")
    else:
        if meta.get("config_sha256") != ctx.config_id:
            conjuncts.append("sweep_metadata_config_sha256")
        if meta.get("candidates") != workload["candidates"] or \
                meta.get("instances") != workload["instances"]:
            conjuncts.append("sweep_metadata_lists")
        raw = meta.get("matrix")
        pairs = [(c, i) for c in workload["candidates"] for i in workload["instances"]]
        seen = []
        if not isinstance(raw, list):
            conjuncts.append("sweep_metadata_matrix")
            raw = []
        for m in raw:
            if not isinstance(m, dict):
                conjuncts.append("sweep_metadata_entry")
                continue
            cand, inst = m.get("candidate"), m.get("instance")
            seen.append((cand, inst))
            failed = []
            if (cand, inst) not in pairs:
                failed.append("pair_not_in_schedule")
            dims = workload["dimensions_by_candidate"].get(str(cand))
            if "dimensions" in m and m.get("dimensions") != dims:
                failed.append("dimensions")
            if m.get("status") != "ok":
                failed.append("status")
            failed.extend("preflight_" + c for c in validate_preflight(
                m.get("preflight"), {"backend": workload["backend"], "k": cand,
                                     "hashes_per_field": workload["hashes_per_field"],
                                     "instance": inst}, ctx.config_id, threads, sampler))
            matrix.append({"candidate": cand, "instance": inst, "dimensions": dims,
                           "status": m.get("status"), "admissible": not failed,
                           "failed_conjuncts": failed, "preflight": m.get("preflight")})
        if sorted(seen, key=str) != sorted(pairs, key=str) or len(seen) != len(pairs):
            conjuncts.append("sweep_metadata_matrix_incomplete")
    ctx.matrix = matrix
    inadmissible = [m for m in matrix if not m["admissible"]]
    record = {"matrix": matrix, "attempt_result": imported["attempt_result"],
              "failed_conjuncts": conjuncts, "admissible": not conjuncts and not inadmissible,
              "error": None, "bytes": meta_bytes}
    ctx.preflight_evidence = {"attempt_result": imported["attempt_result"], "matrix": matrix}
    if record["admissible"]:
        entry["status"] = "ok"
    else:
        first = inadmissible[0] if inadmissible else None
        msg = "preflight matrix failed: %s" % ", ".join(
            conjuncts + (["k%s inst%s: %s" % (first["candidate"], first["instance"],
                                               ", ".join(first["failed_conjuncts"]))]
                         if first else []))
        record["error"] = {"stage": "preflight", "error_code": "PreflightFailed", "message": msg}
        entry["status"] = "failed"
        entry["error"] = record["error"]
        ctx.reason("preflight_failed", "execution", entry["name"])
    ctx.write_journal()
    return record


# ---------------------------------------------------------------------------
# Criterion children (form 3).
# ---------------------------------------------------------------------------

def run_child(ctx: Context, entry: dict, final_rel: str, coordinate: dict, order,
              sequence_ordinal: int, expectation: dict) -> None:
    """Stage one Criterion child under `.tmp/`, validate, and move it into place."""
    tmp_root = ctx.path(TMP_DIR)
    os.makedirs(tmp_root, exist_ok=True)
    tmp = os.path.join(tmp_root, "child-%d" % entry["ordinal"])
    _rmtree(tmp)
    os.mkdir(tmp)
    artifact_dir = os.path.join(tmp, "artifact")
    os.mkdir(artifact_dir)
    child_config = {
        "schema": CHILD_CONFIG_SCHEMA,
        "session_id": ctx.config.get("session_id"),
        "parent_config_sha256": ctx.config_id,
        "parent_config": ctx.config,
        "system_role": ctx.role,
        "coordinate": coordinate,
        "order": order,
        "ordinal": sequence_ordinal,
    }
    cc_bytes = pc.write_canonical_json(os.path.join(tmp, "child-config.json"), child_config)
    entry["child_config_sha256"] = pc.sha256_hex(cc_bytes)
    entry["path"] = final_rel
    # The bench refuses symlinked path components: hand it realpath'ed absolute paths.
    argv = [ctx.cargo()] + pc.child_command(
        ctx.features, os.path.realpath(os.path.join(tmp, "child-config.json")),
        entry["child_config_sha256"], os.path.realpath(artifact_dir))[1:]
    try:
        try:
            rec, sink = run_subprocess(ctx, entry, argv, ctx.env_closed)
        except pc.RunnerError as exc:
            raise pc.RunnerError("child", "ChildFailed", "%s: %s" % (entry["name"],
                                                                     exc.message)) from None
        pc.write_bytes(os.path.join(tmp, "bench.log"), sink.log_bytes())
        if rec.get("exit") != 0:
            raise pc.RunnerError("child", "ChildFailed", "%s exited %s (signal %s)" %
                                 (entry["name"], rec.get("exit"), rec.get("signal")))
        produced = sorted(os.listdir(artifact_dir))
        if produced != ["criterion"] or os.path.islink(os.path.join(artifact_dir, "criterion")):
            raise pc.RunnerError("child", "CriterionTreeInvalid",
                                 "child wrote %s, expected only criterion/" % produced)
        os.rename(os.path.join(artifact_dir, "criterion"), os.path.join(tmp, "criterion"))
        os.rmdir(artifact_dir)
        pc.validate_criterion_tree(os.path.join(tmp, "criterion"), expectation)
        files = pc.hashed_file_list(os.path.join(tmp, "criterion"), "criterion/")
        child_result = {"schema": CHILD_RESULT_SCHEMA,
                        "child_config_sha256": entry["child_config_sha256"],
                        "exit_code": rec.get("exit"), "signal": rec.get("signal"),
                        "pre_audit": entry["pre_audit"], "post_audit": entry["post_audit"],
                        "log_sha256": entry["log_sha256"], "criterion_files": files}
        cr_bytes = pc.write_canonical_json(os.path.join(tmp, "child-result.json"),
                                           child_result)
        entry["child_result_sha256"] = pc.sha256_hex(cr_bytes)
        final = ctx.path(*final_rel.split("/"))
        os.makedirs(os.path.dirname(final), exist_ok=True)
        af.rename_dir_noreplace(tmp, final)
        entry["status"] = "ok"
        entry["import_status"] = "imported"
    except pc.RunnerError as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        entry["status"] = "failed"
        entry["import_status"] = "discarded"
        entry["error"] = exc.to_json()
        ctx.reason("child_failed", "execution", entry["name"])
        raise
    finally:
        ctx.write_journal()


def check_child_trees(root: str, mode: str) -> list:
    """Exact child directory set for `(mode, complete)` and per-child Criterion binding
    (plan 2300-2302, 2384-2388). Returns the child paths."""
    err = lambda msg: pc.RunnerError("seal", "FileSetMismatch", msg)  # noqa: E731
    children = os.path.join(root, "children")
    if mode == "psize":
        if os.path.lexists(children):
            raise err("psize archives have no child tree")
        return []
    if mode == "normal":
        expected = ["children/primary/%s/block-%d" % (m, b) for m in pc.PRIMARY_GROUPS
                    for b in range(6)] + ["children/diagnostic"]
    else:
        expected = ["children/block-%d/%s" % (b, t1.child_dir_name(o, c, i)) for (b, o, c, i) in
                    t1.schedule(list(pc.CANDIDATES), list(pc.TUNING_INSTANCES))]
    present = []
    for rel in pc.list_files(root):
        if rel.startswith("children/"):
            parts = rel.split("/")
            depth = 4 if mode == "normal" and parts[1] == "primary" else \
                2 if mode == "normal" else 3
            if len(parts) <= depth:
                raise err("stray child payload %s" % rel)
            present.append("/".join(parts[:depth]))
    present = sorted(set(present))
    if present != sorted(expected):
        raise err("child directories %s != expected %s" % (present, sorted(expected)))
    for rel in expected:
        child = os.path.join(root, *rel.split("/"))
        names = sorted(n for n in os.listdir(child))
        if names != sorted(list(CHILD_FILES) + ["criterion"]):
            raise err("%s contains %s" % (rel, names))
        with open(os.path.join(child, "child-result.json"), "rb") as f:
            result = pc.load_canonical_json(f.read(), rel + "/child-result.json")
        if result.get("schema") != CHILD_RESULT_SCHEMA or result.get("child_config_sha256") != \
                pc.sha256_file(os.path.join(child, "child-config.json")):
            raise err("%s/child-result.json does not bind its child config" % rel)
        if result.get("criterion_files") != pc.hashed_file_list(os.path.join(child, "criterion"),
                                                                "criterion/"):
            raise err("%s/child-result.json Criterion list differs from the tree" % rel)
    return expected


# ---------------------------------------------------------------------------
# Post-run checks, validity, sealing.
# ---------------------------------------------------------------------------

def recheck_immutables(ctx: Context) -> None:
    if pc.sha256_file(ctx.path("Cargo.lock.archived")) != ctx.config["lockfile"]["sha256"]:
        raise pc.RunnerError("seal", "IntegrityMismatch", "Cargo.lock.archived was modified")
    if pc.sha256_file(ctx.path("run-config.json")) != ctx.config_id:
        raise pc.RunnerError("seal", "IntegrityMismatch", "run-config.json was modified")
    if pc.sha256_file(ctx.path("dependency-sources.json")) != \
            ctx.config["dependency_sources"]["dependency_source_id"]:
        raise pc.RunnerError("seal", "IntegrityMismatch", "dependency-sources.json was modified")


def terminal_recheck(ctx: Context) -> None:
    recheck_immutables(ctx)
    hashes = full_recheck(ctx, "terminal")
    for e in ctx.journal:
        if e["kind"] == "evidence_metadata":
            e["attempt_result"]["terminal_recheck"] = hashes


def evaluate(ctx: Context, status: str, exit_code: int) -> dict:
    execution_valid = (status == "complete" and exit_code == 0
                       and not ctx.has_scope("execution"))
    provenance_valid = not ctx.has_scope("provenance")
    eligible = []
    if execution_valid and provenance_valid and not ctx.has_scope("role"):
        wl = ctx.config["workload"]
        canonical = (wl["hashes_per_field"] == pc.CANONICAL_HASHES and wl["threads"] == 1
                     and ctx.config["build"]["parallel"] is False
                     and ctx.config.get("features") == [])
        if ctx.mode == "sweep" and canonical:
            eligible.append("tuning")
        elif ctx.mode == "normal" and canonical and wl["instance"] == pc.HELD_OUT_INSTANCE:
            eligible.append("timing_table")
            if ctx.config.get("comparison_key") and not ctx.has_scope("size"):
                eligible.append("size_table")
        elif ctx.mode == "psize" and canonical and not ctx.has_scope("size"):
            eligible.append("size_table")
    return {"artifact_integrity_valid": True, "execution_valid": execution_valid,
            "provenance_valid": provenance_valid, "eligible_for": eligible}


def cleanup_tmp(ctx: Context) -> dict:
    return {"tmp_removed": _rmtree(ctx.path(TMP_DIR))}


def seal(ctx: Context, status: str, exit_code: int, error=None) -> str:
    """Journal, index and manifest pair inside staging (no rename)."""
    ctx.write_gates()
    ctx.write_journal()
    required, globs = expected_payload_set(ctx.mode, status)
    present = af.payload_files(ctx.staging)
    missing = [r for r in required if r not in present]
    unexpected = [p for p in present if not af._match_allowed(p, required, globs)]
    if missing or unexpected:
        raise pc.RunnerError("seal", "FileSetMismatch",
                             "payload set for (%s, %s): missing %s, unexpected %s" %
                             (status, ctx.mode, missing, unexpected))
    if status == "complete":
        check_child_trees(ctx.staging, ctx.mode)
    _, index_sha, _ = af.write_artifact_index(ctx.staging)
    validity = evaluate(ctx, status, exit_code)
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": status, "exit_code": exit_code,
        "mode": ctx.mode, "system_role": ctx.role,
        "artifact_integrity_valid": validity["artifact_integrity_valid"],
        "execution_valid": validity["execution_valid"],
        "provenance_valid": validity["provenance_valid"],
        "eligible_for": validity["eligible_for"],
        "failure_reasons": [r["code"] + (":" + json.dumps(r["detail"], sort_keys=True)
                                         if r["detail"] is not None else "")
                            for r in ctx.reasons],
        "artifact_index_sha256": index_sha,
        "config_id": ctx.config_id,
        "session_id": ctx.config.get("session_id"),
        "ids": ctx.config["ids"],
        "source_snapshot_id": ctx.config["source"]["source_snapshot_id"],
        "dependency_source_id": ctx.config["dependency_sources"]["dependency_source_id"],
        "tuning_id": ctx.tuning_id if ctx.mode == "sweep" else ctx.config.get("tuning_id"),
        "tuning_epoch_id": ctx.config.get("tuning_epoch_id"),
        "comparison_key": ctx.config.get("comparison_key"),
        "error_stage": error["stage"] if error else None,
        "error_code": error["error_code"] if error else None,
    }
    return af.seal_manifest(ctx.staging, manifest, failed=(status == "failed"))


def collect_preflight_evidence(ctx: Context) -> dict:
    """Every completed preflight output, from staging files or the journal (plan 2315)."""
    evidence = {"attempt_result": None, "preflight": None, "proof_size": None,
                "sweep_metadata": None}
    for name, key in (("preflight.json", "preflight"), ("proof-size.json", "proof_size"),
                      ("sweep-metadata.json", "sweep_metadata")):
        p = ctx.path(name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                try:
                    evidence[key] = pc.load_canonical_json(f.read(), name)
                except pc.RunnerError:
                    evidence[key] = None
    for e in ctx.journal:
        if e["kind"] == "preflight" and e.get("attempt_result") is not None:
            ar = e["attempt_result"]
            if isinstance(ar, dict) and ar.get("schema") == ATTEMPT_RESULT_SCHEMA:
                evidence["attempt_result"] = ar
            elif isinstance(ar, dict):
                evidence["attempt_result"] = ar.get("attempt_result")
                for key in ("preflight", "proof_size"):
                    if evidence[key] is None:
                        evidence[key] = ar.get(key)
    if ctx.preflight_evidence:
        for key, value in ctx.preflight_evidence.items():
            if key in evidence and evidence[key] is None:
                evidence[key] = value
    return evidence


def write_failure(ctx: Context, err: dict, coordinate, extra: dict) -> None:
    evidence = collect_preflight_evidence(ctx)
    for name in ("children", "sweep-metadata.json", "tuning-result.json", "tuning-result.sha256",
                 "preflight.json", "proof-size.json", "normal-link.json", "tuning-bundle.json"):
        p = ctx.path(name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        elif os.path.lexists(p):
            os.unlink(p)
    cleanup = cleanup_tmp(ctx)
    cleanup["child_tree_removed"] = True
    failure = {"schema": FAILURE_SCHEMA, "mode": ctx.mode, "stage": err["stage"],
               "error_code": err["error_code"], "message": err["message"],
               "coordinate": coordinate, "exit_code": err.get("exit_code"),
               "signal": err.get("signal"), "cleanup": cleanup,
               "preflight_evidence": evidence}
    if ctx.mode == "sweep":
        failure["preflight_matrix"] = ctx.matrix if ctx.matrix is not None else (
            evidence["sweep_metadata"] or {}).get("matrix")
    failure.update(extra)
    pc.write_canonical_json(ctx.path("failure.json"), failure)


def finalize_and_commit(ctx: Context, final_path: str, status: str, exit_code: int,
                        error=None) -> None:
    cleanup_tmp(ctx)
    seal(ctx, status, exit_code, error)
    af.commit_staging(ctx.staging, final_path)


# ---------------------------------------------------------------------------
# Tuning bundle / origin validation (normal and psize).
# ---------------------------------------------------------------------------

def load_bundle(path: str) -> tuple:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise pc.RunnerError("link", "BundleInvalid", "TUNING_BUNDLE: %s" % exc) from None
    obj = pc.load_canonical_json(data, "tuning-bundle.json")
    if not isinstance(obj, dict) or not isinstance(obj.get("groups"), list):
        raise pc.RunnerError("link", "BundleInvalid", "bundle lacks a groups list")
    return obj, data, pc.sha256_hex(data)


def validate_origin(store: str, group: dict) -> dict:
    """Verify one sweep origin archive in the content-addressed store (full recursive
    checks and durability confirmation)."""
    origin_sha = group.get("origin_manifest_sha256")
    tuning_id = group.get("tuning_id")
    backend = group.get("backend")
    if not pc.is_hex64(origin_sha) or not pc.is_hex64(tuning_id):
        raise pc.RunnerError("link", "BundleInvalid", "group entry lacks valid digests")
    if backend not in pc.BACKENDS:
        raise pc.RunnerError("link", "BundleInvalid", "group entry backend %r is not a limber "
                             "backend" % (backend,))
    root = os.path.join(store, origin_sha)
    if not os.path.isdir(root) or os.path.islink(root):
        raise pc.RunnerError("link", "BundleInvalid", "origin %s is not in the store" % origin_sha)
    required, globs = expected_payload_set("sweep", "complete")
    manifest, manifest_sha, status = af.verify_artifact_dir(root, required, globs)
    check_child_trees(root, "sweep")
    if manifest_sha != origin_sha:
        raise pc.RunnerError("link", "BundleInvalid", "origin manifest digest mismatch")
    if manifest.get("mode") != "sweep" or "tuning" not in manifest.get("eligible_for", []):
        raise pc.RunnerError("link", "BundleInvalid", "origin %s is not a tuning-eligible sweep"
                             % origin_sha)
    if pc.read_detached_digest(os.path.join(root, "tuning-result.sha256")) != tuning_id \
            or pc.sha256_file(os.path.join(root, "tuning-result.json")) != tuning_id \
            or manifest.get("tuning_id") != tuning_id:
        raise pc.RunnerError("link", "BundleInvalid", "tuning_id does not match the origin")
    with open(os.path.join(root, "tuning-result.json"), "rb") as f:
        result = pc.load_canonical_json(f.read(), "tuning-result.json")
    t1.recheck_tuning_result(result)
    if result["group"] != {"backend": backend}:
        raise pc.RunnerError("link", "BundleInvalid", "origin group mismatch")
    selected = group.get("selected")
    if result["decision"]["selected"] != selected or not _is_int(selected) or \
            not pc.K_RANGE[0] <= selected <= pc.K_RANGE[1]:
        raise pc.RunnerError("link", "BundleInvalid", "bundle selected candidate differs from "
                             "the origin decision")
    with open(os.path.join(root, "run-config.json"), "rb") as f:
        origin_cfg = pc.load_canonical_json(f.read(), "origin run-config.json")
    if origin_cfg.get("backend") != backend or origin_cfg.get("system_role") != backend:
        raise pc.RunnerError("link", "BundleInvalid", "origin run-config backend is not %s" %
                             backend)
    return {"backend": backend, "origin_manifest_sha256": origin_sha, "tuning_id": tuning_id,
            "selected": selected, "source_sha": origin_cfg["source"]["git_sha"],
            "source_snapshot_id": origin_cfg["source"].get("source_snapshot_id"),
            "lockfile_sha256": origin_cfg["lockfile"]["sha256"],
            "dependency_source_id": origin_cfg.get("dependency_sources", {}).get(
                "dependency_source_id")}


def resolve_tuning(ctx: Context, bundle: dict, epoch_id: str, store: str, backend: str,
                   rendered_defaults) -> dict:
    """Revalidate every origin the bundle references; resolve this backend's default.
    `rendered_defaults` is the fresh rendering of the generated module (bytes; `None`
    renders it from the bundle here) that must equal the tracked module."""
    origins = {}
    for g in bundle["groups"]:
        key = g.get("backend")
        if key in origins:
            raise pc.RunnerError("link", "BundleInvalid", "bundle lists %s twice" % (key,))
        origins[key] = validate_origin(store, g)
        if bundle.get("source_sha") != origins[key]["source_sha"] or \
                bundle.get("lockfile_sha256") != origins[key]["lockfile_sha256"]:
            raise pc.RunnerError("link", "BundleInvalid", "origin %s source/lock epoch differs "
                                 "from the bundle" % (key,))
    if backend not in origins:
        raise pc.RunnerError("link", "BundleInvalid", "bundle has no entry for backend %s" %
                             backend)
    origin = origins[backend]
    if bundle.get("lockfile_sha256") != ctx.lock_sha:
        ctx.reason("lockfile_differs_from_epoch", "role")
    module = bundle.get("generated_module_path")
    if not isinstance(module, str):
        raise pc.RunnerError("link", "BundleInvalid", "bundle lacks generated_module_path")
    if module != apply_tuning.MODULE_PATH:
        raise pc.RunnerError("link", "BundleInvalid", "bundle generated_module_path %r is not "
                             "%s" % (module, apply_tuning.MODULE_PATH))
    changed = pc.git_diff_names(ctx.repo, bundle["source_sha"], ctx.snapshot["commit"])
    if any(p != module for p in changed):
        ctx.reason("non_generated_source_diff", "role", changed)
    if rendered_defaults is None:
        try:
            rendered = apply_tuning.render(bundle)
        except pc.RunnerError as exc:
            raise pc.RunnerError("link", "BundleInvalid", "bundle cannot be rendered: %s" %
                                 exc.message) from None
    else:
        rendered = rendered_defaults
    module_path = os.path.join(ctx.repo, module)
    try:
        with open(module_path, "rb") as f:
            tracked = f.read()
        reproduced = rendered == tracked
    except OSError:
        reproduced = False
    if not reproduced:
        ctx.reason("generated_defaults_not_reproduced", "role")
    return {"tuning_id": origin["tuning_id"], "tuning_epoch_id": epoch_id,
            "selected": origin["selected"],
            "origin_manifest_sha256": origin["origin_manifest_sha256"],
            "origin_dependency_source_id": origin["dependency_source_id"],
            "origin_source_snapshot_id": origin["source_snapshot_id"],
            "generated_module_path": module, "generated_defaults_reproduced": reproduced,
            "source_diff_from_epoch": changed,
            "origins_validated": sorted(origins),
            "group_set": bundle.get("group_set")}


def read_rendered_defaults(path):
    if path is None:
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as exc:
        raise pc.RunnerError("cli", "UsageError", "--rendered-defaults: %s" % exc) from None


# ---------------------------------------------------------------------------
# Sweep mode.
# ---------------------------------------------------------------------------

def sweep_workload(backend: str, candidates: list, instances: list, dimensions: dict) -> dict:
    return {"backend": backend, "k": None, "hashes_per_field": pc.CANONICAL_HASHES,
            "threads": 1, "instance": None, "candidates": candidates, "instances": instances,
            "simplicity_order": list(pc.SIMPLICITY_ORDER),
            "dimensions_by_candidate": {str(c): dimensions[str(c)] for c in candidates}}


def sweep_main(env, repo: str) -> int:
    knobs = parse_knobs("sweep", env)
    ctx = Context("sweep", env, repo, knobs["backend"])
    ctx.knobs = knobs
    parent = check_output_parent(ctx.knobs["output_parent"], [ctx.repo])
    try:
        preconfigure(ctx, 1)
        backend = ctx.knobs["backend"]
        capability_check(ctx, backend)
        candidates, instances = list(pc.CANDIDATES), list(pc.TUNING_INSTANCES)
        orders = t1.block_orders(candidates)
        schedule = t1.schedule(candidates, instances)
        workload = sweep_workload(backend, candidates, instances, ctx.dimensions)
        build_run_config(ctx, workload, {
            "tuning_id": None, "tuning_epoch_id": None, "comparison_key": None,
            "sweep": {"candidates": candidates, "simplicity_order": list(pc.SIMPLICITY_ORDER),
                      "instances": instances,
                      "dimensions_by_candidate": workload["dimensions_by_candidate"],
                      "blocks": len(orders),
                      "block_orders": [{"block": b, "order": o} for b, o in enumerate(orders)],
                      "schedule": [{"block": b, "ordinal": o, "candidate": c, "instance": i}
                                   for (b, o, c, i) in schedule]},
            "tune1": {"group": {"backend": backend}, "candidates": candidates,
                      "simplicity_order": list(pc.SIMPLICITY_ORDER),
                      "instances": instances, "blocks": [{"block": b, "order": o}
                                                         for b, o in enumerate(orders)],
                      "statistic": "prove_e2e median", "band": ["103", "100"],
                      "wins_threshold": t1.WINS_THRESHOLD},
        })
        run_name = "run-%s-%s" % (pc.compact_timestamp(ctx.config["created_utc"]),
                                  ctx.config_id[:12])
        final = os.path.join(parent, run_name)
        staging = final + ".staging"
        if os.path.lexists(final) or os.path.lexists(staging):
            raise pc.RunnerError("staging", "StagingExists", "%s or its staging exists" % final)
        af.create_staging_exclusive(staging)
        ctx.staging = staging
    except pc.RunnerError:
        ctx.cleanup_role_dirs()
        raise
    child_plan = []
    for n, (b, o, cand, inst) in enumerate(schedule):
        rel = "children/block-%d/%s" % (b, t1.child_dir_name(o, cand, inst))
        coord = {"kind": "sweep", "block": b, "ordinal": o, "candidate": cand, "instance": inst}
        e = journal_entry(4 + n, "child", "block%d-%s" % (b, t1.child_dir_name(o, cand, inst)),
                          coord)
        e["path"] = rel
        child_plan.append((e, n + 1, b, o, cand, inst, rel))
    init_staging(ctx, [c[0] for c in child_plan], "preflight_matrix")
    try:
        try:
            run_all_gates(ctx)
            entry = ctx.entry("preflight_matrix")
            imported = run_preflight_attempt(ctx, entry)
            rec = evaluate_sweep_matrix(ctx, entry, imported, workload)
            if not rec["admissible"]:
                raise pc.RunnerError("preflight", "PreflightFailed", rec["error"]["message"])
            pc.write_bytes(ctx.path("sweep-metadata.json"), rec["bytes"])
            processes = {}
            children = []
            estimates_files = []
            for (entry, seq, b, o, cand, inst, rel) in child_plan:
                expectation = {"groups": {"prove_e2e"}, "backend": backend,
                               "hashes": pc.CANONICAL_HASHES, "instance": inst, "k": cand,
                               "threads": 1, "kind": "primary", "block": b,
                               "dimensions": ctx.dimensions[str(cand)]}
                run_child(ctx, entry, rel, entry["coordinate"], None, seq, expectation)
                recs = pc.validate_criterion_tree(ctx.path(*rel.split("/"), "criterion"),
                                                  expectation)
                est = recs[0]["estimates"]["median"]
                processes[(cand, inst, b)] = {"point": est["point_estimate"],
                                              "lower": est["lower_bound"],
                                              "upper": est["upper_bound"]}
                est_rel = rel + "/criterion/" + recs[0]["estimates_path"]
                full = ctx.path(*est_rel.split("/"))
                estimates_files.append([est_rel, os.path.getsize(full), pc.sha256_file(full)])
                children.append({"block": b, "ordinal": o, "candidate": cand, "instance": inst,
                                 "path": rel, "child_config_sha256": entry["child_config_sha256"],
                                 "child_result_sha256": entry["child_result_sha256"]})
            terminal_recheck(ctx)
            ids = dict(ctx.config["ids"]["compiled"])
            ids["tracked_specs"] = ctx.config["ids"]["tracked_specs"]
            result = t1.tuning_result({"backend": backend}, ids, candidates, instances,
                                      processes, {
                                          "run_config_sha256": ctx.config_id,
                                          "source_sha": ctx.snapshot["commit"],
                                          "source_snapshot_id":
                                              ctx.snapshot["source_snapshot_id"],
                                          "lockfile_sha256": ctx.lock_sha,
                                          "dependency_source_id":
                                              ctx.dep_audit["dependency_source_id"],
                                          "sweep_metadata_sha256": pc.sha256_hex(rec["bytes"]),
                                          "children": children,
                                          "estimates_files": sorted(estimates_files),
                                      }, simplicity_order=list(pc.SIMPLICITY_ORDER))
            result_bytes = pc.write_canonical_json(ctx.path("tuning-result.json"), result)
            ctx.tuning_id = pc.sha256_hex(result_bytes)
            pc.write_detached_digest(ctx.path("tuning-result.sha256"), ctx.tuning_id)
            t1.recheck_tuning_result(pc.load_canonical_json(result_bytes, "tuning-result.json"))
            finalize_and_commit(ctx, final, "complete", 0)
            pc.eprint("poseidon_runner: sweep complete: %s (eligible_for=%s)" %
                      (final, seal_eligibility(final)))
            return 0
        except pc.RunnerError as exc:
            return fail_single_shot(ctx, final, exc)
    finally:
        ctx.cleanup_role_dirs()


def seal_eligibility(final: str) -> list:
    with open(os.path.join(final, "manifest.json"), "rb") as f:
        return json.loads(f.read())["eligible_for"]


def fail_single_shot(ctx: Context, final: str, exc: pc.RunnerError) -> int:
    pc.eprint("poseidon_runner: error %s" % exc)
    failed_coord = None
    for e in ctx.journal:
        if e["status"] == "failed":
            failed_coord = e["coordinate"] or {"kind": e["kind"], "name": e["name"]}
    extra = {}
    if ctx.link is not None:
        extra["normal_link"] = ctx.link
    err = exc.to_json()
    for e in ctx.journal:
        if e["status"] == "failed":
            err["exit_code"], err["signal"] = e.get("exit_code"), e.get("signal")
    try:
        if exc.error_code not in ("SourceChanged",):
            try:
                terminal_recheck(ctx)
            except pc.RunnerError:
                pass
        write_failure(ctx, err, failed_coord, extra)
        finalize_and_commit(ctx, final, "failed", exc.exit_code, exc.to_json())
        pc.eprint("poseidon_runner: failed attempt archived at %s" % final)
    except pc.RunnerError as exc2:
        pc.eprint("poseidon_runner: error %s (staging %s retained)" % (exc2, ctx.staging)
                  if exc2.error_code != "DurabilityUnconfirmed" else
                  "poseidon_runner: error %s" % exc2)
        return exc2.exit_code
    return exc.exit_code


# ---------------------------------------------------------------------------
# psize mode.
# ---------------------------------------------------------------------------

def verify_session_root(session_root: str, normal_manifest_sha: str, role: str) -> dict:
    """Verify the enclosing completed session envelope of a nested normal archive:
    root pair/index/files, every nested archive's index and manifest pair, bottom-up
    fsync of the whole envelope and of its parent (plan 2350-2360)."""
    base = os.path.basename(session_root)
    if not base.startswith("session-") or base.endswith(".staging"):
        raise pc.RunnerError("link", "LinkInvalid", "%s is not a final session envelope" % base)
    root_files = sorted(n for n in os.listdir(session_root)
                        if os.path.isfile(os.path.join(session_root, n)))
    expected = sorted(SESSION_FILES + ("session-artifacts.sha256", "session-manifest.json",
                                       "session-manifest.sha256"))
    if root_files != expected:
        raise pc.RunnerError("link", "LinkInvalid", "session root files %s != %s" %
                             (root_files, expected))
    with open(os.path.join(session_root, "session-manifest.json"), "rb") as f:
        sm_bytes = f.read()
    sm_sha = pc.sha256_hex(sm_bytes)
    if pc.read_detached_digest(os.path.join(session_root, "session-manifest.sha256")) != sm_sha:
        raise pc.RunnerError("link", "LinkInvalid", "session-manifest digest mismatch")
    sm = pc.load_canonical_json(sm_bytes, "session-manifest.json")
    by_role = sm.get("parent_manifest_sha256_by_role")
    roles = sorted(pc.SESSION_SYSTEMS)
    if (sorted(sm) != ["artifact_index_sha256", "parent_manifest_sha256_by_role",
                       "session_id", "session_result_id", "status"]
            or sm.get("status") != "complete" or not isinstance(by_role, dict)
            or sorted(by_role) != roles
            or by_role[role] != normal_manifest_sha):
        raise pc.RunnerError("link", "LinkInvalid", "session manifest does not bind this "
                             "normal archive as the %s role" % role)
    with open(os.path.join(session_root, "session-artifacts.sha256"), "rb") as f:
        entries = af.parse_artifact_index(f.read())
    if sm.get("artifact_index_sha256") != pc.sha256_file(os.path.join(session_root,
                                                                     "session-artifacts.sha256")):
        raise pc.RunnerError("link", "LinkInvalid", "session artifact index digest mismatch")
    indexed = {rel for rel, _, _ in entries}
    needed = set(SESSION_FILES)
    for r in roles:
        for n in ("artifacts.sha256", "manifest.json", "manifest.sha256"):
            needed.add("systems/%s/%s" % (r, n))
    if indexed != needed:
        raise pc.RunnerError("link", "LinkInvalid", "session index covers %s, expected %s" %
                             (sorted(indexed), sorted(needed)))
    for rel, size, sha in entries:
        full = os.path.join(session_root, rel)
        if os.path.islink(full) or os.path.getsize(full) != size or pc.sha256_file(full) != sha:
            raise pc.RunnerError("link", "LinkInvalid", "session payload %s was modified" % rel)
    for name in ("comparison-session-config", "comparison-session-result"):
        digest = pc.read_detached_digest(os.path.join(session_root, name + ".sha256"))
        if pc.sha256_file(os.path.join(session_root, name + ".json")) != digest:
            raise pc.RunnerError("link", "LinkInvalid", "%s digest mismatch" % name)
    session_id = pc.read_detached_digest(os.path.join(session_root,
                                                      "comparison-session-config.sha256"))
    result_id = pc.read_detached_digest(os.path.join(session_root,
                                                     "comparison-session-result.sha256"))
    if sm.get("session_id") != session_id or sm.get("session_result_id") != result_id:
        raise pc.RunnerError("link", "LinkInvalid", "session manifest IDs differ from the files")
    systems = os.path.join(session_root, "systems")
    if sorted(os.listdir(systems)) != roles:
        raise pc.RunnerError("link", "LinkInvalid", "session systems/ is not exactly the three "
                             "roles")
    for r in roles:
        nested = os.path.join(systems, r)
        _, sha, status = af.verify_indexed_dir(nested, confirm_durability=False)
        if status != "complete" or sha != by_role[r]:
            raise pc.RunnerError("link", "LinkInvalid", "nested %s archive does not rehash to "
                                 "the root manifest" % r)
    af.fsync_tree_bottom_up(session_root)
    try:
        af.fsync_dir(os.path.dirname(session_root))
    except OSError as exc:
        raise pc.RunnerError("commit", "DurabilitySyncFailed",
                             "envelope parent fsync failed: %s" % exc) from None
    return {"session_manifest_sha256": sm_sha, "session_id": session_id,
            "session_result_id": result_id}


def psize_main(env, repo: str) -> int:
    knobs = parse_knobs("psize", env)
    normal_run = check_input_dir(knobs["normal_run"], "NORMAL_RUN")
    role = os.path.basename(normal_run)
    if os.path.basename(os.path.dirname(normal_run)) != "systems" or role not in pc.SYSTEM_ROLES:
        raise pc.RunnerError("link", "LinkInvalid",
                             "NORMAL_RUN must be <session>/systems/<%s>" % "|".join(
                                 pc.SYSTEM_ROLES))
    ctx = Context("psize", env, repo, role)
    ctx.knobs = knobs
    store = check_input_dir(ctx.knobs["tuning_store"], "TUNING_STORE")
    session_root = os.path.dirname(os.path.dirname(normal_run))
    parent = check_output_parent(ctx.knobs["output_parent"],
                                 [ctx.repo, normal_run, session_root, store])
    required, globs = expected_payload_set("normal", "complete")
    n_manifest, n_manifest_sha, _ = af.verify_artifact_dir(normal_run, required, globs)
    check_child_trees(normal_run, "normal")
    if n_manifest.get("mode") != "normal" or "timing_table" not in \
            n_manifest.get("eligible_for", []) or n_manifest.get("system_role") != role:
        raise pc.RunnerError("link", "LinkInvalid", "NORMAL_RUN is not a timing-eligible normal "
                             "%s archive" % role)
    session = verify_session_root(session_root, n_manifest_sha, role)
    for name in ("comparison-session-config.json", "comparison-session-result.json"):
        with open(os.path.join(session_root, name), "rb") as a, \
                open(os.path.join(normal_run, name), "rb") as b:
            if a.read() != b.read():
                raise pc.RunnerError("link", "LinkInvalid", "%s differs between root and nested "
                                     "archive" % name)
    with open(os.path.join(normal_run, "run-config.json"), "rb") as f:
        n_cfg = pc.load_canonical_json(f.read(), "normal run-config.json")
    if pc.read_detached_digest(os.path.join(normal_run, "run-config.sha256")) != \
            n_manifest.get("config_id"):
        raise pc.RunnerError("link", "LinkInvalid", "normal config_id mismatch")
    if n_cfg.get("system_role") != role or n_cfg.get("backend") != role:
        raise pc.RunnerError("link", "LinkInvalid", "normal run-config is not a %s config" % role)
    with open(os.path.join(normal_run, "proof-size.json"), "rb") as f:
        n_size = pc.load_canonical_json(f.read(), "normal proof-size.json")
    try:
        preconfigure(ctx, n_cfg["workload"]["threads"])
        if ctx.snapshot["commit"] != n_cfg["source"]["git_sha"]:
            raise pc.RunnerError("link", "LinkInvalid", "checkout %s differs from the normal "
                                 "run's %s" % (ctx.snapshot["commit"],
                                               n_cfg["source"]["git_sha"]))
        if ctx.snapshot["source_snapshot_id"] != n_cfg["source"]["source_snapshot_id"]:
            raise pc.RunnerError("link", "LinkInvalid", "source snapshot differs from the "
                                 "normal run")
        if ctx.lock_sha != n_cfg["lockfile"]["sha256"]:
            raise pc.RunnerError("link", "LinkInvalid", "Cargo.lock differs from the normal run")
        if ctx.dep_audit["dependency_source_id"] != \
                n_cfg["dependency_sources"]["dependency_source_id"]:
            raise pc.RunnerError("link", "LinkInvalid", "DEPENDENCY_SOURCE_ID differs from the "
                                 "normal run")
        wl_n = n_cfg["workload"]
        backend, k = wl_n["backend"], wl_n["k"]
        if backend != role or not _is_int(k) or str(k) not in ctx.dimensions:
            raise pc.RunnerError("link", "LinkInvalid", "normal workload backend/k are invalid")
        if n_cfg.get("dimensions_by_candidate") != ctx.dimensions:
            raise pc.RunnerError("link", "LinkInvalid", "compiled dimensions differ from the "
                                 "normal run")
        capability_check(ctx, backend)
        bundle, _, epoch_id = load_bundle(os.path.join(normal_run, "tuning-bundle.json"))
        if epoch_id != n_cfg.get("tuning_epoch_id"):
            raise pc.RunnerError("link", "LinkInvalid", "copied bundle hash != tuning_epoch_id")
        tuning = resolve_tuning(ctx, bundle, epoch_id, store, backend, None)
        ctx.reasons = [r for r in ctx.reasons if r["code"] != "generated_defaults_not_reproduced"]
        if tuning["tuning_id"] != n_cfg.get("tuning_id") or tuning["selected"] != k:
            raise pc.RunnerError("link", "LinkInvalid", "tuning ids/selection differ from the "
                                 "normal run")
        workload = {"backend": backend, "k": k, "hashes_per_field": wl_n["hashes_per_field"],
                    "instance": wl_n["instance"], "threads": wl_n["threads"]}
        compiled_tuning_check(ctx, backend, tuning["tuning_id"], epoch_id, k)
        key = comparison_key(ctx, workload, tuning["tuning_id"], epoch_id)
        if key != n_cfg.get("comparison_key"):
            ctx.reason("comparison_key_mismatch", "size", {"normal": n_cfg.get("comparison_key"),
                                                           "psize": key})
        ctx.link = {"normal_manifest_sha256": n_manifest_sha,
                    "session_manifest_sha256": session["session_manifest_sha256"],
                    "session_id": session["session_id"],
                    "session_result_id": session["session_result_id"],
                    "system_role": role, "comparison_key": key}
        build_run_config(ctx, workload, {"tuning_id": tuning["tuning_id"],
                                         "tuning_epoch_id": epoch_id, "comparison_key": key,
                                         "linked_normal": {"manifest_sha256": n_manifest_sha,
                                                           "config_id": n_manifest["config_id"],
                                                           "session_id": session["session_id"]}})
        run_name = "run-%s-%s" % (pc.compact_timestamp(ctx.config["created_utc"]),
                                  ctx.config_id[:12])
        final = os.path.join(parent, run_name)
        staging = final + ".staging"
        if os.path.lexists(final) or os.path.lexists(staging):
            raise pc.RunnerError("staging", "StagingExists", "%s or its staging exists" % final)
        af.create_staging_exclusive(staging)
        ctx.staging = staging
    except pc.RunnerError:
        ctx.cleanup_role_dirs()
        raise
    init_staging(ctx, [], "preflight")
    try:
        try:
            run_all_gates(ctx)
            entry = ctx.entry("preflight")
            imported = run_preflight_attempt(ctx, entry)
            rec = evaluate_single_preflight(ctx, entry, imported,
                                            preflight_expectation(ctx, workload), key)
            if not rec["admissible"]:
                raise pc.RunnerError("preflight", "PreflightFailed", rec["error"]["message"])
            pc.write_bytes(ctx.path("preflight.json"), rec["preflight_bytes"])
            pc.write_bytes(ctx.path("proof-size.json"), rec["proof_size_bytes"])
            size = rec["proof_size"]
            # Every canonical size field must reproduce; only the per-run config digest the
            # bench stamps into the file is expected to differ.
            diff = sorted(k_ for k_ in set(size) | set(n_size)
                          if k_ != "config_sha256" and size.get(k_) != n_size.get(k_))
            if diff:
                ctx.reason("size_fields_differ_from_normal", "size", diff)
            pc.write_canonical_json(ctx.path("normal-link.json"), ctx.link)
            terminal_recheck(ctx)
            finalize_and_commit(ctx, final, "complete", 0)
            pc.eprint("poseidon_runner: psize complete: %s (eligible_for=%s)" %
                      (final, seal_eligibility(final)))
            return 0
        except pc.RunnerError as exc:
            return fail_single_shot(ctx, final, exc)
    finally:
        ctx.cleanup_role_dirs()


# ---------------------------------------------------------------------------
# Normal mode (orchestrator-driven, several processes sharing one nested staging).
# ---------------------------------------------------------------------------

def load_session_spec(path: str, backend: str, threads: int, role: str) -> dict:
    if not os.path.isabs(path) or not path.endswith(".json"):
        raise pc.RunnerError("link", "SessionSpecInvalid", "SESSION_SPEC must be an absolute "
                             ".json path")
    digest_path = path[:-len(".json")] + ".sha256"
    try:
        with open(path, "rb") as f:
            data = f.read()
        digest = pc.read_detached_digest(digest_path)
    except OSError as exc:
        raise pc.RunnerError("link", "SessionSpecInvalid", str(exc)) from None
    if pc.sha256_hex(data) != digest:
        raise pc.RunnerError("link", "SessionSpecInvalid", "session spec digest mismatch")
    spec = pc.load_canonical_json(data, "comparison-session-config.json")
    try:
        if spec["hashes_per_field"] != pc.CANONICAL_HASHES or \
                spec["instance"] != pc.HELD_OUT_INSTANCE:
            raise ValueError("session H/instance are not canonical")
        if spec["threads"] != threads:
            raise ValueError("session threads %r != THREADS %r" % (spec["threads"], threads))
        slot = spec["systems"][role]
        if role != backend or slot.get("backend") != backend:
            raise ValueError("session slot %s is not backend %s" % (role, backend))
        if spec["schedule"]["metric_order"] != list(pc.PRIMARY_GROUPS):
            raise ValueError("session metric order is not the literal primary order")
        if spec["schedule"]["block_orders"] != [list(o) for o in pc.SESSION_BLOCK_ORDERS]:
            raise ValueError("session block orders differ from the literal schedule")
        if "session_id" in spec:
            raise ValueError("session spec must not carry session_id")
    except (KeyError, TypeError, ValueError) as exc:
        raise pc.RunnerError("link", "SessionSpecInvalid", str(exc)) from None
    return {"bytes": data, "digest": digest, "obj": spec, "slot": slot, "path": path,
            "digest_bytes": pc.format_detached_digest(digest)}


def normal_child_plan() -> list:
    plan = []
    for metric in pc.PRIMARY_GROUPS:
        for b in range(6):
            plan.append(({"kind": "primary", "metric": metric, "block": b},
                         "children/primary/%s/block-%d" % (metric, b)))
    plan.append(({"kind": "diagnostic", "metric": None, "block": "diag"}, "children/diagnostic"))
    return plan


def check_work_dir(path: str, repo: str, must_exist: bool) -> str:
    if not os.path.isabs(path):
        raise pc.RunnerError("staging", "OutputParentInvalid", "--staging must be absolute")
    if _overlaps(os.path.realpath(path), os.path.realpath(repo)):
        raise pc.RunnerError("staging", "PathOverlap", "%s overlaps the source worktree" % path)
    if must_exist:
        if not os.path.isdir(path) or os.path.islink(path):
            raise pc.RunnerError("staging", "OutputParentInvalid",
                                 "%s must be an existing directory" % path)
    elif not os.path.isdir(os.path.dirname(path)):
        raise pc.RunnerError("staging", "OutputParentInvalid",
                             "the parent of %s must exist" % path)
    return path.rstrip("/")


def normal_preconfigure(args, env, repo: str) -> int:
    knobs = parse_knobs("normal", env, session_spec_required=False)
    ctx = Context("normal", env, repo, knobs["backend"])
    ctx.knobs = knobs
    work = check_work_dir(args.staging, repo, must_exist=False)
    af.create_staging_exclusive(work)
    try:
        preconfigure(ctx, ctx.knobs["threads"])
        capability_check(ctx, ctx.knobs["backend"])
        write_evidence(ctx, work)
        ctx.write_state(os.path.join(work, STATE_FILE))
        identity = {
            "schema": PRECONFIGURE_SCHEMA, "system_role": ctx.role,
            "repository": pc.REPOSITORY_NAME, "backend": ctx.knobs["backend"],
            "threads": ctx.threads, "features": ctx.features,
            "git_sha": ctx.snapshot["commit"], "tree": ctx.snapshot["tree"],
            "dirty": ctx.snapshot["dirty"],
            "source_snapshot_id": ctx.snapshot["source_snapshot_id"],
            "lockfile_sha256": ctx.lock_sha,
            "dependency_source_id": ctx.dep_audit["dependency_source_id"],
            "toolchain_closure_sha256": ctx.toolchain["toolchain_closure_sha256"],
            "native_closure_sha256": ctx.native["native_closure_sha256"],
            "compiled": ctx.metadata.compiled_ids(), "executable": ctx.executable,
            "dimensions_by_candidate": ctx.dimensions,
            "role_target": ctx.role_target, "role_temp": ctx.role_temp,
            "failure_reasons": ctx.reasons,
        }
        pc.write_canonical_json(os.path.join(work, "preconfigure.json"), identity)
    except pc.RunnerError:
        ctx.cleanup_role_dirs()
        raise
    print("preconfigure %s" % os.path.join(work, "preconfigure.json"))
    return 0


def normal_prepare(args, env, repo: str) -> int:
    knobs = parse_knobs("normal", env)
    ctx = Context("normal", env, repo, knobs["backend"])
    work = check_work_dir(args.preconfigured, repo, must_exist=True)
    staging = check_work_dir(args.staging, repo, must_exist=False)
    rendered_defaults = read_rendered_defaults(args.rendered_defaults)
    ctx.load_state(os.path.join(work, STATE_FILE))
    if ctx.mode != "normal" or ctx.repo != repo:
        raise pc.RunnerError("staging", "OutputParentInvalid", "pre-configuration state was "
                             "made for another mode/repository")
    for key in ("backend", "allow_dirty"):
        if ctx.knobs.get(key) != knobs.get(key):
            raise pc.RunnerError("env", "InvalidKnob", "%s differs from the pre-configured "
                                 "value %r" % (key, ctx.knobs.get(key)))
    ctx.knobs = knobs
    backend, threads = ctx.knobs["backend"], ctx.knobs["threads"]
    if threads != ctx.threads:
        raise pc.RunnerError("env", "InvalidKnob", "THREADS %d differs from the pre-configured "
                             "%d" % (threads, ctx.threads))
    load_specs(ctx)
    read_evidence(ctx, work)
    with open(os.path.join(repo, "Cargo.lock"), "rb") as f:
        ctx.lock_bytes = f.read()
    if pc.sha256_hex(ctx.lock_bytes) != ctx.lock_sha:
        raise pc.RunnerError("source", "SourceChanged", "Cargo.lock changed since the snapshot")
    full_recheck(ctx, "prepare")
    bind_spec_checks(ctx)
    metadata_value_check(ctx)
    if ctx.knobs["allow_dirty"]:
        ctx.reason("dirty_override_active", "role")
    if ctx.snapshot["dirty"]:
        ctx.reason("dirty_worktree_allowed", "provenance")
    session = load_session_spec(ctx.knobs["session_spec"], backend, threads, ctx.role)
    store = check_input_dir(ctx.knobs["tuning_store"], "TUNING_STORE")
    capability_check(ctx, backend)
    bundle, bundle_bytes, epoch_id = load_bundle(ctx.knobs["tuning_bundle"])
    tuning = resolve_tuning(ctx, bundle, epoch_id, store, backend, rendered_defaults)
    slot = session["slot"]
    for key, value in (("tuning_id", tuning["tuning_id"]), ("tuning_epoch_id", epoch_id),
                       ("selected", tuning["selected"]), ("k", tuning["selected"])):
        if key in slot and slot[key] != value:
            raise pc.RunnerError("link", "SessionSpecInvalid", "session %s differs from the "
                                 "bundle" % key)
    for key, value in (("source_snapshot_id", ctx.snapshot["source_snapshot_id"]),
                       ("dependency_source_id", ctx.dep_audit["dependency_source_id"]),
                       ("lockfile_sha256", ctx.lock_sha)):
        if key in slot and slot[key] != value:
            raise pc.RunnerError("link", "SessionSpecInvalid", "session %s differs from this "
                                 "pre-configuration" % key)
    if threads != 1:
        ctx.reason("threads_not_one", "role", threads)
    k = tuning["selected"]
    workload = {"backend": backend, "k": k, "hashes_per_field": pc.CANONICAL_HASHES,
                "instance": pc.HELD_OUT_INSTANCE, "threads": threads}
    compiled_tuning_check(ctx, backend, tuning["tuning_id"], epoch_id, k)
    key = comparison_key(ctx, workload, tuning["tuning_id"], epoch_id)
    build_run_config(ctx, workload, {
        "tuning_id": tuning["tuning_id"], "tuning_epoch_id": epoch_id, "comparison_key": key,
        "session_id": session["digest"],
        "tuning_origin": {k_: tuning[k_] for k_ in ("origin_manifest_sha256",
                                                     "origin_dependency_source_id",
                                                     "origin_source_snapshot_id",
                                                     "generated_module_path",
                                                     "generated_defaults_reproduced",
                                                     "source_diff_from_epoch",
                                                     "origins_validated", "group_set")},
        "session_schedule": {"metric_order": list(pc.PRIMARY_GROUPS),
                             "block_orders": [list(o) for o in pc.SESSION_BLOCK_ORDERS],
                             "diagnostic_order": list(pc.diagnostic_groups_for(backend))},
    })
    af.create_staging_exclusive(staging)
    ctx.staging = staging
    child_entries = []
    for i, (coord, rel) in enumerate(normal_child_plan()):
        name = ("%s-block%d" % (coord["metric"], coord["block"]) if coord["kind"] == "primary"
                else "diagnostic")
        e = journal_entry(4 + i, "child", name, coord)
        e["path"] = rel
        child_entries.append(e)
    os.mkdir(ctx.path(TMP_DIR))
    ctx.write_state(ctx.path(TMP_DIR, STATE_FILE))
    init_staging(ctx, child_entries, "preflight", before_config=(
        ("tuning-bundle.json", bundle_bytes),
        ("comparison-session-config.json", session["bytes"]),
        ("comparison-session-config.sha256", session["digest_bytes"])))
    pc.eprint("poseidon_runner: normal prepare complete at %s (config_id %s)" %
              (staging, ctx.config_id))
    return 0


def load_context(staging: str, env, repo: str) -> Context:
    if not os.path.isabs(staging) or not os.path.isdir(staging):
        raise pc.RunnerError("staging", "OutputParentInvalid", "--staging must be an existing "
                             "absolute directory")
    role = pc.knob_enum(env, "BACKEND", pc.BACKENDS)
    ctx = Context("normal", env, repo, role)
    ctx.staging = staging
    ctx.load_state(ctx.path(TMP_DIR, STATE_FILE))
    if ctx.repo != repo:
        raise pc.RunnerError("staging", "OutputParentInvalid", "state repository differs")
    with open(ctx.path("run-config.json"), "rb") as f:
        ctx.config_bytes = f.read()
    ctx.config_id = pc.sha256_hex(ctx.config_bytes)
    if pc.read_detached_digest(ctx.path("run-config.sha256")) != ctx.config_id:
        raise pc.RunnerError("seal", "IntegrityMismatch", "run-config.json does not match its "
                             "detached digest")
    ctx.config = pc.load_canonical_json(ctx.config_bytes, "run-config.json")
    if ctx.config.get("mode") != "normal" or ctx.config.get("system_role") != ctx.role:
        raise pc.RunnerError("cli", "UsageError", "staging is not a normal-mode %s archive" %
                             ctx.role)
    with open(ctx.path("process-journal.json"), "rb") as f:
        journal = pc.load_canonical_json(f.read(), "process-journal.json")
    ctx.journal = journal["entries"]
    ctx.reasons = journal["failure_reasons"]
    with open(ctx.path("gate-results.json"), "rb") as f:
        ctx.gates = pc.load_canonical_json(f.read(), "gate-results.json")
    load_specs(ctx)
    read_evidence(ctx, staging)
    with open(ctx.path("Cargo.lock.archived"), "rb") as f:
        ctx.lock_bytes = f.read()
    recheck_immutables(ctx)
    return ctx


def normal_gate(args, env, repo: str) -> int:
    ctx = load_context(args.staging, env, repo)
    try:
        if not run_gate(ctx, args.gate):
            raise pc.RunnerError("gates", "GateFailed", "gate %s failed (recorded)" % args.gate)
    finally:
        ctx.write_gates()
        ctx.write_journal()
    pc.eprint("poseidon_runner: gate %s passed" % args.gate)
    return 0


def normal_preflight(args, env, repo: str) -> int:
    ctx = load_context(args.staging, env, repo)
    for name in pc.GATE_ORDER:
        if ctx.gates[name]["status"] != "ok":
            raise pc.RunnerError("preflight", "PreflightFailed",
                                 "gate %s has not passed" % name)
    entry = ctx.entry("preflight")
    wl = ctx.config["workload"]
    try:
        imported = run_preflight_attempt(ctx, entry)
        rec = evaluate_single_preflight(ctx, entry, imported, preflight_expectation(ctx, wl),
                                        ctx.config["comparison_key"])
        if not rec["admissible"]:
            raise pc.RunnerError("preflight", "PreflightFailed", rec["error"]["message"])
        pc.write_bytes(ctx.path("preflight.json"), rec["preflight_bytes"])
        pc.write_bytes(ctx.path("proof-size.json"), rec["proof_size_bytes"])
    finally:
        ctx.write_journal()
    pc.eprint("poseidon_runner: preflight complete at %s" % ctx.staging)
    return 0


def normal_child(args, env, repo: str) -> int:
    ctx = load_context(args.staging, env, repo)
    try:
        order = json.loads(args.order)
    except ValueError as exc:
        raise pc.RunnerError("cli", "UsageError", "--order must be a JSON list: %s" % exc)
    if not isinstance(order, list) or sorted(order) != sorted(pc.SESSION_SYSTEMS):
        raise pc.RunnerError("cli", "UsageError", "--order must list the three systems")
    wl = ctx.config["workload"]
    if args.kind == "primary":
        if args.metric is None or args.block is None:
            raise pc.RunnerError("cli", "UsageError", "primary children need --metric and "
                                 "--block")
        coord = {"kind": "primary", "metric": args.metric, "block": args.block}
        rel = "children/primary/%s/block-%d" % (args.metric, args.block)
        groups, block = [args.metric], args.block
    else:
        if args.metric is not None or args.block is not None:
            raise pc.RunnerError("cli", "UsageError", "diagnostic children take no --metric/"
                                 "--block")
        coord = {"kind": "diagnostic", "metric": None, "block": "diag"}
        rel = "children/diagnostic"
        groups, block = list(pc.diagnostic_groups_for(wl["backend"])), "diag"
    if ctx.entry("preflight")["status"] != "ok":
        raise pc.RunnerError("child", "ChildFailed", "the preflight attempt has not succeeded")
    entry = next((e for e in ctx.journal if e["coordinate"] == coord), None)
    if entry is None or entry["status"] != "not_run":
        raise pc.RunnerError("child", "ChildFailed", "coordinate %s is not a planned not_run "
                             "child" % coord)
    entry["session_ordinal"] = args.ordinal
    expectation = {"groups": set(groups), "backend": wl["backend"],
                   "hashes": wl["hashes_per_field"], "instance": wl["instance"],
                   "k": wl["k"], "threads": wl["threads"], "kind": coord["kind"],
                   "block": block, "dimensions": ctx.config["dimensions"]}
    sequence = [e for e in ctx.journal if e["kind"] == "child"].index(entry) + 1
    try:
        run_child(ctx, entry, rel, coord, order, sequence, expectation)
    finally:
        ctx.write_journal()
    pc.eprint("poseidon_runner: child %s imported at %s" % (entry["name"], rel))
    return 0


def normal_seal(args, env, repo: str) -> int:
    ctx = load_context(args.staging, env, repo)
    result_path = args.session_result
    if not os.path.isabs(result_path) or not result_path.endswith(".json"):
        raise pc.RunnerError("cli", "UsageError", "--session-result must be an absolute .json")
    with open(result_path, "rb") as f:
        result_bytes = f.read()
    result_digest = pc.read_detached_digest(result_path[:-len(".json")] + ".sha256")
    if pc.sha256_hex(result_bytes) != result_digest:
        raise pc.RunnerError("link", "SessionSpecInvalid", "session result digest mismatch")
    result = pc.load_canonical_json(result_bytes, "comparison-session-result.json")
    if result.get("session_id") != ctx.config.get("session_id"):
        raise pc.RunnerError("link", "SessionSpecInvalid", "session result session_id differs")
    pc.write_bytes(ctx.path("comparison-session-result.json"), result_bytes)
    pc.write_detached_digest(ctx.path("comparison-session-result.sha256"), result_digest)
    try:
        try:
            terminal_recheck(ctx)
        except pc.RunnerError as exc:
            if not args.failure:
                raise
            pc.eprint("poseidon_runner: terminal recheck failed: %s" % exc)
        if args.failure:
            with open(args.failure, "rb") as f:
                info = json.loads(f.read().decode("utf-8"))
            err = {"stage": info.get("stage", "child"),
                   "error_code": info.get("error_code", "ChildFailed"),
                   "message": info.get("message", "session failure"),
                   "exit_code": info.get("exit_code"), "signal": info.get("signal")}
            if err["stage"] not in pc.STAGES or err["error_code"] not in pc.ERROR_CODES:
                raise pc.RunnerError("cli", "UsageError", "--failure carries unknown enums")
            ctx.reason("session_failed", "execution", info.get("coordinate"))
            write_failure(ctx, err, info.get("coordinate"),
                          {"session_result_id": result_digest})
            seal(ctx, "failed", pc.ERROR_CODES[err["error_code"]], err)
            pc.eprint("poseidon_runner: failed normal archive sealed at %s" % ctx.staging)
            return 0
        not_ok = [e["name"] for e in ctx.journal if e["status"] != "ok"]
        if not_ok:
            raise pc.RunnerError("seal", "SealFailed", "cannot seal success with entries %s; "
                                 "use --failure" % not_ok)
        children = [e for e in ctx.journal if e["kind"] == "child"]
        if len(children) != pc.NORMAL_CHILDREN or sum(
                1 for e in children if e["coordinate"]["kind"] == "primary") != \
                pc.NORMAL_PRIMARY_CHILDREN:
            raise pc.RunnerError("seal", "SealFailed", "expected %d primary + 1 diagnostic "
                                 "children" % pc.NORMAL_PRIMARY_CHILDREN)
        entries = result.get("entries")
        if isinstance(entries, list):
            mine = [e for e in entries if e.get("system") == ctx.role]
            if len(mine) != pc.NORMAL_CHILDREN or any(e.get("status") != "ok" for e in mine):
                ctx.reason("session_result_not_ok", "role")
        cleanup_tmp(ctx)
        seal(ctx, "complete", 0)
        pc.eprint("poseidon_runner: normal archive sealed at %s" % ctx.staging)
        return 0
    finally:
        ctx.cleanup_role_dirs()


def normal_cleanup(args, env, repo: str) -> int:
    """Remove the role target/temp directories recorded in a work dir or nested staging."""
    ctx = Context("normal", env, repo, pc.knob_enum(env, "BACKEND", pc.BACKENDS))
    for candidate in (os.path.join(args.staging, STATE_FILE),
                      os.path.join(args.staging, TMP_DIR, STATE_FILE)):
        if os.path.isfile(candidate):
            ctx.load_state(candidate)
            out = ctx.cleanup_role_dirs()
            pc.eprint("poseidon_runner: cleanup %s: %s" % (args.staging, out))
            return 0
    pc.eprint("poseidon_runner: cleanup %s: no state (nothing to remove)" % args.staging)
    return 0


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poseidon_runner.py",
        description="limber Poseidon2 benchmark runner. Without a subcommand the mode comes "
                    "from SWEEP=1 / PSIZE=1 (see the module docstring).")
    parser.add_argument("--repo", default=pc.repo_root(), help="repository checkout")
    sub = parser.add_subparsers(dest="command")
    normal = sub.add_parser("normal", help="orchestrator-driven normal mode")
    nsub = normal.add_subparsers(dest="step", required=True)
    pre = nsub.add_parser("preconfigure")
    pre.add_argument("--staging", required=True, help="pre-configuration work directory")
    prep = nsub.add_parser("prepare")
    prep.add_argument("--staging", required=True)
    prep.add_argument("--preconfigured", required=True)
    prep.add_argument("--rendered-defaults")
    gate = nsub.add_parser("gate")
    gate.add_argument("--staging", required=True)
    gate.add_argument("--gate", required=True, choices=list(pc.GATE_ORDER))
    pf = nsub.add_parser("preflight")
    pf.add_argument("--staging", required=True)
    child = nsub.add_parser("child")
    child.add_argument("--staging", required=True)
    child.add_argument("--kind", required=True, choices=["primary", "diagnostic"])
    child.add_argument("--metric", choices=list(pc.PRIMARY_GROUPS))
    child.add_argument("--block", type=int, choices=list(range(6)))
    child.add_argument("--order", required=True)
    child.add_argument("--ordinal", required=True, type=int)
    sealp = nsub.add_parser("seal")
    sealp.add_argument("--staging", required=True)
    sealp.add_argument("--session-result", required=True)
    sealp.add_argument("--failure")
    clean = nsub.add_parser("cleanup")
    clean.add_argument("--staging", required=True)
    return parser


def main(argv=None, env=None) -> int:
    env = dict(os.environ if env is None else env)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        repo = check_repo(args.repo)
        if args.command == "normal":
            steps = {"preconfigure": normal_preconfigure, "prepare": normal_prepare,
                     "gate": normal_gate, "preflight": normal_preflight, "child": normal_child,
                     "seal": normal_seal, "cleanup": normal_cleanup}
            return steps[args.step](args, env, repo)
        mode = pc.select_mode(env)
        if mode == "sweep":
            return sweep_main(env, repo)
        if mode == "psize":
            return psize_main(env, repo)
        raise pc.RunnerError("cli", "UsageError", "normal mode is driven by the session "
                             "orchestrator: use `poseidon_runner.py normal preconfigure|prepare"
                             "|gate|preflight|child|seal` or set SWEEP=1 / PSIZE=1")
    except pc.RunnerError as exc:
        pc.eprint("poseidon_runner: error %s" % exc)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
