#!/usr/bin/env python3
"""Shared helpers for the Poseidon2 limber benchmark tooling (Zinc plan v10, section 9;
limber amendment contract sections 1, 4, 5, 6).

This is the limber adaptation of the Zinc+ `scripts/poseidon_common.py`. The module
structure and every public function name are kept so that the Zinc session orchestrator and
publisher interoperate with the limber runner; the differences are collected in the "system
profile" block below. Summary of the limber differences:

  * Roles/systems are the two limber backends `hyrax` and `brakedown` (`SYSTEM_ROLES`); the
    workload knob is `BACKEND` (replacing Zinc's `VARIANT`/`FL`); the candidate parameter
    is the integer `k` (7..=13) in the pinned TUNE-1 order `K_ORDER`, with ascending `k` as
    the simplicity order (`SIMPLICITY_ORDER`), instead of Zinc's int-PP shapes.
  * Canonical commands name the `poseidon_modp` bench of the single `limber` package (no
    `-p`), the gates are `poseidon2::tests::kat_gate` / `poseidon2::tests::tuning_corpus_gate`
    of the crate's `--lib` tests, and there is no crate feature for any thread count: the
    thread count reaches the bench only through `RAYON_NUM_THREADS` (`T > 1` is exploratory
    and ineligible; `features_for_threads` is always `[]`).
  * Compiled metadata keys (`METADATA_REQUIRED_KEYS`): no `checked_flag`/`unchecked_flag`
    (their presence is a recorded execution failure), `check_mode == "not_applicable"`,
    `benchmark_coins_id`, `comparison_schema_id`, `dimensions_by_k` (`{k: {log_cons,
    log_vars}}` for H = 10, copied into the run config as `dimensions_by_candidate`).
  * The approved capability tuple is `(protocol_wire_id, prime_sampler_id,
    benchmark_coins_id)`; `common_lift_binding_id` is not applicable (null when present).
  * Check modes are `{shadow: "not_applicable", timed: "release"}`; the preflight is a
    deterministic-coins double construction with the P0-D prime audit and
    `rayon_threads_asserted`; the proof-size metric is `mixed_payload_estimate`.
  * Criterion IDs follow the limber grammar (contract section 5): `{backend}/mixed3/Hpf{H}
    -total{3H}/c2^{log_cons}v2^{log_vars}/k{k}/inst{i}/thr{T}/{primary/blk{b}|diagnostic/
    blkdiag}`; diagnostic groups are `setup, advice (hyrax only), commit_witness,
    prove_after_input_commit` (`diagnostic_groups_for(backend)`).
  * The tune corpus lives at `tests/data/tune-corpus-v1.json`; the security file is
    limber's own `specs/poseidon/security-accounting-v1.json`; the shared spec files are
    byte-identical copies of Zinc's.
  * The rustc argv grammar for the limber bench is read from the shared timing schema under
    `rustc_argv_grammar.roles.<GRAMMAR_ROLE>`; the evidence build fails closed while that
    role is absent (see `poseidon_build_env`).

Standard library only. Everything that decides an outcome is exact: JSON numbers from
Criterion are parsed as `Fraction`, ratios are reduced integer pairs, medians of even counts
are exact means, and every JSON file this tooling writes goes through `canonical_json_bytes`
(sorted keys, no insignificant whitespace, ASCII, one trailing LF, floats forbidden).

There are no environment-based test hooks: the runner's only environment inputs are the
mode knobs in `KNOB_NAMES`; build/toolchain/environment facts come from
`poseidon_build_env` (package C), which tests replace by monkeypatching the `be` module
attribute of the importing module with `scripts/tests/stub_build_env.py`. This module owns
the `STAGES` and `ERROR_CODES` tables; package C adds codes only inside its reserved section.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import re
import stat as _stat
import subprocess
import sys
import time
import tomllib
from fractions import Fraction

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Stable enums: stages, error codes, exit codes (the single place for them).
# ---------------------------------------------------------------------------

STAGES = (
    "cli", "env", "source", "toolchain", "dependency", "evidence", "metadata", "specs",
    "config", "staging", "gates", "preflight", "child", "aggregate", "link", "seal", "commit",
    "durability", "internal",
)

# error_code -> process exit code
ERROR_CODES = {
    "UsageError": 2, "InvalidKnob": 2, "RejectedKnob": 2, "ModeConflict": 2,
    "RejectedEnvironment": 2,
    "DirtyWorktree": 3, "LockfileUntracked": 3, "SourceChanged": 3, "GitCommandFailed": 3,
    "ToolchainUnavailable": 3, "DependencyMismatch": 3, "SourceSnapshotInvalid": 3,
    "MetadataUnavailable": 4, "MetadataMalformed": 4,
    "SpecMissing": 5, "SpecMismatch": 5, "DigestMalformed": 5, "IntegrityMismatch": 5,
    "UnsafePath": 5, "SymlinkRejected": 5, "FileSetMismatch": 5, "ManifestInvalid": 5,
    "CanonicalJsonInvalid": 5,
    "OutputParentInvalid": 6, "PathOverlap": 6, "StagingExists": 6, "RenameRefused": 6,
    "UnsupportedPlatform": 6, "TempDirInvalid": 6,
    "GateFailed": 7,
    "PreflightFailed": 8, "AttemptResultInvalid": 8,
    "ChildFailed": 9, "CriterionTreeInvalid": 9,
    "TuningResultInvalid": 10, "LinkInvalid": 10, "SessionSpecInvalid": 10, "BundleInvalid": 10,
    "SealFailed": 11, "DurabilitySyncFailed": 11,
    "DurabilityUnconfirmed": 12,
    "InternalError": 70,
    # --- build-env codes --- (package C: poseidon_build_env; add codes only here)
    "CargoConfigRejected": 3, "CargoConfigChanged": 3, "DependencySourceInvalid": 3,
    "DependencyClosureChanged": 3, "MetadataReplayMismatch": 3, "ToolchainMismatch": 3,
    "ToolchainChanged": 3, "NativeToolInvalid": 3, "NativeClosureChanged": 3,
    "EvidenceBuildFailed": 4, "ExecutableChanged": 4, "RustcArgvInvalid": 4,
    "IdentityProjectionInvalid": 4,
}


class RunnerError(Exception):
    """A typed, stable error: `(stage, error_code, message)`."""

    def __init__(self, stage: str, error_code: str, message: str):
        if stage not in STAGES:
            raise ValueError("unknown stage %r" % (stage,))
        if error_code not in ERROR_CODES:
            raise ValueError("unknown error code %r" % (error_code,))
        super().__init__("[%s/%s] %s" % (stage, error_code, message))
        self.stage = stage
        self.error_code = error_code
        self.message = message

    @property
    def exit_code(self) -> int:
        return ERROR_CODES[self.error_code]

    def to_json(self) -> dict:
        return {"stage": self.stage, "error_code": self.error_code, "message": self.message}


# ---------------------------------------------------------------------------
# System profile: everything that distinguishes limber from Zinc+ (contract sections 1-6).
# ---------------------------------------------------------------------------

SYSTEM = "limber"
REPOSITORY_NAME = "limber-impl"
SYSTEM_ROLES = ("hyrax", "brakedown")
BACKENDS = SYSTEM_ROLES
BACKEND_TAGS = {"hyrax": 0, "brakedown": 1}
# `VARIANTS` keeps its Zinc name for the orchestrator/publisher: the compiled metadata's
# `variants` list is the backend list.
VARIANTS = BACKENDS
# TUNE-1 candidate parameter: the integer k in the pinned literal order (tuning protocol
# `systems.limber.k_order`); simplicity order is ascending k.
# Single-candidate TUNE-1 epoch (tuning-protocol-v2 `single_candidate_epoch`): the
# pinned candidate order is just the persisted default k = 9; K_RANGE stays the
# admissible range for IDs and configs.
K_ORDER = (9,)
K_RANGE = (7, 13)
CANDIDATES = K_ORDER
SIMPLICITY_ORDER = tuple(sorted(K_ORDER))
TUNING_INSTANCES = tuple(range(1, 10))
HELD_OUT_INSTANCE = 0
CANONICAL_HASHES = 10
# Plan v10: only prove_e2e and verify_core have aligned cross-system boundaries.
PRIMARY_GROUPS = ("prove_e2e", "verify_core")
# Diagnostic groups in literal registration order; `advice` exists for hyrax only.
DIAGNOSTIC_GROUPS = ("setup", "advice", "commit_witness", "prove_after_input_commit")
HYRAX_ONLY_DIAGNOSTICS = ("advice",)


def diagnostic_groups_for(backend: str) -> tuple:
    """The literal diagnostic group order a backend's diagnostic child registers."""
    if backend not in BACKENDS:
        raise RunnerError("config", "InvalidKnob", "unknown backend %r" % (backend,))
    if backend == "hyrax":
        return DIAGNOSTIC_GROUPS
    return tuple(g for g in DIAGNOSTIC_GROUPS if g not in HYRAX_ONLY_DIAGNOSTICS)


# Normal-archive child counts: two comparable metrics x six blocks + one diagnostic.
NORMAL_PRIMARY_CHILDREN = len(PRIMARY_GROUPS) * 6
NORMAL_CHILDREN = NORMAL_PRIMARY_CHILDREN + 1
# Session-wide children: three systems.
SESSION_CHILDREN = 3 * NORMAL_CHILDREN
ALL_GROUPS = DIAGNOSTIC_GROUPS + PRIMARY_GROUPS
SESSION_SYSTEMS = ("zinc", "hyrax", "brakedown")
SESSION_BLOCK_ORDERS = (
    ("zinc", "hyrax", "brakedown"), ("zinc", "brakedown", "hyrax"),
    ("hyrax", "zinc", "brakedown"), ("hyrax", "brakedown", "zinc"),
    ("brakedown", "zinc", "hyrax"), ("brakedown", "hyrax", "zinc"),
)
CRITERION_SETTINGS = {
    "version": "0.7.0", "sample_size": 10, "warm_up_time_s": 1, "measurement_time_s": 20,
    "default_features": False, "features": ["cargo_bench_support"],
    "one_group_per_primary_process": True,
}
RUSTFLAGS = "-C target-cpu=native"
# limber has no crate feature for the benchmark and no parallel feature flag: the thread
# count reaches the bench only through RAYON_NUM_THREADS. `PARALLEL_FEATURE` is kept (as
# None) so callers written against the Zinc module keep working.
BENCH_FEATURES = []
PARALLEL_FEATURE = None
CHECK_MODES = {"shadow": "not_applicable", "timed": "release"}
CHECK_MODE_METADATA = "not_applicable"
# Compiled identifiers (contract section 1) the runner checks against the bench.
PROTOCOL_WIRE_ID = "limber/wire/v-p0d"
TRANSCRIPT_ID = "limber/transcript/keccak256-p0d"
PRIME_SAMPLER_ID = "limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096"
PRIME_SAMPLER_CAPS = {"t_max": 4096, "mr_rounds": 72, "base_draw_max": 160,
                      "max_invocations": 4096}
BENCHMARK_COINS_ID = "poseidon-bench-chacha20-v1"
# The tuning protocol's framing string (tuning-protocol-v2.json
# `systems.limber.benchmark_coins.framing`); the runner reads it from the tracked spec copy
# and falls back to this literal only for a spec that predates the key.
BENCHMARK_COINS_FRAMING = ('"limber-poseidon2-v1/bench-coins/v1\\0" || backend_u8 || '
                           'candidate_index_le32 || instance_le32')
KAT_FIXTURE_SHA256 = "db2d7fd7e3de653c8813b21be8ee1ffb9234dc36505bbfc0edd85af41bbed0bd"
# Bench target of the single `limber` package (contract sections 4 and 6).
PACKAGE_NAME = "limber"
BENCH_NAME = "poseidon_modp"
BENCH_SRC_PATH = "benches/poseidon_modp.rs"
# Role key under `rustc_argv_grammar.roles` of the shared timing schema for the limber bench
# (one binary serves both backends). The integrator captures the grammar from a real
# evidence build and adds it there; until then the evidence build fails closed.
GRAMMAR_ROLE = "limber"
# limber has no zstd dependency: the bundled-zstd evidence requirement is Zinc-only.
REQUIRE_BUNDLED_ZSTD = False
# Exact canonical commands (contract sections 3, 4 and 6). `cargo` stands for the resolved
# `<SYSROOT>/bin/cargo`; the runner substitutes the resolved path in argv[0] and records the
# normalized `<SYSROOT>/bin/cargo` token.
CARGO_BENCH_COMMON = ["bench", "--locked", "--offline", "--profile", "bench", "--color", "never"]
METADATA_ARGS = ["--", "--print-protocol-metadata"]
GATE_COMMANDS = {
    "kat": ["cargo", "test", "--locked", "--offline", "--color", "never", "--lib",
            "-vv", "--", "--exact", "poseidon2::tests::kat_gate"],
    "tune_corpus": ["cargo", "test", "--locked", "--offline", "--color", "never",
                    "--lib", "-vv", "--", "--exact", "poseidon2::tests::tuning_corpus_gate"],
}
GATE_ORDER = ("kat", "tune_corpus")
GATE_TEST_NAMES = {"kat": "poseidon2::tests::kat_gate",
                   "tune_corpus": "poseidon2::tests::tuning_corpus_gate"}
GATE_OK_PREFIX = "test result: ok. 1 passed; 0 failed; 0 ignored"
ARGV_FORMS_VERSION = 2


def features_for_threads(threads: int) -> list:
    """Always `[]`: limber has no parallel feature; `T > 1` only changes RAYON_NUM_THREADS."""
    if not isinstance(threads, int) or threads < 1:
        raise RunnerError("config", "InvalidKnob", "threads must be >= 1 (got %r)" % (threads,))
    return list(BENCH_FEATURES)


def _feature_tokens(features) -> list:
    if not features:
        return []
    raise RunnerError("config", "InvalidKnob",
                      "the limber bench takes no cargo features (got %r)" % (features,))


def bootstrap_feature_tokens(features) -> list:
    """Feature tokens of the bootstrap `cargo metadata` command (always none for limber)."""
    return _feature_tokens(features)


def bench_prefix(features) -> list:
    """`cargo bench --locked --offline --profile bench --color never --bench poseidon_modp
    -vv` (argv[0] = `cargo`)."""
    return (["cargo"] + CARGO_BENCH_COMMON + _feature_tokens(features)
            + ["--bench", BENCH_NAME, "-vv"])


def evidence_command(features) -> list:
    return (["cargo"] + CARGO_BENCH_COMMON + ["--message-format=json-render-diagnostics"]
            + _feature_tokens(features) + ["--bench", BENCH_NAME, "--no-run", "-vv"])


def metadata_command(features) -> list:
    return bench_prefix(features) + METADATA_ARGS


def preflight_command(features, config_path: str, config_sha256: str, artifact_dir: str) -> list:
    return bench_prefix(features) + ["--", "--run-config", config_path, "--config-sha256",
                                     config_sha256, "--attempt", "preflight", "--artifact-dir",
                                     artifact_dir]


def child_command(features, child_config_path: str, child_sha256: str, artifact_dir: str) -> list:
    return bench_prefix(features) + ["--", "--child-config", child_config_path,
                                     "--child-config-sha256", child_sha256, "--artifact-dir",
                                     artifact_dir]


def gate_command(name: str) -> list:
    return list(GATE_COMMANDS[name])


def evidence_artifact_rules(timing_schema: dict) -> dict:
    """The shared schema's `evidence.compiler_artifact` block with the limber bench target
    (`poseidon_modp` at `<SOURCE>/benches/poseidon_modp.rs`) in place of Zinc's."""
    rules = dict(timing_schema["evidence"]["compiler_artifact"])
    rules["target_name"] = BENCH_NAME
    rules["src_path"] = "<SOURCE>/" + BENCH_SRC_PATH
    return rules


# archive name -> (repository-relative source path, metadata key carrying the compiled ID)
SPEC_FILES = {
    "timing-schema-v2.json": ("specs/poseidon/timing-schema-v2.json", "timing_schema_id"),
    "tuning-protocol-v2.json": ("specs/poseidon/tuning-protocol-v2.json", "tuning_protocol_id"),
    "tune-corpus-v1.json": ("tests/data/tune-corpus-v1.json", "tuning_corpus_id"),
    "security-accounting-v1.json": ("specs/poseidon/security-accounting-v1.json",
                                    "security_accounting_id"),
}
SPEC_ORDER = ("timing-schema-v2.json", "tuning-protocol-v2.json", "tune-corpus-v1.json",
              "security-accounting-v1.json")
# Tracked but not archived: the comparison schema (its digest is checked against the
# compiled `comparison_schema_id` when present).
COMPARISON_SCHEMA_PATH = "specs/poseidon/comparison-schema-v2.json"

METADATA_REQUIRED_KEYS = (
    "transcript_domain_separator", "protocol_wire_id", "prime_sampler_id", "prime_sampler_caps",
    "benchmark_coins_id", "security_accounting_id", "timing_schema_id", "tuning_protocol_id",
    "comparison_schema_id", "kat_fixture_sha256", "tuning_corpus_id", "criterion_version",
    "panic_strategy", "debug_assertions", "overflow_checks", "argv_forms_version", "variants",
    "primary_groups", "diagnostic_groups", "tuning_epoch_id", "tuning_group_set",
    "tuned_defaults", "check_mode", "dimensions_by_k",
)
# Keys that must be absent from limber metadata (present -> recorded execution failure).
METADATA_FORBIDDEN_KEYS = ("checked_flag", "unchecked_flag")
# Values the compiled metadata must carry for `execution_valid`. A different value is a
# recorded execution failure.
METADATA_REQUIRED_VALUES = {
    "panic_strategy": "unwind", "debug_assertions": False, "overflow_checks": False,
    "argv_forms_version": ARGV_FORMS_VERSION, "check_mode": CHECK_MODE_METADATA,
    "variants": list(VARIANTS), "primary_groups": list(PRIMARY_GROUPS),
    "diagnostic_groups": list(DIAGNOSTIC_GROUPS), "prime_sampler_caps": PRIME_SAMPLER_CAPS,
    "kat_fixture_sha256": KAT_FIXTURE_SHA256,
}
# Keys the runner reads but which need not be present (null when absent).
METADATA_OPTIONAL_KEYS = ("common_lift_binding_id", "features")
# Capability keys that identify one compiled protocol schedule (contract section 1).
CAPABILITY_KEYS = ("protocol_wire_id", "prime_sampler_id", "benchmark_coins_id")

# Runner-side list of approved (wire, sampler, coins) tuples. A run whose compiled metadata
# is not listed here has `execution_valid = false` (`capability_tuple_not_approved`).
APPROVED_CAPABILITY_TUPLES = (
    {
        "protocol_wire_id": PROTOCOL_WIRE_ID,
        "prime_sampler_id": PRIME_SAMPLER_ID,
        "benchmark_coins_id": BENCHMARK_COINS_ID,
        "variants": BACKENDS,
    },
)

KNOB_NAMES = ("BACKEND", "HASHES", "INSTANCE", "PSIZE", "SWEEP", "THREADS", "NORMAL_RUN",
              "TUNING_BUNDLE", "TUNING_STORE", "SESSION_SPEC", "OUTPUT_PARENT",
              "POSEIDON_ALLOW_DIRTY")


def repo_root(env=None) -> str:
    """This checkout (the parent of `scripts/`); runners take `--repo` to name another."""
    return os.path.dirname(SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Canonical JSON and digests.
# ---------------------------------------------------------------------------

def _check_canonical_value(obj, path: str) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, int):
        return
    if isinstance(obj, float):
        raise TypeError("float at %s is not allowed in canonical JSON" % path)
    if isinstance(obj, str):
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_canonical_value(item, "%s[%d]" % (path, i))
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError("non-string key at %s" % path)
            _check_canonical_value(value, "%s.%s" % (path, key))
        return
    raise TypeError("unsupported type %s at %s" % (type(obj).__name__, path))


def canonical_json_bytes(obj) -> bytes:
    """Sorted keys, compact separators, ASCII, one trailing LF; floats are rejected."""
    _check_canonical_value(obj, "$")
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return (text + "\n").encode("ascii")


def _reject_float(token: str):
    raise ValueError("float token %r is not allowed" % token)


def load_canonical_json(data: bytes, what: str = "json"):
    """Parse bytes that must be canonical JSON (re-encoding reproduces the exact bytes)."""
    try:
        obj = json.loads(data.decode("utf-8"), parse_float=_reject_float)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunnerError("specs", "CanonicalJsonInvalid", "%s: %s" % (what, exc)) from None
    if canonical_json_bytes(obj) != data:
        raise RunnerError("specs", "CanonicalJsonInvalid", "%s is not canonical JSON" % what)
    return obj


def parse_exact_json(data):
    """JSON with every number token exact: floats become `Fraction`, integers stay `int`."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data, parse_float=Fraction, parse_int=int)


HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_hex64(value) -> bool:
    return isinstance(value, str) and HEX64_RE.match(value) is not None


def format_detached_digest(hexdigest: str) -> bytes:
    if not is_hex64(hexdigest):
        raise RunnerError("seal", "DigestMalformed", "digest %r is not 64 lowercase hex" %
                          (hexdigest,))
    return (hexdigest + "\n").encode("ascii")


def parse_detached_digest(data: bytes, what: str = "digest") -> str:
    if len(data) != 65 or data[64:] != b"\n" or not HEX64_RE.match(data[:64].decode("ascii",
                                                                                     "replace")):
        raise RunnerError("specs", "DigestMalformed",
                          "%s must be exactly 64 lowercase hex characters plus LF" % what)
    return data[:64].decode("ascii")


def write_detached_digest(path: str, hexdigest: str) -> None:
    write_bytes(path, format_detached_digest(hexdigest))


def read_detached_digest(path: str) -> str:
    with open(path, "rb") as f:
        return parse_detached_digest(f.read(), path)


def write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def write_canonical_json(path: str, obj) -> bytes:
    data = canonical_json_bytes(obj)
    write_bytes(path, data)
    return data


# ---------------------------------------------------------------------------
# Strict environment knobs.
# ---------------------------------------------------------------------------

USIZE_RE = re.compile(r"^(0|[1-9][0-9]*)$")


def knob_flag(env, name: str) -> bool:
    value = env.get(name)
    if value is None or value == "0":
        return False
    if value == "1":
        return True
    raise RunnerError("env", "InvalidKnob", "%s=%r must be 0 or 1" % (name, value))


def knob_usize(env, name: str, default=None):
    value = env.get(name)
    if value is None:
        return default
    if not USIZE_RE.match(value):
        raise RunnerError("env", "InvalidKnob",
                          "%s=%r must be a canonical unsigned decimal integer" % (name, value))
    return int(value)


def knob_enum(env, name: str, allowed, default=None):
    value = env.get(name)
    if value is None:
        return default
    allowed_str = [str(a) for a in allowed]
    if value not in allowed_str:
        raise RunnerError("env", "InvalidKnob", "%s=%r must be one of %s" %
                          (name, value, ", ".join(allowed_str)))
    return allowed[allowed_str.index(value)]


def knob_string(env, name: str, required: bool = False):
    value = env.get(name)
    if value is None:
        if required:
            raise RunnerError("env", "InvalidKnob", "%s is required" % name)
        return None
    if value == "":
        raise RunnerError("env", "InvalidKnob", "%s must not be empty" % name)
    return value


def reject_knobs(env, names, mode: str) -> None:
    for name in names:
        if name in env:
            raise RunnerError("env", "RejectedKnob",
                              "%s is rejected in %s mode (present with value %r)" %
                              (name, mode, env[name]))


def select_mode(env) -> str:
    sweep = knob_flag(env, "SWEEP")
    psize = knob_flag(env, "PSIZE")
    if sweep and psize:
        raise RunnerError("env", "ModeConflict", "SWEEP=1 and PSIZE=1 are mutually exclusive")
    if sweep:
        return "sweep"
    if psize:
        return "psize"
    return "normal"


# ---------------------------------------------------------------------------
# Exact arithmetic helpers.
# ---------------------------------------------------------------------------

def fraction_to_decimal(value) -> str:
    """Exact terminating decimal rendering of a rational (raises if non-terminating)."""
    fr = Fraction(value)
    num, den = fr.numerator, fr.denominator
    sign = "-" if num < 0 else ""
    num = abs(num)
    d, twos, fives = den, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        raise ValueError("%s is not a terminating decimal" % fr)
    k = max(twos, fives)
    scaled = num * (10 ** k) // den
    digits = str(scaled).rjust(k + 1, "0")
    if k == 0:
        return sign + digits
    int_part, frac_part = digits[:-k], digits[-k:].rstrip("0")
    return sign + int_part + ("." + frac_part if frac_part else "")


def fraction_from_decimal(text: str) -> Fraction:
    if not isinstance(text, str) or not re.match(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$", text):
        raise ValueError("%r is not a canonical decimal string" % (text,))
    return Fraction(text)


def ratio_pair(value) -> list:
    """Reduced `[numerator, denominator]` decimal-integer strings, positive denominator."""
    fr = Fraction(value)
    return [str(fr.numerator), str(fr.denominator)]


def ratio_from_pair(pair) -> Fraction:
    if (not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(p, str) for p in pair)
            or not re.match(r"^-?(0|[1-9][0-9]*)$", pair[0])
            or not re.match(r"^[1-9][0-9]*$", pair[1])):
        raise ValueError("%r is not a reduced ratio pair" % (pair,))
    fr = Fraction(int(pair[0]), int(pair[1]))
    if [str(fr.numerator), str(fr.denominator)] != pair:
        raise ValueError("%r is not reduced" % (pair,))
    return fr


def exact_median(values) -> Fraction:
    items = sorted(Fraction(v) for v in values)
    n = len(items)
    if n == 0:
        raise ValueError("median of an empty list")
    if n % 2 == 1:
        return items[n // 2]
    return (items[n // 2 - 1] + items[n // 2]) / 2


def positive_fraction(value, what: str) -> Fraction:
    fr = Fraction(value)
    if fr <= 0:
        raise ValueError("%s must be positive, got %s" % (what, fr))
    return fr


# ---------------------------------------------------------------------------
# Time.
# ---------------------------------------------------------------------------

def utc_rfc3339_ns(ns=None) -> str:
    """`YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ` from `time.time_ns()`."""
    if ns is None:
        ns = time.time_ns()
    seconds, nanos = divmod(int(ns), 10 ** 9)
    dt = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + ".%09dZ" % nanos


RFC3339_NS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{9})Z$")


def compact_timestamp(rfc3339: str) -> str:
    """`YYYYMMDDTHHMMSS.nnnnnnnnnZ` for run/session basenames."""
    m = RFC3339_NS_RE.match(rfc3339)
    if not m:
        raise ValueError("%r is not an RFC3339 nanosecond UTC timestamp" % (rfc3339,))
    y, mo, d, h, mi, s, ns = m.groups()
    return "%s%s%sT%s%s%s.%sZ" % (y, mo, d, h, mi, s, ns)


# ---------------------------------------------------------------------------
# Subprocesses, git, toolchain.
# ---------------------------------------------------------------------------

def run_capture(args, cwd, env=None, stage="internal", code="InternalError"):
    try:
        return subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise RunnerError(stage, code, "cannot execute %s: %s" % (args[0], exc)) from None


def git(repo: str, *args) -> str:
    proc = run_capture(["git", *args], repo, stage="source", code="GitCommandFailed")
    if proc.returncode != 0:
        raise RunnerError("source", "GitCommandFailed", "git %s failed: %s" %
                          (" ".join(args), proc.stderr.decode("utf-8", "replace").strip()))
    return proc.stdout.decode("utf-8", "replace")


def git_head(repo: str) -> str:
    sha = git(repo, "rev-parse", "HEAD").strip()
    if not re.match(r"^[0-9a-f]{40}$", sha):
        raise RunnerError("source", "GitCommandFailed", "unexpected HEAD %r" % sha)
    return sha


def git_status_porcelain(repo: str) -> list:
    """Modified/added/deleted and untracked paths (untracked included on purpose)."""
    out = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return [line for line in out.split("\n") if line]


def git_is_dirty(repo: str) -> bool:
    return len(git_status_porcelain(repo)) > 0


def git_is_tracked(repo: str, path: str) -> bool:
    proc = run_capture(["git", "ls-files", "--error-unmatch", path], repo, stage="source",
                       code="GitCommandFailed")
    return proc.returncode == 0


def git_diff_stat(repo: str, a: str, b: str) -> str:
    return git(repo, "diff", "--stat", a, b)


def git_diff_names(repo: str, a: str, b: str) -> list:
    out = git(repo, "diff", "--name-only", a, b)
    return sorted(line for line in out.split("\n") if line)


# ---------------------------------------------------------------------------
# Cargo.lock / Cargo.toml dependency checks.
# ---------------------------------------------------------------------------

def parse_cargo_lock(text: str) -> dict:
    """name -> list of {version, source, checksum} from `[[package]]` blocks."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RunnerError("source", "DependencyMismatch", "Cargo.lock: %s" % exc) from None
    packages = {}
    for pkg in doc.get("package", []):
        packages.setdefault(pkg["name"], []).append(
            {"version": pkg.get("version"), "source": pkg.get("source"),
             "checksum": pkg.get("checksum")})
    return packages


def criterion_manifest_spec(cargo_toml_text: str) -> dict:
    """limber declares Criterion as a `[dev-dependencies]` table of the single package."""
    try:
        doc = tomllib.loads(cargo_toml_text)
    except tomllib.TOMLDecodeError as exc:
        raise RunnerError("source", "DependencyMismatch", "Cargo.toml: %s" % exc) from None
    dep = doc.get("dev-dependencies", {}).get("criterion")
    if dep is None:
        dep = doc.get("workspace", {}).get("dependencies", {}).get("criterion")
    if not isinstance(dep, dict):
        raise RunnerError("source", "DependencyMismatch",
                          "Cargo.toml has no table-form criterion dev-dependency")
    return {"version": dep.get("version"),
            "default_features": dep.get("default-features", True),
            "features": list(dep.get("features", []))}


def dependency_versions(repo: str) -> dict:
    """Resolved Criterion version plus the exact Criterion dependency check (limber has no
    zstd dependency; the `zstd` record is kept for schema parity and is null)."""
    with open(os.path.join(repo, "Cargo.lock"), "r", encoding="utf-8") as f:
        lock_text = f.read()
    with open(os.path.join(repo, "Cargo.toml"), "r", encoding="utf-8") as f:
        toml_text = f.read()
    packages = parse_cargo_lock(lock_text)
    crit = packages.get("criterion", [])
    if len(crit) != 1 or crit[0]["version"] != CRITERION_SETTINGS["version"]:
        raise RunnerError("source", "DependencyMismatch",
                          "Cargo.lock must resolve exactly one criterion %s, found %s" %
                          (CRITERION_SETTINGS["version"], [c["version"] for c in crit]))
    spec = criterion_manifest_spec(toml_text)
    if (spec["version"] != "=" + CRITERION_SETTINGS["version"]
            or spec["default_features"] is not False
            or spec["features"] != CRITERION_SETTINGS["features"]):
        raise RunnerError("source", "DependencyMismatch",
                          "Cargo.toml criterion dependency must be version \"=0.7.0\", "
                          "default-features = false, features = [\"cargo_bench_support\"]; "
                          "found %r" % (spec,))
    zstd = packages.get("zstd", [])
    zstd_sys = packages.get("zstd-sys", [])
    return {
        "criterion": {"version": crit[0]["version"], "checksum": crit[0]["checksum"],
                      "default_features": False, "features": spec["features"]},
        "zstd": {"crate": zstd[0]["version"] if len(zstd) == 1 else None,
                 "sys": zstd_sys[0]["version"] if len(zstd_sys) == 1 else None},
    }


# ---------------------------------------------------------------------------
# Machine fingerprint.
# ---------------------------------------------------------------------------

def _read_text(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def machine_fingerprint() -> dict:
    system = platform.system()
    fp = {"os": system, "arch": platform.machine(), "platform": platform.platform(),
          "cpu_model": None, "cpu_microcode": None, "physical_cores": None,
          "logical_cores": None, "memory_bytes": None, "kernel": None, "hostname_sha256":
          sha256_hex(platform.node().encode("utf-8"))}
    if system == "Darwin":
        keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
                "kern.osversion"]
        proc = run_capture(["sysctl", "-n", *keys], "/", stage="toolchain",
                           code="ToolchainUnavailable")
        lines = proc.stdout.decode("utf-8", "replace").split("\n")
        if proc.returncode == 0 and len(lines) >= 5:
            fp["cpu_model"] = lines[0].strip()
            fp["physical_cores"] = int(lines[1].strip())
            fp["logical_cores"] = int(lines[2].strip())
            fp["memory_bytes"] = int(lines[3].strip())
            fp["kernel"] = lines[4].strip()
        fp["os_release"] = platform.mac_ver()[0]
    elif system == "Linux":
        cpuinfo = _read_text("/proc/cpuinfo") or ""
        for line in cpuinfo.split("\n"):
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "model name" and fp["cpu_model"] is None:
                fp["cpu_model"] = value
            elif key == "microcode" and fp["cpu_microcode"] is None:
                fp["cpu_microcode"] = value
        proc = run_capture(["nproc"], "/", stage="toolchain", code="ToolchainUnavailable")
        if proc.returncode == 0:
            fp["logical_cores"] = int(proc.stdout.decode().strip())
        cores = set()
        for line in cpuinfo.split("\n"):
            if line.startswith("core id"):
                cores.add(line.partition(":")[2].strip())
        fp["physical_cores"] = len(cores) or None
        meminfo = _read_text("/proc/meminfo") or ""
        for line in meminfo.split("\n"):
            if line.startswith("MemTotal:"):
                fp["memory_bytes"] = int(line.split()[1]) * 1024
        proc = run_capture(["uname", "-r"], "/", stage="toolchain", code="ToolchainUnavailable")
        if proc.returncode == 0:
            fp["kernel"] = proc.stdout.decode().strip()
    return fp


# ---------------------------------------------------------------------------
# Specification files and compiled protocol metadata.
# ---------------------------------------------------------------------------

def spec_source_path(repo: str, archive_name: str) -> str:
    rel, _ = SPEC_FILES[archive_name]
    return os.path.join(repo, rel)


def load_spec(repo: str, archive_name: str):
    """Exact bytes and SHA-256 of a tracked specification file (missing -> SpecMissing)."""
    path = spec_source_path(repo, archive_name)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise RunnerError("specs", "SpecMissing", "%s is absent (expected at %s)" %
                          (archive_name, path)) from None
    return data, sha256_hex(data)


def benchmark_coins_framing(tuning_protocol) -> str:
    """The framing string of the tuning protocol's limber benchmark-coins policy."""
    try:
        framing = tuning_protocol["systems"]["limber"]["benchmark_coins"]["framing"]
    except (KeyError, TypeError):
        return BENCHMARK_COINS_FRAMING
    return framing if isinstance(framing, str) else BENCHMARK_COINS_FRAMING


def benchmark_coins_policy(tuning_protocol) -> dict:
    """`ids.benchmark_coins_policy` of a limber run config (contract section 4)."""
    return {"id": BENCHMARK_COINS_ID, "framing": benchmark_coins_framing(tuning_protocol)}


class ProtocolMetadata:
    """The one JSON object printed by `--print-protocol-metadata` (kept exactly)."""

    def __init__(self, raw: dict, command: list):
        self.raw = raw
        self.command = command

    def get(self, key: str):
        return self.raw.get(key)

    def compiled_ids(self) -> dict:
        """The exact object the bench printed (the bench later requires equality)."""
        return dict(self.raw)

    def capability_tuple(self) -> dict:
        return {k: self.raw.get(k) for k in CAPABILITY_KEYS}

    def tuned_default(self, backend: str) -> dict:
        """`{k, tuning_id}` of the compiled tuned default for `backend` (both null when the
        generated module carries no entry). Accepts the object form `{backend, k,
        tuning_id}` and the tuple form `[backend, k, tuning_id]` of `tuned_defaults`."""
        for entry in self.raw.get("tuned_defaults") or []:
            if isinstance(entry, dict):
                if entry.get("backend") == backend:
                    return {"k": entry.get("k"), "tuning_id": entry.get("tuning_id")}
            elif isinstance(entry, list) and len(entry) == 3 and entry[0] == backend:
                return {"k": entry[1], "tuning_id": entry[2]}
        return {"k": None, "tuning_id": None}


def parse_protocol_metadata(stdout: bytes, command: list) -> ProtocolMetadata:
    """`stdout` must be exactly one canonical JSON object line (plus LF)."""
    if not stdout.endswith(b"\n") or stdout.count(b"\n") != 1:
        raise RunnerError("metadata", "MetadataMalformed",
                          "metadata stdout must be exactly one JSON line plus LF")
    raw = load_canonical_json(stdout, "protocol metadata")
    if not isinstance(raw, dict):
        raise RunnerError("metadata", "MetadataMalformed", "metadata is not a JSON object")
    missing = [k for k in METADATA_REQUIRED_KEYS if k not in raw]
    if missing:
        raise RunnerError("metadata", "MetadataMalformed", "metadata lacks keys: %s" %
                          ", ".join(missing))
    return ProtocolMetadata(raw, list(command))


def _dimension_entry(value, what: str) -> dict:
    if not isinstance(value, dict) or sorted(value) != ["log_cons", "log_vars"] or not all(
            isinstance(value[k], int) and not isinstance(value[k], bool) and value[k] >= 0
            for k in ("log_cons", "log_vars")):
        raise RunnerError("metadata", "MetadataMalformed",
                          "%s must be {log_cons: int, log_vars: int}" % what)
    return {"log_cons": value["log_cons"], "log_vars": value["log_vars"]}


def dimensions_by_candidate(metadata: ProtocolMetadata, candidates=None) -> dict:
    """`{str(k): {log_cons, log_vars}}` for every candidate k, from the compiled
    `dimensions_by_k` (H = 10). Every candidate must be present."""
    raw = metadata.get("dimensions_by_k")
    if not isinstance(raw, dict):
        raise RunnerError("metadata", "MetadataMalformed", "dimensions_by_k is not an object")
    out = {}
    for k in (CANDIDATES if candidates is None else candidates):
        key = str(k)
        if key not in raw:
            raise RunnerError("metadata", "MetadataMalformed",
                              "dimensions_by_k lacks candidate k = %s" % key)
        out[key] = _dimension_entry(raw[key], "dimensions_by_k[%s]" % key)
    return out


# ---------------------------------------------------------------------------
# Criterion 0.7.0 output-tree grammar.
# ---------------------------------------------------------------------------

CRITERION_LEAVES = ("benchmark.json", "estimates.json", "sample.json", "tukey.json")
CRITERION_BASELINES = ("base", "new")
CRITERION_DIR_TRUNCATION = 64
CRITERION_TITLE_LIMIT = 100
_UNSAFE_ID_CHARS = '?"/\\*<>:|^'
# Contract section 5.
FUNCTION_ID_RE = re.compile(
    r"^(?P<backend>hyrax|brakedown)/mixed3/Hpf(?P<hashes>[0-9]+)-total(?P<total>[0-9]+)/"
    r"c2\^(?P<log_cons>[0-9]+)v2\^(?P<log_vars>[0-9]+)/k(?P<k>[0-9]+)/inst(?P<instance>[0-9]+)/"
    r"thr(?P<threads>[0-9]+)/(?:primary/blk(?P<block>[0-9]+)|diagnostic/blkdiag)$"
)
ESTIMATE_STATISTICS = ("mean", "median", "median_abs_dev", "slope", "std_dev")
# TUNE-1 reads `median.confidence_interval` and requires exactly this level.
CRITERION_CONFIDENCE_LEVEL = Fraction(95, 100)


def function_id(backend: str, hashes: int, dimensions: dict, k: int, instance: int,
                threads: int, kind: str, block) -> str:
    """Render one limber Criterion function id (the inverse of `parse_function_id`)."""
    tail = "primary/blk%d" % block if kind == "primary" else "diagnostic/blkdiag"
    return "%s/mixed3/Hpf%d-total%d/c2^%dv2^%d/k%d/inst%d/thr%d/%s" % (
        backend, hashes, 3 * hashes, dimensions["log_cons"], dimensions["log_vars"], k,
        instance, threads, tail)


def criterion_filename_safe(component: str) -> str:
    out = "".join("_" if ch in _UNSAFE_ID_CHARS else ch for ch in component)
    encoded = out.encode("utf-8")
    if len(encoded) > CRITERION_DIR_TRUNCATION:
        boundary = CRITERION_DIR_TRUNCATION
        while boundary > 0 and (encoded[boundary] & 0xC0) == 0x80:
            boundary -= 1
        out = encoded[:boundary].decode("utf-8")
    return out


def criterion_title_matches(expected: str, actual: str) -> bool:
    """Criterion keeps titles unique per process: a title that collides with an earlier
    registration gets the suffix `" #N"` (N >= 2, `report.rs::ensure_title_unique`)."""
    if actual == expected:
        return True
    prefix = expected + " #"
    if not actual.startswith(prefix):
        return False
    suffix = actual[len(prefix):]
    return suffix.isdigit() and suffix == str(int(suffix)) and int(suffix) >= 2


def criterion_names(group_id: str, function_id: str, value_str: str) -> dict:
    """`full_id`, `directory_name` and `title` as Criterion 0.7.0 derives them."""
    full_id = "%s/%s/%s" % (group_id, function_id, value_str)
    directory_name = "%s/%s/%s" % (criterion_filename_safe(group_id),
                                   criterion_filename_safe(function_id),
                                   criterion_filename_safe(value_str))
    if len(full_id) > CRITERION_TITLE_LIMIT:
        title = full_id[:CRITERION_TITLE_LIMIT] + "..."
    else:
        title = full_id
    return {"full_id": full_id, "directory_name": directory_name, "title": title}


def parse_function_id(function_id: str) -> dict:
    m = FUNCTION_ID_RE.match(function_id)
    if not m:
        raise RunnerError("child", "CriterionTreeInvalid",
                          "function_id %r does not match the limber grammar" % function_id)
    d = m.groupdict()
    out = {
        "backend": d["backend"], "hashes": int(d["hashes"]), "total": int(d["total"]),
        "log_cons": int(d["log_cons"]), "log_vars": int(d["log_vars"]), "k": int(d["k"]),
        "instance": int(d["instance"]), "threads": int(d["threads"]),
        "kind": "primary" if d["block"] is not None else "diagnostic",
        "block": int(d["block"]) if d["block"] is not None else "diag",
    }
    if out["total"] != 3 * out["hashes"]:
        raise RunnerError("child", "CriterionTreeInvalid",
                          "function_id %r: total != 3*H" % function_id)
    if not K_RANGE[0] <= out["k"] <= K_RANGE[1]:
        raise RunnerError("child", "CriterionTreeInvalid",
                          "function_id %r: k outside %s" % (function_id, list(K_RANGE)))
    return out


def parse_estimates(data: bytes) -> dict:
    """Exact-decimal `estimates.json`: every statistic's point/lower/upper as `Fraction`."""
    obj = parse_exact_json(data)
    if not isinstance(obj, dict):
        raise RunnerError("child", "CriterionTreeInvalid", "estimates.json is not an object")
    out = {}
    for stat in ESTIMATE_STATISTICS:
        if stat not in obj:
            raise RunnerError("child", "CriterionTreeInvalid",
                              "estimates.json lacks %r" % stat)
        entry = obj[stat]
        if stat == "slope" and entry is None:
            # Criterion writes `"slope": null` whenever it measured with flat
            # sampling, which is the normal outcome for the `iter_batched`
            # comparable groups; TUNE-1 and the publisher read `median` only.
            out[stat] = None
            continue
        if not isinstance(entry, dict):
            raise RunnerError("child", "CriterionTreeInvalid",
                              "estimates.json %r is not an object" % stat)
        ci = entry.get("confidence_interval")
        try:
            out[stat] = {
                "point_estimate": Fraction(entry["point_estimate"]),
                "standard_error": Fraction(entry["standard_error"]),
                "confidence_level": Fraction(ci["confidence_level"]),
                "lower_bound": Fraction(ci["lower_bound"]),
                "upper_bound": Fraction(ci["upper_bound"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError("child", "CriterionTreeInvalid",
                              "estimates.json %s: %s" % (stat, exc)) from None
    return out


def list_files(root: str) -> list:
    """Sorted relative paths of every regular file below `root`; symlinks are rejected."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise RunnerError("seal", "SymlinkRejected", "symlink directory %s" %
                                  os.path.join(dirpath, name))
        for name in filenames:
            full = os.path.join(dirpath, name)
            st = os.lstat(full)
            if _stat.S_ISLNK(st.st_mode):
                raise RunnerError("seal", "SymlinkRejected", "symlink %s" % full)
            if not _stat.S_ISREG(st.st_mode):
                raise RunnerError("seal", "UnsafePath", "not a regular file: %s" % full)
            out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out, key=lambda p: p.encode("utf-8"))


def validate_criterion_tree(criterion_root: str, expectation: dict) -> list:
    """Validate a child's Criterion output directory against the observed 0.7.0 grammar.

    `expectation` keys: `groups` (exact set of registered group ids), `backend`, `hashes`,
    `instance`, `k`, `threads`, `kind`, `block` (int or "diag") and optionally `dimensions`
    (`{log_cons, log_vars}` the IDs must carry). Returns one record per registered ID with
    its exact `estimates` (from `new/`).
    """
    err = lambda msg: RunnerError("child", "CriterionTreeInvalid", msg)  # noqa: E731
    if not os.path.isdir(criterion_root):
        raise err("criterion output directory %s is missing" % criterion_root)
    files = list_files(criterion_root)
    if not files:
        raise err("criterion output directory is empty")
    by_dir = {}
    for rel in files:
        parts = rel.split("/")
        if len(parts) < 4 or parts[-2] not in CRITERION_BASELINES or parts[-1] not in \
                CRITERION_LEAVES:
            raise err("unlisted Criterion leaf %r" % rel)
        by_dir.setdefault("/".join(parts[:-2]), set()).add("/".join(parts[-2:]))
    expected_leaves = {"%s/%s" % (b, leaf) for b in CRITERION_BASELINES for leaf in
                       CRITERION_LEAVES}
    records = []
    seen_groups = set()
    seen_ids = set()
    seen_titles = set()
    for id_dir in sorted(by_dir):
        if by_dir[id_dir] != expected_leaves:
            raise err("directory %r has leaf set %s, expected the eight canonical files" %
                      (id_dir, sorted(by_dir[id_dir])))
        base_dir = os.path.join(criterion_root, id_dir, "base")
        new_dir = os.path.join(criterion_root, id_dir, "new")
        for leaf in CRITERION_LEAVES:
            with open(os.path.join(base_dir, leaf), "rb") as f:
                base_bytes = f.read()
            with open(os.path.join(new_dir, leaf), "rb") as f:
                new_bytes = f.read()
            if base_bytes != new_bytes:
                raise err("%s/base/%s differs from new/%s" % (id_dir, leaf, leaf))
        with open(os.path.join(new_dir, "benchmark.json"), "rb") as f:
            bench = json.loads(f.read().decode("utf-8"))
        for key in ("group_id", "function_id", "value_str", "full_id", "directory_name",
                    "title"):
            if key not in bench or (key != "throughput" and not isinstance(bench[key], str)):
                raise err("%s/new/benchmark.json lacks string %r" % (id_dir, key))
        if bench.get("throughput") is not None:
            raise err("%s: throughput must be null" % id_dir)
        names = criterion_names(bench["group_id"], bench["function_id"], bench["value_str"])
        for key in ("full_id", "directory_name"):
            if bench[key] != names[key]:
                raise err("%s: %s is %r, expected %r" % (id_dir, key, bench[key], names[key]))
        if not criterion_title_matches(names["title"], bench["title"]):
            raise err("%s: title is %r, expected %r (optionally followed by Criterion's "
                      "collision suffix \" #N\")" % (id_dir, bench["title"], names["title"]))
        if bench["title"] in seen_titles:
            raise err("duplicate title %r" % bench["title"])
        seen_titles.add(bench["title"])
        expected_dir = "/".join(c for c in names["directory_name"].split("/") if c)
        if id_dir != expected_dir:
            raise err("directory %r does not match directory_name %r" %
                      (id_dir, bench["directory_name"]))
        parsed = parse_function_id(bench["function_id"])
        group = bench["group_id"]
        if group not in expectation["groups"]:
            raise err("group %r is not registered for this child" % group)
        if bench["value_str"] != "":
            raise err("group %r must have an empty value_str" % group)
        for key in ("backend", "hashes", "instance", "k", "threads", "kind", "block"):
            if parsed[key] != expectation[key]:
                raise err("%s: %s is %r, expected %r" % (bench["function_id"], key,
                                                         parsed[key], expectation[key]))
        dims = expectation.get("dimensions")
        if dims is not None:
            for key in ("log_cons", "log_vars"):
                if parsed[key] != dims[key]:
                    raise err("%s: %s is %d, expected %d" % (bench["function_id"], key,
                                                             parsed[key], dims[key]))
        if bench["full_id"] in seen_ids:
            raise err("duplicate full_id %r" % bench["full_id"])
        seen_ids.add(bench["full_id"])
        seen_groups.add(group)
        with open(os.path.join(new_dir, "estimates.json"), "rb") as f:
            estimates = parse_estimates(f.read())
        for stat_name, est in estimates.items():
            if est is None:
                continue  # `slope` is null under flat sampling
            if est["confidence_level"] != CRITERION_CONFIDENCE_LEVEL:
                raise err("%s: %s confidence_level is %s, expected exactly 0.95" %
                          (bench["function_id"], stat_name, est["confidence_level"]))
        records.append({"group_id": group, "function_id": bench["function_id"],
                        "value_str": bench["value_str"], "full_id": bench["full_id"],
                        "directory": id_dir, "parsed": parsed, "estimates": estimates,
                        "estimates_path": "%s/new/estimates.json" % id_dir})
    missing = set(expectation["groups"]) - seen_groups
    if missing:
        raise err("registered groups without output: %s" % sorted(missing))
    for group in seen_groups:
        n = sum(1 for r in records if r["group_id"] == group)
        if n != 1:
            raise err("group %r has %d IDs, expected exactly one" % (group, n))
    return records


def hashed_file_list(root: str, rel_prefix: str = "") -> list:
    """Sorted `[path, size, sha256]` triples for every regular file below `root`."""
    out = []
    for rel in list_files(root):
        full = os.path.join(root, rel)
        path = rel_prefix + rel if rel_prefix else rel
        out.append([path, os.path.getsize(full), sha256_file(full)])
    return out


def eprint(*args) -> None:
    print(*args, file=sys.stderr)
