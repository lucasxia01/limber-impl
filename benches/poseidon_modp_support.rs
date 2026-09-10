//! Shared dev-support for the Poseidon2 cross-system benchmark
//! (`benches/poseidon_modp.rs`) and its contract test
//! (`tests/poseidon_bench_contract.rs`), included via `#[path]`; it is not
//! part of the library.
//!
//! It holds the environment-free half of the bench <-> runner contract
//! (`scripts/NOTES-for-rust.md`; Zinc plan v10 §9; limber contract §§1, 4,
//! 5): the argv router ([`route`]), the run/child config parsers, canonical
//! JSON, the compiled metadata object ([`protocol_metadata`]), the
//! path/digest validators of the two file-driven forms, the Criterion ID
//! grammar, the deterministic benchmark coins and the audit-record JSON.
//! The structure mirrors Zinc's `protocol/benches/poseidon_support.rs`.

#![allow(dead_code)]

use limber::{
  poseidon_bench::{
    self as pb, BACKEND_NAMES, BENCHMARK_COINS_FRAMING, BENCHMARK_COINS_ID, BenchBackend,
    CANONICAL_HASHES_PER_FIELD, DIAGNOSTIC_GROUPS, Dimensions, HIDDEN_KNOBS, K_ORDER,
    LIMBER_PRIME_SAMPLER_ID, LIMBER_PROTOCOL_WIRE_ID, LIMBER_TRANSCRIPT_ID, PRIMARY_GROUPS,
  },
  poseidon_tuned_defaults as tuned_defaults,
  poseidon2::ids,
  prime_sampler::{
    BASE_DRAW_MAX, MAX_PRIME_INVOCATIONS, MR_ROUNDS, PrimeSamplerAudit, PrimeSamplerOutcome, T_MAX,
  },
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::{
  collections::BTreeMap,
  hint::black_box,
  path::{Component, Path, PathBuf},
  sync::OnceLock,
};

//
// Compiled constants
//

/// Version of the argv contract implemented by `main`.
pub const ARGV_FORMS_VERSION: u64 = 2;
pub const RUN_CONFIG_SCHEMA: &str = "limber/poseidon-run-config/v2";
pub const CHILD_CONFIG_SCHEMA: &str = "limber/poseidon-child-config/v2";
pub const PREFLIGHT_SCHEMA: &str = "limber/poseidon-preflight/v2";
pub const PROOF_SIZE_SCHEMA: &str = "limber/poseidon-proof-size/v2";
pub const SWEEP_METADATA_SCHEMA: &str = "limber/poseidon-sweep-metadata/v2";
pub const ATTEMPT_RESULT_SCHEMA: &str = "limber/poseidon-attempt-result/v2";
/// The bench binary's name (stderr prefixes).
pub const BENCH_NAME: &str = "poseidon_modp";
/// The pinned `RUSTFLAGS` of a canonical run.
pub const REQUIRED_RUSTFLAGS: &str = "-C target-cpu=native";
/// The metadata / run-config check mode: limber has no checked/unchecked
/// configuration.
pub const CHECK_MODE: &str = "not_applicable";
/// `check_modes.timed` of every run config.
pub const TIMED_CHECK_MODE: &str = "release";
/// The proof-size metric kind (the record mixes exact serialized
/// components with an analytical remainder).
pub const PROOF_SIZE_METRIC_KIND: &str = "mixed_payload_estimate";
/// The pinned Criterion settings of every child.
pub const CRITERION_VERSION: &str = "0.7.0";
pub const CRITERION_SAMPLE_SIZE: u64 = 10;
pub const CRITERION_WARM_UP_S: u64 = 1;
pub const CRITERION_MEASUREMENT_S: u64 = 20;
/// The systems of a normal-mode block order.
pub const ORDER_SYSTEMS: [&str; 3] = ["zinc", "hyrax", "brakedown"];
/// The three literal argv forms after the trailing `--bench` is stripped.
pub const ARGV_FORMS: [&str; 3] = [
  "--print-protocol-metadata",
  "--run-config <abs-file> --config-sha256 <64-lower-hex> --attempt preflight --artifact-dir <abs-empty-dir>",
  "--child-config <abs-file> --child-config-sha256 <64-lower-hex> --artifact-dir <abs-empty-dir>",
];
/// The exact eight keys of a child config.
pub const CHILD_CONFIG_KEYS: [&str; 8] = [
  "coordinate",
  "order",
  "ordinal",
  "parent_config",
  "parent_config_sha256",
  "schema",
  "session_id",
  "system_role",
];
/// The exact (sorted) key set of the compiled metadata object.
pub const METADATA_KEYS: [&str; 25] = [
  "argv_forms_version",
  "benchmark_coins_id",
  "check_mode",
  "common_lift_binding_id",
  "comparison_schema_id",
  "criterion_version",
  "debug_assertions",
  "diagnostic_groups",
  "dimensions_by_k",
  "kat_fixture_sha256",
  "overflow_checks",
  "panic_strategy",
  "primary_groups",
  "prime_sampler_caps",
  "prime_sampler_id",
  "protocol_wire_id",
  "security_accounting_id",
  "timing_schema_id",
  "transcript_domain_separator",
  "tuned_defaults",
  "tuning_corpus_id",
  "tuning_epoch_id",
  "tuning_group_set",
  "tuning_protocol_id",
  "variants",
];
/// The Criterion function-id grammar (contract §5).
pub const FUNCTION_ID_REGEX: &str = r"^(?P<backend>hyrax|brakedown)/mixed3/Hpf(?P<hashes>[0-9]+)-total(?P<total>[0-9]+)/c2\^(?P<log_cons>[0-9]+)v2\^(?P<log_vars>[0-9]+)/k(?P<k>[0-9]+)/inst(?P<instance>[0-9]+)/thr(?P<threads>[0-9]+)/(?:primary/blk(?P<block>[0-9]+)|diagnostic/blkdiag)$";

//
// Hex, digests, canonical JSON
//

pub fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
  hex(&Sha256::digest(bytes))
}

/// Non-empty, even-length, lowercase hex.
pub fn is_lower_hex(s: &str) -> bool {
  !s.is_empty()
    && s.len().is_multiple_of(2)
    && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// 64 lowercase hex digits.
pub fn is_sha256_hex(s: &str) -> bool {
  s.len() == 64 && is_lower_hex(s)
}

fn write_canonical_string(s: &str, out: &mut String) {
  out.push('"');
  for ch in s.chars() {
    match ch {
      '"' => out.push_str("\\\""),
      '\\' => out.push_str("\\\\"),
      '\n' => out.push_str("\\n"),
      '\r' => out.push_str("\\r"),
      '\t' => out.push_str("\\t"),
      '\u{8}' => out.push_str("\\b"),
      '\u{c}' => out.push_str("\\f"),
      c if c.is_ascii() && !c.is_ascii_control() => out.push(c),
      c => {
        let mut units = [0u16; 2];
        for unit in c.encode_utf16(&mut units) {
          out.push_str(&format!("\\u{unit:04x}"));
        }
      }
    }
  }
  out.push('"');
}

fn write_canonical(v: &serde_json::Value, out: &mut String) {
  match v {
    serde_json::Value::Null => out.push_str("null"),
    serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    serde_json::Value::Number(n) => {
      assert!(!n.is_f64(), "canonical JSON forbids floats, got {n}");
      out.push_str(&n.to_string());
    }
    serde_json::Value::String(s) => write_canonical_string(s, out),
    serde_json::Value::Array(items) => {
      out.push('[');
      for (i, item) in items.iter().enumerate() {
        if i > 0 {
          out.push(',');
        }
        write_canonical(item, out);
      }
      out.push(']');
    }
    serde_json::Value::Object(map) => {
      let sorted: BTreeMap<&String, &serde_json::Value> = map.iter().collect();
      out.push('{');
      for (i, (k, item)) in sorted.iter().enumerate() {
        if i > 0 {
          out.push(',');
        }
        write_canonical_string(k, out);
        out.push(':');
        write_canonical(item, out);
      }
      out.push('}');
    }
  }
}

/// Canonical JSON: sorted keys (explicitly, independent of serde_json's
/// map type), compact separators, ASCII (non-ASCII escaped as `\uXXXX`
/// like Python's `ensure_ascii`), no floats, one trailing LF.
pub fn canonical_json(value: &serde_json::Value) -> String {
  let mut text = String::new();
  write_canonical(value, &mut text);
  text.push('\n');
  text
}

//
// Compiled metadata
//

/// `"unwind"` or `"abort"` from `cfg!(panic = ...)`.
pub fn panic_strategy() -> &'static str {
  if cfg!(panic = "unwind") {
    "unwind"
  } else {
    "abort"
  }
}

/// Whether a dynamic `u64` addition overflow panics in this build: a
/// black-boxed, caught probe evaluated once.
pub fn overflow_checks_probe() -> bool {
  static PROBE: OnceLock<bool> = OnceLock::new();
  *PROBE.get_or_init(|| {
    if !cfg!(panic = "unwind") {
      // A panicking probe would abort the process instead of being
      // caught; report the compile-time proxy instead (the bench
      // profile is `unwind`, so this arm is never the canonical one).
      return cfg!(debug_assertions);
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(|| {
      let lhs = black_box(u64::MAX);
      let rhs = black_box(1_u64);
      black_box(lhs + rhs)
    });
    std::panic::set_hook(previous);
    outcome.is_err()
  })
}

/// The compiled `H = 10` dimensions per candidate `k`.
pub fn compiled_dimensions() -> BTreeMap<usize, Dimensions> {
  pb::dimensions_by_k(CANONICAL_HASHES_PER_FIELD).expect("H = 10 dimensions")
}

/// `{log_cons, log_vars}` of one dimension entry.
pub fn dimensions_json(d: &Dimensions) -> serde_json::Value {
  serde_json::json!({ "log_cons": d.log_cons, "log_vars": d.log_vars })
}

/// The compiled metadata object printed by `--print-protocol-metadata` and
/// required, verbatim, as `ids.compiled` of every run config. Exactly the
/// keys of [`METADATA_KEYS`].
pub fn protocol_metadata() -> serde_json::Value {
  let dims: serde_json::Map<String, serde_json::Value> = compiled_dimensions()
    .iter()
    .map(|(k, d)| (k.to_string(), dimensions_json(d)))
    .collect();
  serde_json::json!({
    "transcript_domain_separator": LIMBER_TRANSCRIPT_ID,
    "protocol_wire_id": LIMBER_PROTOCOL_WIRE_ID,
    "prime_sampler_id": LIMBER_PRIME_SAMPLER_ID,
    "prime_sampler_caps": {
      "t_max": T_MAX,
      "mr_rounds": MR_ROUNDS,
      "base_draw_max": BASE_DRAW_MAX,
      "max_invocations": MAX_PRIME_INVOCATIONS,
    },
    "common_lift_binding_id": serde_json::Value::Null,
    "benchmark_coins_id": BENCHMARK_COINS_ID,
    "security_accounting_id": ids::SECURITY_ACCOUNTING_ID,
    "timing_schema_id": ids::TIMING_SCHEMA_ID,
    "tuning_protocol_id": ids::TUNING_PROTOCOL_ID,
    "comparison_schema_id": ids::COMPARISON_SCHEMA_ID,
    "kat_fixture_sha256": ids::KAT_FIXTURE_SHA256,
    "tuning_corpus_id": ids::TUNING_CORPUS_ID,
    "criterion_version": CRITERION_VERSION,
    "panic_strategy": panic_strategy(),
    "debug_assertions": cfg!(debug_assertions),
    "overflow_checks": overflow_checks_probe(),
    "argv_forms_version": ARGV_FORMS_VERSION,
    "variants": BACKEND_NAMES,
    "primary_groups": PRIMARY_GROUPS,
    "diagnostic_groups": DIAGNOSTIC_GROUPS,
    "tuning_epoch_id": tuned_defaults::TUNING_EPOCH_ID,
    "tuning_group_set": tuned_defaults::TUNING_GROUP_SET,
    "tuned_defaults": tuned_defaults::TUNED_DEFAULTS
      .iter()
      .map(|(backend, k, tuning_id)| serde_json::json!({
        "backend": backend, "k": k, "tuning_id": tuning_id,
      }))
      .collect::<Vec<_>>(),
    "check_mode": CHECK_MODE,
    "dimensions_by_k": dims,
  })
}

//
// Benchmark coins and Criterion IDs
//

/// The benchmark-coins seed of `(backend, k, instance)`: `k` must be a
/// candidate of `K_ORDER` (its zero-based position is the
/// `candidate_index`).
pub fn benchmark_seed(backend: BenchBackend, k: usize, instance: u32) -> [u8; 32] {
  let index = pb::candidate_index(k).unwrap_or_else(|| panic!("k = {k} is not a candidate"));
  pb::benchmark_coins_seed(backend, index, instance)
}

/// The deterministic prover RNG of `(backend, k, instance)`
/// (`ChaCha20Rng::from_seed` of [`benchmark_seed`]). Constructed inside
/// every timed prover sample.
pub fn benchmark_rng(backend: BenchBackend, k: usize, instance: u32) -> ChaCha20Rng {
  ChaCha20Rng::from_seed(benchmark_seed(backend, k, instance))
}

/// The `{primary|diagnostic}/blk..` token of a coordinate.
pub fn coordinate_token(block: Option<u64>) -> String {
  match block {
    Some(b) => format!("primary/blk{b}"),
    None => "diagnostic/blkdiag".to_string(),
  }
}

/// Render one Criterion function id (contract §5).
pub fn function_id(
  backend: BenchBackend,
  hashes: usize,
  dims: &Dimensions,
  k: usize,
  instance: u32,
  threads: usize,
  token: &str,
) -> String {
  format!(
    "{}/mixed3/Hpf{}-total{}/c2^{}v2^{}/k{}/inst{}/thr{}/{}",
    backend.name(),
    hashes,
    3 * hashes,
    dims.log_cons,
    dims.log_vars,
    k,
    instance,
    threads,
    token,
  )
}

//
// P0-D audit records
//

/// JSON form of one sampler audit record.
pub fn record_json(rec: &PrimeSamplerAudit) -> serde_json::Value {
  let outcome = match rec.outcome() {
    PrimeSamplerOutcome::Success(bytes) => serde_json::json!({ "success": hex(bytes) }),
    PrimeSamplerOutcome::Failure(kind) => serde_json::json!({ "failure": format!("{kind:?}") }),
  };
  serde_json::json!({
    "purpose": rec.purpose().name(),
    "width_bits": rec.width_bits(),
    "candidates": rec.candidates(),
    "bases_accepted": rec.bases_accepted(),
    "bases_rejected": rec.bases_rejected(),
    "mr_rounds_completed": rec.mr_rounds_completed(),
    "rolling_digest": hex(rec.rolling_digest()),
    "outcome": outcome,
  })
}

/// JSON form of an ordered record list.
pub fn records_json(records: &[PrimeSamplerAudit]) -> serde_json::Value {
  serde_json::Value::Array(records.iter().map(record_json).collect())
}

//
// Argv router
//

/// One of the accepted invocation forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Form {
  /// No argument at all: the smoke check.
  Smoke,
  /// `--print-protocol-metadata`.
  PrintProtocolMetadata,
  /// `--run-config P --config-sha256 H --attempt preflight --artifact-dir D`.
  RunConfig {
    config: PathBuf,
    config_sha256: String,
    artifact_dir: PathBuf,
  },
  /// `--child-config P --child-config-sha256 H --artifact-dir D`.
  ChildConfig {
    config: PathBuf,
    config_sha256: String,
    artifact_dir: PathBuf,
  },
}

/// A rejected argv (exit code 64).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UsageError(pub String);

impl std::fmt::Display for UsageError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{BENCH_NAME}: usage error: {}", self.0)
  }
}

fn usage(message: impl Into<String>) -> UsageError {
  UsageError(message.into())
}

fn syntactic_file(flag: &str, value: &str) -> Result<PathBuf, UsageError> {
  if value.starts_with("--") {
    return Err(usage(format!(
      "{flag} expects a path, got the flag {value:?}"
    )));
  }
  let path = PathBuf::from(value);
  if !path.is_absolute() {
    return Err(usage(format!(
      "{flag} must be an absolute path, got {value:?}"
    )));
  }
  Ok(path)
}

fn syntactic_hex(flag: &str, value: &str) -> Result<String, UsageError> {
  if !is_sha256_hex(value) {
    return Err(usage(format!(
      "{flag} must be 64 lowercase hex digits, got {value:?}"
    )));
  }
  Ok(value.to_string())
}

/// The pure argv router of `main` (`args` is `argv[1..]`).
///
/// * `[]` -> [`Form::Smoke`].
/// * Otherwise exactly one `--bench` token, which must be the last argument
///   (Cargo appends it after every user argument), is stripped and the rest
///   must match one of the three literal forms in [`ARGV_FORMS`] in their
///   literal token order; missing, duplicate, unknown or mixed tokens are
///   rejected.
pub fn route(args: &[String]) -> Result<Form, UsageError> {
  if args.is_empty() {
    return Ok(Form::Smoke);
  }
  let bench_tokens = args.iter().filter(|a| a.as_str() == "--bench").count();
  if bench_tokens != 1 {
    return Err(usage(format!(
      "expected exactly one trailing --bench token, found {bench_tokens}"
    )));
  }
  let (last, rest) = args.split_last().ok_or_else(|| usage("empty argv"))?;
  if last != "--bench" {
    return Err(usage("the --bench token must be the last argument"));
  }
  let rest: Vec<&str> = rest.iter().map(String::as_str).collect();
  match rest.as_slice() {
    [] => Err(usage(format!(
      "no form given after --bench; expected one of {ARGV_FORMS:?}"
    ))),
    ["--print-protocol-metadata"] => Ok(Form::PrintProtocolMetadata),
    [
      "--run-config",
      config,
      "--config-sha256",
      digest,
      "--attempt",
      attempt,
      "--artifact-dir",
      dir,
    ] => {
      if *attempt != "preflight" {
        return Err(usage(format!(
          "--attempt must be \"preflight\", got {attempt:?}"
        )));
      }
      Ok(Form::RunConfig {
        config: syntactic_file("--run-config", config)?,
        config_sha256: syntactic_hex("--config-sha256", digest)?,
        artifact_dir: syntactic_file("--artifact-dir", dir)?,
      })
    }
    [
      "--child-config",
      config,
      "--child-config-sha256",
      digest,
      "--artifact-dir",
      dir,
    ] => Ok(Form::ChildConfig {
      config: syntactic_file("--child-config", config)?,
      config_sha256: syntactic_hex("--child-config-sha256", digest)?,
      artifact_dir: syntactic_file("--artifact-dir", dir)?,
    }),
    other => Err(usage(format!(
      "unrecognized argument form {other:?}; expected exactly one of {ARGV_FORMS:?} \
       (literal token order, no extra, duplicate or mixed tokens)"
    ))),
  }
}

//
// Input files of forms 2 and 3
//

/// Read `path` (absolute, regular file, not a symlink) and require the
/// SHA-256 of its exact bytes to equal `expected_sha256`.
pub fn read_hashed_file(path: &Path, expected_sha256: &str) -> Result<Vec<u8>, String> {
  if !path.is_absolute() {
    return Err(format!("{} is not an absolute path", path.display()));
  }
  let meta =
    std::fs::symlink_metadata(path).map_err(|e| format!("cannot stat {}: {e}", path.display()))?;
  if !meta.file_type().is_file() {
    return Err(format!("{} is not a regular file", path.display()));
  }
  let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
  let digest = sha256_hex(&bytes);
  if digest != expected_sha256 {
    return Err(format!(
      "SHA-256 of {} is {digest}, expected {expected_sha256}",
      path.display()
    ));
  }
  Ok(bytes)
}

/// Require `dir` to be an absolute, existing, empty directory whose path
/// has no `.`/`..` and no symlink component.
pub fn validate_artifact_dir(dir: &Path) -> Result<(), String> {
  if !dir.is_absolute() {
    return Err(format!("{} is not an absolute path", dir.display()));
  }
  let mut prefix = PathBuf::new();
  for component in dir.components() {
    match component {
      Component::CurDir | Component::ParentDir => {
        return Err(format!("{} contains a . or .. component", dir.display()));
      }
      Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {}
    }
    prefix.push(component);
    let meta = std::fs::symlink_metadata(&prefix)
      .map_err(|e| format!("cannot stat {}: {e}", prefix.display()))?;
    if meta.file_type().is_symlink() {
      return Err(format!(
        "{} is a symlink component of {}",
        prefix.display(),
        dir.display()
      ));
    }
  }
  let meta = std::fs::metadata(dir).map_err(|e| format!("cannot stat {}: {e}", dir.display()))?;
  if !meta.is_dir() {
    return Err(format!("{} is not a directory", dir.display()));
  }
  let mut entries =
    std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
  if entries.next().is_some() {
    return Err(format!("{} is not empty", dir.display()));
  }
  Ok(())
}

/// The seven repository knobs must be absent from the process environment
/// (forms 2 and 3; the names present otherwise).
pub fn present_hidden_knobs() -> Vec<&'static str> {
  HIDDEN_KNOBS
    .iter()
    .copied()
    .filter(|k| std::env::var_os(k).is_some())
    .collect()
}

//
// Run config (form 2 / embedded in form 3)
//

/// A rejected config document: the offending key (dotted path, empty for
/// the document itself) and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
  pub key: String,
  pub message: String,
}

impl std::fmt::Display for ConfigError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    if self.key.is_empty() {
      write!(f, "{}", self.message)
    } else {
      write!(f, "{}: {}", self.key, self.message)
    }
  }
}

fn cfg_err(key: impl Into<String>, message: impl Into<String>) -> ConfigError {
  ConfigError {
    key: key.into(),
    message: message.into(),
  }
}

fn join_key(path: &str, key: &str) -> String {
  if path.is_empty() {
    key.to_string()
  } else {
    format!("{path}.{key}")
  }
}

/// A required child of an object (`path` names the object).
fn child<'a>(
  v: &'a serde_json::Value,
  path: &str,
  key: &str,
) -> Result<(&'a serde_json::Value, String), ConfigError> {
  let full = join_key(path, key);
  let obj = v
    .as_object()
    .ok_or_else(|| cfg_err(path, "must be a JSON object"))?;
  obj
    .get(key)
    .map(|c| (c, full.clone()))
    .ok_or_else(|| cfg_err(full, "missing required key"))
}

fn want_str<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str, ConfigError> {
  v.as_str()
    .ok_or_else(|| cfg_err(key, format!("must be a string, got {v}")))
}

fn want_u64(v: &serde_json::Value, key: &str) -> Result<u64, ConfigError> {
  v.as_u64()
    .ok_or_else(|| cfg_err(key, format!("must be a non-negative integer, got {v}")))
}

fn want_u32(v: &serde_json::Value, key: &str) -> Result<u32, ConfigError> {
  u32::try_from(want_u64(v, key)?).map_err(|_| cfg_err(key, "must fit in 32 bits"))
}

fn want_usize(v: &serde_json::Value, key: &str) -> Result<usize, ConfigError> {
  usize::try_from(want_u64(v, key)?).map_err(|_| cfg_err(key, "must fit in usize"))
}

fn want_bool(v: &serde_json::Value, key: &str, expected: bool) -> Result<(), ConfigError> {
  match v.as_bool() {
    Some(b) if b == expected => Ok(()),
    _ => Err(cfg_err(key, format!("must be {expected}, got {v}"))),
  }
}

fn want_obj<'a>(
  v: &'a serde_json::Value,
  key: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, ConfigError> {
  v.as_object()
    .ok_or_else(|| cfg_err(key, "must be a JSON object"))
}

fn want_arr<'a>(
  v: &'a serde_json::Value,
  key: &str,
) -> Result<&'a [serde_json::Value], ConfigError> {
  v.as_array()
    .map(Vec::as_slice)
    .ok_or_else(|| cfg_err(key, "must be a JSON array"))
}

fn want_null(v: &serde_json::Value, key: &str) -> Result<(), ConfigError> {
  if v.is_null() {
    Ok(())
  } else {
    Err(cfg_err(key, format!("must be null, got {v}")))
  }
}

fn want_literal_str(v: &serde_json::Value, key: &str, expected: &str) -> Result<(), ConfigError> {
  let s = want_str(v, key)?;
  if s == expected {
    Ok(())
  } else {
    Err(cfg_err(key, format!("must be {expected:?}, got {s:?}")))
  }
}

fn want_literal_u64(v: &serde_json::Value, key: &str, expected: u64) -> Result<(), ConfigError> {
  let n = want_u64(v, key)?;
  if n == expected {
    Ok(())
  } else {
    Err(cfg_err(key, format!("must be {expected}, got {n}")))
  }
}

fn want_enum<'s>(
  v: &serde_json::Value,
  key: &str,
  allowed: &[&'s str],
) -> Result<&'s str, ConfigError> {
  let s = want_str(v, key)?;
  allowed
    .iter()
    .copied()
    .find(|a| *a == s)
    .ok_or_else(|| cfg_err(key, format!("must be one of {allowed:?}, got {s:?}")))
}

fn want_hex(v: &serde_json::Value, key: &str) -> Result<String, ConfigError> {
  let s = want_str(v, key)?;
  if is_lower_hex(s) {
    Ok(s.to_string())
  } else {
    Err(cfg_err(key, format!("must be lowercase hex, got {s:?}")))
  }
}

fn want_opt_hex(v: &serde_json::Value, key: &str) -> Result<Option<String>, ConfigError> {
  if v.is_null() {
    Ok(None)
  } else {
    want_hex(v, key).map(Some)
  }
}

fn want_backend(v: &serde_json::Value, key: &str) -> Result<BenchBackend, ConfigError> {
  let name = want_enum(v, key, &BACKEND_NAMES)?;
  BenchBackend::parse(name).ok_or_else(|| cfg_err(key, "must be a backend name"))
}

fn want_candidate_k(v: &serde_json::Value, key: &str) -> Result<usize, ConfigError> {
  let k = want_usize(v, key)?;
  if pb::is_candidate_k(k) {
    Ok(k)
  } else {
    Err(cfg_err(
      key,
      format!("must be a candidate k in {:?}, got {k}", pb::K_RANGE),
    ))
  }
}

/// `{log_cons, log_vars}` equal to the compiled dimensions of candidate `k`.
fn want_dims(
  v: &serde_json::Value,
  key: &str,
  k: usize,
  compiled: &BTreeMap<usize, Dimensions>,
) -> Result<Dimensions, ConfigError> {
  want_obj(v, key)?;
  let (lc, lc_key) = child(v, key, "log_cons")?;
  let log_cons = want_usize(lc, &lc_key)?;
  let (lv, lv_key) = child(v, key, "log_vars")?;
  let log_vars = want_usize(lv, &lv_key)?;
  let expected = compiled
    .get(&k)
    .ok_or_else(|| cfg_err(key, format!("k = {k} has no compiled dimensions")))?;
  if log_cons != expected.log_cons || log_vars != expected.log_vars {
    return Err(cfg_err(
      key,
      format!(
        "must equal the compiled dimensions of k = {k} {{log_cons: {}, log_vars: {}}}, got \
         {{log_cons: {log_cons}, log_vars: {log_vars}}}",
        expected.log_cons, expected.log_vars
      ),
    ));
  }
  Ok(*expected)
}

/// A `dimensions_by_candidate` object: decimal-string keys naming
/// candidate `k`s, each equal to the compiled entry; `required` must all
/// be present.
fn want_dims_by_candidate(
  v: &serde_json::Value,
  key: &str,
  required: &[usize],
  compiled: &BTreeMap<usize, Dimensions>,
) -> Result<BTreeMap<usize, Dimensions>, ConfigError> {
  let obj = want_obj(v, key)?;
  let mut out = BTreeMap::new();
  for (name, entry) in obj {
    let sub = join_key(key, name);
    let k: usize = name
      .parse()
      .ok()
      .filter(|k| pb::is_candidate_k(*k) && k.to_string() == *name)
      .ok_or_else(|| cfg_err(&sub, "key must be a candidate k as a decimal string"))?;
    out.insert(k, want_dims(entry, &sub, k, compiled)?);
  }
  if out.is_empty() {
    return Err(cfg_err(key, "must not be empty"));
  }
  for k in required {
    if !out.contains_key(k) {
      return Err(cfg_err(key, format!("lacks candidate k = {k}")));
    }
  }
  Ok(out)
}

fn want_str_list(v: &serde_json::Value, key: &str) -> Result<Vec<String>, ConfigError> {
  want_arr(v, key)?
    .iter()
    .enumerate()
    .map(|(i, x)| want_str(x, &format!("{key}[{i}]")).map(str::to_string))
    .collect()
}

/// The run mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
  Normal,
  Sweep,
  Psize,
}

impl Mode {
  pub fn name(self) -> &'static str {
    match self {
      Mode::Normal => "normal",
      Mode::Sweep => "sweep",
      Mode::Psize => "psize",
    }
  }
}

/// The sweep matrix of a `sweep` run config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SweepSpec {
  /// Candidate `k`s in the literal `K_ORDER`.
  pub candidates: Vec<usize>,
  /// Message instances (`>= 1`; `0` is the held-out workload).
  pub instances: Vec<u32>,
  /// The compiled dimensions per candidate.
  pub dimensions_by_candidate: BTreeMap<usize, Dimensions>,
}

/// The keys of `run-config.json` the bench reads (unknown keys are
/// ignored).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunConfig {
  pub mode: Mode,
  pub backend: BenchBackend,
  /// `None` only in `sweep`.
  pub k: Option<usize>,
  pub hashes: usize,
  /// `None` only in `sweep`.
  pub instance: Option<u32>,
  pub threads: usize,
  /// `None` only in `sweep`.
  pub dimensions: Option<Dimensions>,
  pub dimensions_by_candidate: BTreeMap<usize, Dimensions>,
  /// `Some` only in `sweep`.
  pub sweep: Option<SweepSpec>,
  pub tuning_id: Option<String>,
  pub tuning_epoch_id: Option<String>,
  /// `None` only in `sweep`.
  pub comparison_key: Option<String>,
  pub features: Vec<String>,
}

/// Parse and validate a run-config document against this binary's compiled
/// metadata (`compiled` must be [`protocol_metadata`] at run time).
pub fn parse_run_config(
  text: &str,
  compiled: &serde_json::Value,
) -> Result<RunConfig, ConfigError> {
  let value: serde_json::Value =
    serde_json::from_str(text).map_err(|e| cfg_err("", format!("not a JSON document: {e}")))?;
  parse_run_config_value(&value, compiled)
}

/// [`parse_run_config`] on an already parsed document.
pub fn parse_run_config_value(
  value: &serde_json::Value,
  compiled: &serde_json::Value,
) -> Result<RunConfig, ConfigError> {
  if !value.is_object() {
    return Err(cfg_err("", "the run config must be a JSON object"));
  }
  let compiled_dims = compiled_dimensions();
  let (schema, key) = child(value, "", "schema")?;
  want_literal_str(schema, &key, RUN_CONFIG_SCHEMA)?;
  let (mode, key) = child(value, "", "mode")?;
  let mode = match want_enum(mode, &key, &["normal", "sweep", "psize"])? {
    "normal" => Mode::Normal,
    "sweep" => Mode::Sweep,
    _ => Mode::Psize,
  };
  let (role, key) = child(value, "", "system_role")?;
  let backend = want_backend(role, &key)?;
  let (b, key) = child(value, "", "backend")?;
  want_literal_str(b, &key, backend.name())?;

  // k, dimensions: the compiled tuned default (else the v9 default) in
  // normal/psize, null in sweep.
  let (k, key) = child(value, "", "k")?;
  let k = if mode == Mode::Sweep {
    want_null(k, &key)?;
    None
  } else {
    let k = want_candidate_k(k, &key)?;
    let expected = pb::default_k(backend);
    if k != expected {
      return Err(cfg_err(
        key,
        format!(
          "must equal the compiled default k = {expected} of backend {} (tuned default \
           {:?}, else the v9 default {}), got {k}",
          backend.name(),
          pb::tuned_default(backend),
          pb::V9_DEFAULT_K
        ),
      ));
    }
    Some(k)
  };
  let (dims, key) = child(value, "", "dimensions")?;
  let dimensions = match k {
    None => {
      want_null(dims, &key)?;
      None
    }
    Some(k) => Some(want_dims(dims, &key, k, &compiled_dims)?),
  };
  let (by_candidate, key) = child(value, "", "dimensions_by_candidate")?;
  let required: Vec<usize> = k.into_iter().collect();
  let dimensions_by_candidate =
    want_dims_by_candidate(by_candidate, &key, &required, &compiled_dims)?;

  let (workload, wkey) = child(value, "", "workload")?;
  want_obj(workload, &wkey)?;
  let (wb, key) = child(workload, &wkey, "backend")?;
  want_literal_str(wb, &key, backend.name())?;
  let (wk, key) = child(workload, &wkey, "k")?;
  match k {
    None => want_null(wk, &key)?,
    Some(k) => want_literal_u64(wk, &key, k as u64)?,
  }
  let (hashes, key) = child(workload, &wkey, "hashes_per_field")?;
  want_literal_u64(hashes, &key, CANONICAL_HASHES_PER_FIELD as u64)?;
  let hashes = want_usize(hashes, &key)?;
  let (instance, key) = child(workload, &wkey, "instance")?;
  let instance = if mode == Mode::Sweep {
    want_null(instance, &key)?;
    None
  } else {
    Some(want_u32(instance, &key)?)
  };
  let (threads, key) = child(workload, &wkey, "threads")?;
  let threads = want_usize(threads, &key)?;
  if threads == 0 {
    return Err(cfg_err(key, "must be at least 1"));
  }

  let (sweep, skey) = child(value, "", "sweep")?;
  let sweep = if mode == Mode::Sweep {
    want_obj(sweep, &skey)?;
    let (candidates, key) = child(sweep, &skey, "candidates")?;
    let candidates: Vec<usize> = want_arr(candidates, &key)?
      .iter()
      .enumerate()
      .map(|(i, c)| want_candidate_k(c, &format!("{key}[{i}]")))
      .collect::<Result<_, _>>()?;
    if candidates != K_ORDER.to_vec() {
      return Err(cfg_err(
        key,
        format!("must be the literal candidate order {K_ORDER:?}, got {candidates:?}"),
      ));
    }
    let (instances, key) = child(sweep, &skey, "instances")?;
    let instances: Vec<u32> = want_arr(instances, &key)?
      .iter()
      .enumerate()
      .map(|(i, x)| want_u32(x, &format!("{key}[{i}]")))
      .collect::<Result<_, _>>()?;
    if instances.is_empty() {
      return Err(cfg_err(key, "must not be empty"));
    }
    for (i, inst) in instances.iter().enumerate() {
      if *inst == 0 {
        return Err(cfg_err(
          format!("{key}[{i}]"),
          "instance 0 is the held-out workload and cannot be swept",
        ));
      }
      if instances[..i].contains(inst) {
        return Err(cfg_err(format!("{key}[{i}]"), "duplicate instance"));
      }
    }
    let (by_candidate, key) = child(sweep, &skey, "dimensions_by_candidate")?;
    let dimensions_by_candidate =
      want_dims_by_candidate(by_candidate, &key, &candidates, &compiled_dims)?;
    Some(SweepSpec {
      candidates,
      instances,
      dimensions_by_candidate,
    })
  } else {
    want_null(sweep, &skey)?;
    None
  };

  let (check_modes, ckey) = child(value, "", "check_modes")?;
  want_obj(check_modes, &ckey)?;
  let (shadow, key) = child(check_modes, &ckey, "shadow")?;
  want_literal_str(shadow, &key, CHECK_MODE)?;
  let (timed, key) = child(check_modes, &ckey, "timed")?;
  want_literal_str(timed, &key, TIMED_CHECK_MODE)?;

  let (criterion, ckey) = child(value, "", "criterion")?;
  want_obj(criterion, &ckey)?;
  let (n, key) = child(criterion, &ckey, "sample_size")?;
  want_literal_u64(n, &key, CRITERION_SAMPLE_SIZE)?;
  let (n, key) = child(criterion, &ckey, "warm_up_time_s")?;
  want_literal_u64(n, &key, CRITERION_WARM_UP_S)?;
  let (n, key) = child(criterion, &ckey, "measurement_time_s")?;
  want_literal_u64(n, &key, CRITERION_MEASUREMENT_S)?;

  let (tuning, tkey) = child(value, "", "tuning")?;
  want_obj(tuning, &tkey)?;
  let (tuning_id, key) = child(tuning, &tkey, "tuning_id")?;
  let tuning_id = want_opt_hex(tuning_id, &key)?;
  // A sweep precedes any tuning result, so both fields are null there;
  // normal/psize runs must carry exactly the compiled tuned default and
  // epoch of this binary.
  let expected_tuning_id = if mode == Mode::Sweep {
    None
  } else {
    pb::tuned_default(backend).map(|(_, id)| id.to_string())
  };
  if tuning_id != expected_tuning_id {
    return Err(cfg_err(
      key,
      format!(
        "must equal the compiled tuned default of backend {} {expected_tuning_id:?} (null in \
         sweep), got {tuning_id:?}",
        backend.name()
      ),
    ));
  }
  let (epoch, key) = child(tuning, &tkey, "tuning_epoch_id")?;
  let tuning_epoch_id = want_opt_hex(epoch, &key)?;
  let expected_epoch = if mode == Mode::Sweep {
    None
  } else {
    tuned_defaults::TUNING_EPOCH_ID
  };
  if tuning_epoch_id.as_deref() != expected_epoch {
    return Err(cfg_err(
      key,
      format!(
        "must equal the compiled tuning epoch {expected_epoch:?} (null in sweep), got \
         {tuning_epoch_id:?}"
      ),
    ));
  }

  let (comparison_key, key) = child(value, "", "comparison_key")?;
  let comparison_key = if mode == Mode::Sweep {
    want_null(comparison_key, &key)?;
    None
  } else {
    Some(want_hex(comparison_key, &key)?)
  };

  let (ids_obj, ikey) = child(value, "", "ids")?;
  want_obj(ids_obj, &ikey)?;
  let (compiled_ids, key) = child(ids_obj, &ikey, "compiled")?;
  if compiled_ids != compiled {
    return Err(cfg_err(
      key,
      "differs from this binary's --print-protocol-metadata object",
    ));
  }
  let (policy, pkey) = child(ids_obj, &ikey, "benchmark_coins_policy")?;
  want_obj(policy, &pkey)?;
  let (id, key) = child(policy, &pkey, "id")?;
  want_literal_str(id, &key, BENCHMARK_COINS_ID)?;
  let (framing, key) = child(policy, &pkey, "framing")?;
  want_literal_str(framing, &key, BENCHMARK_COINS_FRAMING)?;

  let (features, key) = child(value, "", "features")?;
  let features = want_str_list(features, &key)?;
  if !features.is_empty() {
    return Err(cfg_err(
      key,
      format!("must be empty (limber has no bench feature), got {features:?}"),
    ));
  }
  let (rustflags, key) = child(value, "", "rustflags")?;
  want_literal_str(rustflags, &key, REQUIRED_RUSTFLAGS)?;

  let (req, rkey) = child(value, "", "preflight_requirements")?;
  want_obj(req, &rkey)?;
  let (v, key) = child(req, &rkey, "check_mode")?;
  want_literal_str(v, &key, CHECK_MODE)?;
  let (v, key) = child(req, &rkey, "deterministic_coins")?;
  want_literal_str(v, &key, BENCHMARK_COINS_ID)?;
  let (v, key) = child(req, &rkey, "double_construction")?;
  want_bool(v, &key, true)?;
  let (v, key) = child(req, &rkey, "p0d_audit")?;
  want_bool(v, &key, true)?;
  let (v, key) = child(req, &rkey, "rayon_threads_asserted")?;
  want_literal_u64(v, &key, threads as u64)?;

  Ok(RunConfig {
    mode,
    backend,
    k,
    hashes,
    instance,
    threads,
    dimensions,
    dimensions_by_candidate,
    sweep,
    tuning_id,
    tuning_epoch_id,
    comparison_key,
    features,
  })
}

//
// Child config (form 3)
//

/// The coordinate a child measures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Coordinate {
  /// One comparable group of a normal/psize run.
  Primary { metric: &'static str, block: u64 },
  /// The diagnostic groups of a normal/psize run.
  Diagnostic,
  /// `prove_e2e` of one `(candidate, instance)` pair of a sweep.
  Sweep {
    block: u64,
    ordinal: u64,
    candidate: usize,
    instance: u32,
  },
}

/// The keys of `child-config.json` (exactly [`CHILD_CONFIG_KEYS`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChildConfig {
  pub session_id: Option<String>,
  pub parent_config_sha256: String,
  pub parent: RunConfig,
  pub coordinate: Coordinate,
  /// `None` only in `sweep`.
  pub order: Option<Vec<String>>,
  pub ordinal: u64,
}

/// Parse and validate a child-config document; the embedded parent config
/// is re-canonicalized, its digest compared to `parent_config_sha256`, and
/// then validated like a run config.
pub fn parse_child_config(
  text: &str,
  compiled: &serde_json::Value,
) -> Result<ChildConfig, ConfigError> {
  let value: serde_json::Value =
    serde_json::from_str(text).map_err(|e| cfg_err("", format!("not a JSON document: {e}")))?;
  parse_child_config_value(&value, compiled)
}

/// [`parse_child_config`] on an already parsed document.
pub fn parse_child_config_value(
  value: &serde_json::Value,
  compiled: &serde_json::Value,
) -> Result<ChildConfig, ConfigError> {
  let obj = value
    .as_object()
    .ok_or_else(|| cfg_err("", "the child config must be a JSON object"))?;
  let keys: Vec<&str> = obj.keys().map(String::as_str).collect();
  let mut sorted = keys.clone();
  sorted.sort_unstable();
  if sorted != CHILD_CONFIG_KEYS {
    return Err(cfg_err(
      "",
      format!("must have exactly the keys {CHILD_CONFIG_KEYS:?}, got {keys:?}"),
    ));
  }
  let (schema, key) = child(value, "", "schema")?;
  want_literal_str(schema, &key, CHILD_CONFIG_SCHEMA)?;
  let (session_id, key) = child(value, "", "session_id")?;
  let session_id = want_opt_hex(session_id, &key)?;
  let (digest, key) = child(value, "", "parent_config_sha256")?;
  let parent_config_sha256 = want_str(digest, &key)?.to_string();
  if !is_sha256_hex(&parent_config_sha256) {
    return Err(cfg_err(key, "must be 64 lowercase hex digits"));
  }
  let (parent_value, pkey) = child(value, "", "parent_config")?;
  want_obj(parent_value, &pkey)?;
  let canonical = canonical_json(parent_value);
  let actual = sha256_hex(canonical.as_bytes());
  if actual != parent_config_sha256 {
    return Err(cfg_err(
      pkey,
      format!("canonical digest {actual} differs from parent_config_sha256 {parent_config_sha256}"),
    ));
  }
  let parent = parse_run_config_value(parent_value, compiled).map_err(|e| ConfigError {
    key: join_key(&pkey, &e.key),
    message: e.message,
  })?;
  let (role, key) = child(value, "", "system_role")?;
  want_literal_str(role, &key, parent.backend.name())?;

  let (coordinate, ckey) = child(value, "", "coordinate")?;
  want_obj(coordinate, &ckey)?;
  let (kind, key) = child(coordinate, &ckey, "kind")?;
  let coordinate = match want_enum(kind, &key, &["primary", "diagnostic", "sweep"])? {
    "primary" => {
      let (metric, key) = child(coordinate, &ckey, "metric")?;
      let metric = want_enum(metric, &key, &PRIMARY_GROUPS)?;
      let (block, key) = child(coordinate, &ckey, "block")?;
      let block = want_u64(block, &key)?;
      Coordinate::Primary { metric, block }
    }
    "diagnostic" => {
      let (metric, key) = child(coordinate, &ckey, "metric")?;
      want_null(metric, &key)?;
      let (block, key) = child(coordinate, &ckey, "block")?;
      want_literal_str(block, &key, "diag")?;
      Coordinate::Diagnostic
    }
    _ => {
      let (block, key) = child(coordinate, &ckey, "block")?;
      let block = want_u64(block, &key)?;
      let (ordinal, key) = child(coordinate, &ckey, "ordinal")?;
      let ordinal = want_u64(ordinal, &key)?;
      let (candidate, key) = child(coordinate, &ckey, "candidate")?;
      let candidate = want_candidate_k(candidate, &key)?;
      let (instance, key) = child(coordinate, &ckey, "instance")?;
      let instance = want_u32(instance, &key)?;
      Coordinate::Sweep {
        block,
        ordinal,
        candidate,
        instance,
      }
    }
  };
  match (&coordinate, &parent.sweep) {
    (
      Coordinate::Sweep {
        candidate,
        instance,
        ..
      },
      Some(sweep),
    ) => {
      if !sweep.candidates.contains(candidate) {
        return Err(cfg_err(
          join_key(&ckey, "candidate"),
          format!("k = {candidate} is not a sweep candidate"),
        ));
      }
      if !sweep.instances.contains(instance) {
        return Err(cfg_err(
          join_key(&ckey, "instance"),
          format!("{instance} is not a sweep instance"),
        ));
      }
    }
    (Coordinate::Sweep { .. }, None) => {
      return Err(cfg_err(
        join_key(&ckey, "kind"),
        format!("sweep coordinate for a {} run", parent.mode.name()),
      ));
    }
    (_, Some(_)) => {
      return Err(cfg_err(
        join_key(&ckey, "kind"),
        "a sweep run only has sweep coordinates",
      ));
    }
    (_, None) => {}
  }

  let (order, key) = child(value, "", "order")?;
  let order = if parent.mode == Mode::Sweep {
    want_null(order, &key)?;
    None
  } else {
    let names = want_str_list(order, &key)?;
    let mut sorted = names.clone();
    sorted.sort();
    let mut expected: Vec<String> = ORDER_SYSTEMS.iter().map(|s| s.to_string()).collect();
    expected.sort();
    if sorted != expected {
      return Err(cfg_err(
        key,
        format!("must be a permutation of {ORDER_SYSTEMS:?}, got {names:?}"),
      ));
    }
    Some(names)
  };
  let (ordinal, key) = child(value, "", "ordinal")?;
  let ordinal = want_u64(ordinal, &key)?;
  if ordinal == 0 {
    return Err(cfg_err(key, "must be at least 1 (1-based)"));
  }

  Ok(ChildConfig {
    session_id,
    parent_config_sha256,
    parent,
    coordinate,
    order,
    ordinal,
  })
}

//
// Synthetic configs (tests and the smoke check)
//

/// A minimal valid normal-mode run config for `backend` against
/// `compiled`, built the way the runner does (instance 0, one thread, the
/// compiled default k and tuning ids).
pub fn synthetic_run_config(
  backend: BenchBackend,
  compiled: &serde_json::Value,
) -> serde_json::Value {
  let dims = compiled_dimensions();
  let k = pb::default_k(backend);
  let by_candidate: serde_json::Map<String, serde_json::Value> = dims
    .iter()
    .map(|(k, d)| (k.to_string(), dimensions_json(d)))
    .collect();
  serde_json::json!({
    "schema": RUN_CONFIG_SCHEMA,
    "mode": "normal",
    "system_role": backend.name(),
    "backend": backend.name(),
    "k": k,
    "dimensions": dimensions_json(&dims[&k]),
    "dimensions_by_candidate": by_candidate,
    "workload": {
      "backend": backend.name(),
      "k": k,
      "hashes_per_field": CANONICAL_HASHES_PER_FIELD,
      "instance": 0,
      "threads": 1,
    },
    "sweep": serde_json::Value::Null,
    "check_modes": { "shadow": CHECK_MODE, "timed": TIMED_CHECK_MODE },
    "criterion": {
      "sample_size": CRITERION_SAMPLE_SIZE,
      "warm_up_time_s": CRITERION_WARM_UP_S,
      "measurement_time_s": CRITERION_MEASUREMENT_S,
    },
    "tuning": {
      "tuning_id": pb::tuned_default(backend).map(|(_, id)| id),
      "tuning_epoch_id": tuned_defaults::TUNING_EPOCH_ID,
    },
    "comparison_key": "00".repeat(32),
    "ids": {
      "compiled": compiled.clone(),
      "benchmark_coins_policy": { "id": BENCHMARK_COINS_ID, "framing": BENCHMARK_COINS_FRAMING },
    },
    "features": [],
    "rustflags": REQUIRED_RUSTFLAGS,
    "preflight_requirements": {
      "check_mode": CHECK_MODE,
      "deterministic_coins": BENCHMARK_COINS_ID,
      "double_construction": true,
      "p0d_audit": true,
      "rayon_threads_asserted": 1,
    },
  })
}

/// [`synthetic_run_config`] turned into a sweep config (the full candidate
/// order, instances `1..=9`).
pub fn synthetic_sweep_config(
  backend: BenchBackend,
  compiled: &serde_json::Value,
) -> serde_json::Value {
  let mut cfg = synthetic_run_config(backend, compiled);
  let by_candidate = cfg["dimensions_by_candidate"].clone();
  cfg["mode"] = serde_json::json!("sweep");
  cfg["k"] = serde_json::Value::Null;
  cfg["dimensions"] = serde_json::Value::Null;
  cfg["workload"]["k"] = serde_json::Value::Null;
  cfg["workload"]["instance"] = serde_json::Value::Null;
  cfg["sweep"] = serde_json::json!({
    "candidates": K_ORDER,
    "instances": (1u32..=9).collect::<Vec<_>>(),
    "dimensions_by_candidate": by_candidate,
  });
  cfg["tuning"] = serde_json::json!({ "tuning_id": null, "tuning_epoch_id": null });
  cfg["comparison_key"] = serde_json::Value::Null;
  cfg
}

/// A child config embedding `parent` with `coordinate` (JSON), a normal
/// order and ordinal 1.
pub fn synthetic_child_config(
  parent: &serde_json::Value,
  coordinate: serde_json::Value,
) -> serde_json::Value {
  let sweep = parent["mode"] == "sweep";
  serde_json::json!({
    "schema": CHILD_CONFIG_SCHEMA,
    "session_id": serde_json::Value::Null,
    "parent_config_sha256": sha256_hex(canonical_json(parent).as_bytes()),
    "parent_config": parent,
    "system_role": parent["backend"],
    "coordinate": coordinate,
    "order": if sweep { serde_json::Value::Null } else { serde_json::json!(ORDER_SYSTEMS) },
    "ordinal": 1,
  })
}
