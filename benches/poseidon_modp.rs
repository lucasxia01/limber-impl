//! Poseidon2 (three non-native prime fields) limber benchmark: thirty
//! Poseidon2 compressions proven with `imod_r1cs_modp` in ONE mixed-modulus
//! circuit — three independent ten-compression chains, one per non-native
//! prime field (BN254-Fr, BLS12-381-Fr, secp256k1-Fr, in the fixed
//! `FIELD_ORDER`) — under the Hyrax and Brakedown Mod-PCS backends.
//!
//! Custom-harness bench target (`harness = false`, `test = false`) with one
//! unconditional `main` and a strict argv router (`support::route`), the
//! limber implementation of the Zinc plan v10 cross-system contract
//! (`scripts/NOTES-for-rust.md`). There are no environment knobs: every
//! input comes from the argv forms and the config files they name; the
//! only environment variable read is `RAYON_NUM_THREADS` (forms 2 and 3),
//! which must equal `workload.threads` and `rayon::current_num_threads()`,
//! and the seven repository knobs (`BDDIRECT BDSPEC BDROWLEN BDSPLIT
//! CHAIN_BITS GKRSKIP RUST_LOG`) must be absent. Let `args` be `argv[1..]`:
//!
//! * `[]`: the **smoke** check — serialize the compiled metadata to
//!   canonical JSON and parse it back, assert the exact primary/diagnostic
//!   group lists, and assert that the run-config and child-config parsers
//!   reject an empty document. No env reads, no proof, no Criterion. Exit 0.
//! * Otherwise the last token must be the single Cargo-injected `--bench`;
//!   it is stripped and the remaining tokens must match exactly one form:
//!   1. `--print-protocol-metadata`: one canonical JSON object plus LF on
//!      stdout. Exit 0.
//!   2. `--run-config P --config-sha256 H --attempt preflight --artifact-dir
//!      D`: the preflight attempt of `config.mode` (normal/psize: the
//!      deterministic-coins double construction with the P0-D audit ->
//!      `preflight.json` + `proof-size.json`; sweep: the complete
//!      `k x instance` matrix -> `sweep-metadata.json`), then always
//!      `attempt-result.json`. Exit 0, 3 on a recorded preflight failure,
//!      64 on usage/validation errors before any output.
//!   3. `--child-config P --child-config-sha256 H --artifact-dir D`: the
//!      pinned Criterion child (`sample_size(10)`, 1 s warm-up, 20 s
//!      measurement, `output_directory(D/criterion)`, never
//!      `configure_from_args`) registering exactly one primary group, the
//!      diagnostic groups in literal order, or the sweep pair's `prove_e2e`.
//!      Exit 0, or 64 on validation errors.
//!
//! Anything else -> `poseidon_modp: usage error: ...` on stderr, exit 64.
//!
//! Criterion IDs (contract §5): group_id = the group name, value_str empty,
//! function_id
//! `{backend}/mixed3/Hpf{H}-total{3H}/c2^{log_cons}v2^{log_vars}/k{k}/inst{i}/thr{T}/{primary/blk{b}|diagnostic/blkdiag}`.
//!
//! Prover coins: every timed prover sample regenerates `ChaCha20Rng` from
//! the pinned benchmark-coins seed (BLAKE3 of the framed
//! `(backend, candidate_index, instance)`) INSIDE the timed closure, so the
//! witness blinds and the Hyrax/IPA masking are part of `prove_e2e`;
//! Brakedown draws no prover randomness. Brakedown resets its retained
//! cache untimed before every measured commit/prove sample (layout-warm
//! steady state, empty retained cache).
//!
//! No zero-knowledge claim is made for this driver: Hyrax commitments are
//! hiding, Brakedown commitments are not, and the sumcheck transcript
//! carries unmasked witness-dependent data regardless of backend.

#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

#[path = "poseidon_modp_support.rs"]
mod support;

use criterion::{BatchSize, BenchmarkId, Criterion};
use limber::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::{
    IntModSpartanModpProverKey, IntModSpartanModpSNARK, IntModSpartanModpVerifierKey,
  },
  poseidon_bench::{
    BenchBackend, DIAGNOSTIC_GROUPS, Dimensions, LIMBER_PRIME_SAMPLER_ID, LIMBER_PROTOCOL_WIRE_ID,
    POSEIDON_LOG_T, POSEIDON_LOG_T_F, PRIMARY_GROUPS, SUMCHECK_REMAINDER_FORMULA,
    analytical_sumcheck_remainder_bytes,
  },
  poseidon2::{
    FIELD_ORDER, Layout, Poseidon2ParamsSet, PoseidonVerifierKey, build_all_params,
    build_inputs_instance, build_shape, check_canonical_io, compute_advice, expected_chain,
    validate_advice, verify_poseidon_chain, verify_poseidon_chain_with_prime_audit,
  },
  prime_sampler::{PrimeAuditedOutcome, PrimeSamplerAudit},
  provider::{
    T256DynPrimeBdEngine, T256DynPrimeEngine,
    keccak::Keccak256Transcript,
    pcs::{
      bd_retained_cache_reset, bd_retained_cache_stats, f_chunk_len, integer_modpcs::IntEvalParams,
      prewarm_brakedown_params,
    },
  },
  traits::mod_engine::ModEngine,
};
use num_bigint::BigUint;
use rand_chacha::ChaCha20Rng;
use std::{hint::black_box, path::Path, time::Duration};
use support::*;

/// Exit status of a successful run.
const EXIT_OK: i32 = 0;
/// Exit status when a form-2 attempt recorded a fail-closed failure.
const EXIT_PREFLIGHT_FAILED: i32 = 3;
/// Exit status of a usage/validation error before any output.
const EXIT_USAGE: i32 = 64;
/// The Criterion output subdirectory of a child.
const CRITERION_SUBDIR: &str = "criterion";

//
// Entry point
//

fn main() {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let form = match route(&args) {
    Ok(form) => form,
    Err(e) => {
      eprintln!("{e}");
      std::process::exit(EXIT_USAGE);
    }
  };
  let code = match form {
    Form::Smoke => smoke(),
    Form::PrintProtocolMetadata => {
      print!("{}", canonical_json(&protocol_metadata()));
      EXIT_OK
    }
    Form::RunConfig {
      config,
      config_sha256,
      artifact_dir,
    } => run_config_form(&config, &config_sha256, &artifact_dir),
    Form::ChildConfig {
      config,
      config_sha256,
      artifact_dir,
    } => child_config_form(&config, &config_sha256, &artifact_dir),
  };
  std::process::exit(code);
}

/// The no-argument smoke check.
fn smoke() -> i32 {
  let metadata = protocol_metadata();
  let text = canonical_json(&metadata);
  let parsed: serde_json::Value = serde_json::from_str(&text).expect("metadata round-trips");
  assert_eq!(parsed, metadata, "metadata canonical JSON round trip");
  assert_eq!(PRIMARY_GROUPS, ["prove_e2e", "verify_core"]);
  assert_eq!(
    DIAGNOSTIC_GROUPS,
    [
      "setup",
      "advice",
      "commit_witness",
      "prove_after_input_commit"
    ]
  );
  assert_eq!(
    metadata["primary_groups"],
    serde_json::json!(PRIMARY_GROUPS)
  );
  assert_eq!(
    metadata["diagnostic_groups"],
    serde_json::json!(DIAGNOSTIC_GROUPS)
  );
  assert_eq!(metadata["argv_forms_version"], serde_json::json!(2));
  assert_eq!(metadata["check_mode"], serde_json::json!("not_applicable"));
  for document in ["", "{}"] {
    let run = parse_run_config(document, &metadata);
    assert!(run.is_err(), "run config {document:?} must be rejected");
    let child = parse_child_config(document, &metadata);
    assert!(child.is_err(), "child config {document:?} must be rejected");
  }
  eprintln!("{BENCH_NAME}: smoke check ok");
  EXIT_OK
}

/// Print a validation error and return the usage exit status.
fn usage_failure(message: impl std::fmt::Display) -> i32 {
  eprintln!("{BENCH_NAME}: validation error: {message}");
  EXIT_USAGE
}

/// The closed-environment checks of forms 2 and 3: the seven repository
/// knobs must be absent, `RAYON_NUM_THREADS` must be present and equal
/// the config's thread count, and the global rayon pool must have exactly
/// that many threads. Returns the asserted thread count.
fn validate_environment(threads: usize) -> Result<usize, String> {
  let knobs = present_hidden_knobs();
  if !knobs.is_empty() {
    return Err(format!(
      "the repository knobs {knobs:?} are set; a canonical run requires all of \
       BDDIRECT BDSPEC BDROWLEN BDSPLIT CHAIN_BITS GKRSKIP RUST_LOG unset"
    ));
  }
  let n: usize = match std::env::var("RAYON_NUM_THREADS") {
    Ok(v) => v
      .trim()
      .parse()
      .map_err(|e| format!("RAYON_NUM_THREADS={v:?} is not an integer: {e}"))?,
    Err(e) => {
      return Err(format!(
        "RAYON_NUM_THREADS must be set to workload.threads={threads}: {e}"
      ));
    }
  };
  if n != threads {
    return Err(format!(
      "RAYON_NUM_THREADS={n} differs from workload.threads={threads}"
    ));
  }
  let pool = rayon::current_num_threads();
  if pool != threads {
    return Err(format!(
      "rayon::current_num_threads() = {pool} differs from workload.threads={threads}"
    ));
  }
  Ok(pool)
}

//
// Backend adapter and workload fixture
//

/// The one thing with no generic form: the per-engine inherent
/// `setup_with_params`. Bench-local; no public library item is bounded
/// by it.
trait PoseidonBackend: ModEngine<TE = Keccak256Transcript<Self>> + Sized {
  /// The backend this engine implements.
  const BACKEND: BenchBackend;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  >;
}

impl PoseidonBackend for T256DynPrimeEngine {
  const BACKEND: BenchBackend = BenchBackend::Hyrax;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  > {
    IntModSpartanModpSNARK::<Self>::setup_with_params(shape, params)
  }
}

impl PoseidonBackend for T256DynPrimeBdEngine {
  const BACKEND: BenchBackend = BenchBackend::Brakedown;
  fn setup_with_params(
    shape: IntModR1CSShapeModp<Self>,
    params: IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<Self>,
      IntModSpartanModpVerifierKey<Self>,
    ),
    SpartanError,
  > {
    IntModSpartanModpSNARK::<Self>::setup_with_params(shape, params)
  }
}

/// Dispatch a generic function over the validated backend.
macro_rules! dispatch {
  ($backend:expr, $f:ident($($arg:expr),* $(,)?)) => {
    match $backend {
      BenchBackend::Hyrax => $f::<T256DynPrimeEngine>($($arg),*),
      BenchBackend::Brakedown => $f::<T256DynPrimeBdEngine>($($arg),*),
    }
  };
}

/// Reset the retained cache in an untimed setup closure (Brakedown only;
/// a no-op for Hyrax).
fn maybe_reset<B: PoseidonBackend>() {
  if B::BACKEND.uses_retained_cache() {
    bd_retained_cache_reset();
  }
}

/// The keyless workload of one `(backend, H, k, instance)`: the parameter
/// set, the single mixed-modulus shape and layout, the derived IntEval
/// params, the messages, the validated combined advice and the three
/// digests. Everything a timed operation consumes is cloned from here in
/// an untimed setup closure.
struct Workload<B: PoseidonBackend> {
  set: Poseidon2ParamsSet,
  shape: IntModR1CSShapeModp<B>,
  layout: Layout,
  ie: IntEvalParams,
  messages: Vec<BigUint>,
  w: Vec<BigUint>,
  q: Vec<BigUint>,
  digests: [BigUint; 3],
  dims: Dimensions,
}

/// The dimensions of a built layout.
fn layout_dims(layout: &Layout) -> Dimensions {
  Dimensions {
    num_cons: layout.num_cons(),
    num_vars: layout.num_vars(),
    log_cons: layout.num_cons().ilog2() as usize,
    log_vars: layout.num_vars().ilog2() as usize,
    log_n: layout.log_n(),
  }
}

/// A 32-byte big-endian digest as 64 lowercase hex digits.
fn digest_hex(d: &BigUint) -> String {
  format!("{d:064x}")
}

/// One complete construction from the benchmark coins: the seeded RNG is
/// created here, then advice, `W`/`Q` commitment and the audited prove
/// consume it in that order. Returns the witness, instance, proof and the
/// prover's audit records (or the failed outcome's records).
#[allow(clippy::type_complexity)]
fn construct_from_coins<B: PoseidonBackend>(
  wl: &Workload<B>,
  pk: &IntModSpartanModpProverKey<B>,
  instance_index: u32,
  k: usize,
) -> Result<
  (
    IntModR1CSWitnessModp<B>,
    IntModR1CSInstanceModp<B>,
    IntModSpartanModpSNARK<B>,
    Vec<PrimeSamplerAudit>,
  ),
  (String, Vec<PrimeSamplerAudit>),
> {
  let mut rng: ChaCha20Rng = benchmark_rng(B::BACKEND, k, instance_index);
  let (w, q, digests) = compute_advice(&wl.set, &wl.layout, &wl.messages)
    .map_err(|e| (format!("advice: {e}"), Vec::new()))?;
  if digests != wl.digests {
    return Err((
      "advice digests differ from the preflight digests".to_string(),
      Vec::new(),
    ));
  }
  let (witness, instance) =
    IntModR1CSWitnessModp::<B>::new_with_rng(&wl.shape, pk.ck(), w, q, digests.to_vec(), &mut rng)
      .map_err(|e| (format!("commit: {e}"), Vec::new()))?;
  match IntModSpartanModpSNARK::<B>::prove_with_prime_audit_rng(pk, &instance, &witness, &mut rng) {
    PrimeAuditedOutcome::Success { value, records } => Ok((witness, instance, value, records)),
    PrimeAuditedOutcome::Failure { source, records } => Err((format!("prove: {source}"), records)),
  }
}

//
// Form 2: preflight attempt
//

/// One file written into the artifact directory.
struct Output {
  name: &'static str,
  size: usize,
  sha256: String,
}

fn write_canonical(dir: &Path, name: &'static str, value: &serde_json::Value) -> Output {
  let text = canonical_json(value);
  std::fs::write(dir.join(name), &text)
    .unwrap_or_else(|e| panic!("cannot write {}: {e}", dir.join(name).display()));
  Output {
    name,
    size: text.len(),
    sha256: sha256_hex(text.as_bytes()),
  }
}

/// The failure an attempt records in `attempt-result.json`.
struct AttemptError {
  stage: String,
  code: &'static str,
  message: String,
}

/// Everything a form-2 attempt needs beyond the generic backend.
struct AttemptContext<'a> {
  cfg: &'a RunConfig,
  config_sha256: &'a str,
  dir: &'a Path,
  compiled: &'a serde_json::Value,
  rayon_threads_asserted: usize,
}

fn run_config_form(config: &Path, config_sha256: &str, artifact_dir: &Path) -> i32 {
  let compiled = protocol_metadata();
  let bytes = match read_hashed_file(config, config_sha256) {
    Ok(b) => b,
    Err(e) => return usage_failure(e),
  };
  if let Err(e) = validate_artifact_dir(artifact_dir) {
    return usage_failure(e);
  }
  let text = match String::from_utf8(bytes) {
    Ok(t) => t,
    Err(e) => return usage_failure(format!("run config is not UTF-8: {e}")),
  };
  let cfg = match parse_run_config(&text, &compiled) {
    Ok(c) => c,
    Err(e) => return usage_failure(format!("run config: {e}")),
  };
  let rayon_threads_asserted = match validate_environment(cfg.threads) {
    Ok(n) => n,
    Err(e) => return usage_failure(e),
  };
  let ctx = AttemptContext {
    cfg: &cfg,
    config_sha256,
    dir: artifact_dir,
    compiled: &compiled,
    rayon_threads_asserted,
  };
  dispatch!(cfg.backend, attempt(&ctx))
}

fn attempt<B: PoseidonBackend>(ctx: &AttemptContext<'_>) -> i32 {
  let cfg = ctx.cfg;
  let mut outputs: Vec<Output> = Vec::new();
  let outcome: Result<(), AttemptError> = match cfg.mode {
    Mode::Normal | Mode::Psize => {
      let k = cfg.k.expect("validated: k is set");
      let input = PreflightInput {
        k,
        hashes: cfg.hashes,
        instance: cfg.instance.expect("validated: instance is set"),
        expected_dims: cfg.dimensions.expect("validated: dimensions are set"),
        config_sha256: ctx.config_sha256,
        comparison_key: cfg.comparison_key.as_deref(),
        rayon_threads_asserted: ctx.rayon_threads_asserted,
      };
      let result = preflight::<B>(&input);
      outputs.push(write_canonical(ctx.dir, "preflight.json", &result.record));
      match (result.proof_size, result.failure) {
        (Some(size), None) => {
          outputs.push(write_canonical(ctx.dir, "proof-size.json", &size));
          Ok(())
        }
        (_, Some(failure)) => Err(AttemptError {
          stage: failure.stage,
          code: "preflight_failed",
          message: failure.message,
        }),
        (None, None) => unreachable!("a successful preflight has a proof-size record"),
      }
    }
    Mode::Sweep => {
      let sweep = cfg.sweep.as_ref().expect("validated: sweep is set");
      let mut matrix = Vec::new();
      let mut first_failure: Option<AttemptError> = None;
      for candidate in &sweep.candidates {
        for instance in &sweep.instances {
          let dims = sweep.dimensions_by_candidate[candidate];
          let input = PreflightInput {
            k: *candidate,
            hashes: cfg.hashes,
            instance: *instance,
            expected_dims: dims,
            config_sha256: ctx.config_sha256,
            comparison_key: None,
            rayon_threads_asserted: ctx.rayon_threads_asserted,
          };
          let result = preflight::<B>(&input);
          let status = if result.failure.is_none() {
            "ok"
          } else {
            "failed"
          };
          matrix.push(serde_json::json!({
            "candidate": candidate,
            "instance": instance,
            "dimensions": dimensions_json(&dims),
            "status": status,
            "preflight": result.record,
          }));
          if let Some(failure) = result.failure
            && first_failure.is_none()
          {
            first_failure = Some(AttemptError {
              stage: failure.stage,
              code: "sweep_pair_failed",
              message: format!("k{candidate}/inst{instance}: {}", failure.message),
            });
          }
        }
      }
      let record = serde_json::json!({
        "schema": SWEEP_METADATA_SCHEMA,
        "config_sha256": ctx.config_sha256,
        "candidates": sweep.candidates,
        "instances": sweep.instances,
        "matrix": matrix,
      });
      outputs.push(write_canonical(ctx.dir, "sweep-metadata.json", &record));
      first_failure.map_or(Ok(()), Err)
    }
  };
  outputs.sort_by(|a, b| a.name.cmp(b.name));
  let error = outcome
    .as_ref()
    .err()
    .map(|e| serde_json::json!({"stage": e.stage, "code": e.code, "message": e.message}));
  let record = serde_json::json!({
    "schema": ATTEMPT_RESULT_SCHEMA,
    "attempt": "preflight",
    "mode": cfg.mode.name(),
    "config_sha256": ctx.config_sha256,
    "status": if outcome.is_ok() { "ok" } else { "failed" },
    "outputs": outputs.iter().map(|o| serde_json::json!({
      "name": o.name, "size": o.size, "sha256": o.sha256,
    })).collect::<Vec<_>>(),
    "error": error,
    "compiled": ctx.compiled,
  });
  // Always last.
  write_canonical(ctx.dir, "attempt-result.json", &record);
  match outcome {
    Ok(()) => EXIT_OK,
    Err(e) => {
      eprintln!(
        "[preflight] FAILED at stage {} ({}): {}",
        e.stage, e.code, e.message
      );
      EXIT_PREFLIGHT_FAILED
    }
  }
}

//
// Preflight
//

/// A preflight failure: the stage that failed and its message.
struct PreflightFailure {
  stage: String,
  message: String,
}

fn fail(stage: impl Into<String>, message: impl std::fmt::Display) -> PreflightFailure {
  PreflightFailure {
    stage: stage.into(),
    message: message.to_string(),
  }
}

/// The inputs of one preflight (one `(backend, H, k, instance)`).
struct PreflightInput<'a> {
  k: usize,
  hashes: usize,
  instance: u32,
  /// The dimensions the config announces (checked against the layout).
  expected_dims: Dimensions,
  config_sha256: &'a str,
  /// Copied into `proof-size.json` (`None` for a sweep pair, which writes
  /// no size record).
  comparison_key: Option<&'a str>,
  rayon_threads_asserted: usize,
}

/// The serialized components of one construction that the double
/// construction compares.
#[derive(Clone, PartialEq, Eq)]
struct ConstructionBytes {
  commitments: Vec<u8>,
  eval_arg: Vec<u8>,
  remainder: Vec<u8>,
  remainder_counts: (usize, usize, usize),
  records: Vec<PrimeSamplerAudit>,
}

/// The accumulating preflight state (every field is `None`/empty until
/// its stage completed, so a failure record shows how far the preflight
/// got).
#[derive(Default)]
struct PreflightState {
  dims: Option<Dimensions>,
  shape_satisfiable: Option<bool>,
  canonical_io_ok: Option<bool>,
  digests: Vec<String>,
  prover_records: Vec<PrimeSamplerAudit>,
  prover_audit_complete: bool,
  verifier_records: Vec<PrimeSamplerAudit>,
  commitments_equal: Option<bool>,
  components_equal: Option<bool>,
  remainder_equal: Option<bool>,
  audits_equal: Option<bool>,
  remainder_counts: Option<(usize, usize, usize)>,
  remainder_serialized_bytes: Option<usize>,
}

/// What one preflight produces: the record (always), the proof-size record
/// (on success) and the failure (on failure).
struct PreflightResult {
  record: serde_json::Value,
  proof_size: Option<serde_json::Value>,
  failure: Option<PreflightFailure>,
}

/// Preflight (contract §4): build params/shape/layout for `H` at the
/// instance, derive the IntEval params for `k` (shape satisfiability),
/// compute and validate the advice, check every field's reference chain
/// and the canonical IO, construct the keys, run TWO complete fresh
/// constructions from the same benchmark coins (advice + commit + audited
/// prove) requiring identical commitment bytes, identical evaluation
/// argument and sumcheck-remainder bytes, identical remainder structure
/// and identical complete P0-D audit logs, then verify with the audited
/// chain verifier requiring the verifier's audit to equal the prover's.
/// Every failure — including a panic inside library code — is a recorded
/// stage; a sweep continues with the next pair.
fn preflight<B: PoseidonBackend>(input: &PreflightInput<'_>) -> PreflightResult {
  let mut st = PreflightState::default();
  let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    preflight_inner::<B>(input, &mut st)
  }));
  let result = match result {
    Ok(r) => r,
    Err(payload) => {
      let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "non-string panic payload".to_string());
      Err(fail("panic", message))
    }
  };
  let (proof_size, failure) = match result {
    Ok(size) => (Some(size), None),
    Err(e) => (None, Some(e)),
  };
  let record = serde_json::json!({
    "schema": PREFLIGHT_SCHEMA,
    "status": if failure.is_none() { "ok" } else { "failed" },
    "config_sha256": input.config_sha256,
    "backend": B::BACKEND.name(),
    "k": input.k,
    "hashes_per_field": input.hashes,
    "instance": input.instance,
    "dimensions": st.dims.as_ref().map(dimensions_json),
    "rayon_threads_asserted": input.rayon_threads_asserted,
    "prime_audit": {
      "complete": st.prover_audit_complete,
      "prime_sampler_id": LIMBER_PRIME_SAMPLER_ID,
      "invocations": st.prover_records.len(),
      "records": records_json(&st.prover_records),
      "verifier_records": records_json(&st.verifier_records),
    },
    "double_construction": {
      "commitments_equal": st.commitments_equal.unwrap_or(false),
      "components_equal": st.components_equal.unwrap_or(false),
      "remainder_equal": st.remainder_equal.unwrap_or(false),
      "audits_equal": st.audits_equal.unwrap_or(false),
    },
    "remainder": {
      "outer_rounds": st.remainder_counts.map(|c| c.0),
      "inner_rounds": st.remainder_counts.map(|c| c.1),
      "claimed_evaluations": st.remainder_counts.map(|c| c.2),
      "serialized_bytes": st.remainder_serialized_bytes,
    },
    "digests": st.digests,
    "canonical_io_ok": st.canonical_io_ok.unwrap_or(false),
    "shape_satisfiable": st.shape_satisfiable.unwrap_or(false),
    "error": failure.as_ref().map(|e| serde_json::json!({
      "stage": e.stage, "code": "preflight_failed", "message": e.message,
    })),
  });
  PreflightResult {
    record,
    proof_size,
    failure,
  }
}

/// Build the keyless workload of `(H, k, instance)` (stages `params`,
/// `shape`, `dimensions`, `shape_satisfiable`, `messages`, `advice`,
/// `digest`, `canonical_io`, `prewarm`), recording progress in `st`.
fn build_workload<B: PoseidonBackend>(
  input: &PreflightInput<'_>,
  st: &mut PreflightState,
) -> Result<Workload<B>, PreflightFailure> {
  let set = build_all_params().map_err(|e| fail("params", e))?;
  let (shape, layout) = build_shape::<B>(&set, input.hashes).map_err(|e| fail("shape", e))?;
  let dims = layout_dims(&layout);
  st.dims = Some(dims);
  if dims.log_cons != input.expected_dims.log_cons || dims.log_vars != input.expected_dims.log_vars
  {
    return Err(fail(
      "dimensions",
      format!(
        "config announces log_cons = {}, log_vars = {}; the layout has {}, {}",
        input.expected_dims.log_cons, input.expected_dims.log_vars, dims.log_cons, dims.log_vars
      ),
    ));
  }
  let ie = match IntEvalParams::derive(POSEIDON_LOG_T_F, POSEIDON_LOG_T, input.k, layout.log_n()) {
    Ok(ie) => {
      st.shape_satisfiable = Some(true);
      ie
    }
    Err(e) => {
      st.shape_satisfiable = Some(false);
      return Err(fail("shape_satisfiable", e));
    }
  };
  let messages =
    build_inputs_instance(input.instance, input.hashes).map_err(|e| fail("messages", e))?;
  let (w, q, digests) = compute_advice(&set, &layout, &messages).map_err(|e| fail("advice", e))?;
  validate_advice(&set, &layout, &w, &q, &digests).map_err(|e| fail("validate_advice", e))?;
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let chain = expected_chain(set.get(*field), &messages).map_err(|e| fail("digest", e))?;
    if chain.last() != Some(&digests[f]) {
      return Err(fail(
        "digest",
        format!(
          "digest of block {} differs from the reference chain",
          field.name()
        ),
      ));
    }
  }
  st.digests = digests.iter().map(digest_hex).collect();
  match check_canonical_io(&digests, &set) {
    Ok(()) => st.canonical_io_ok = Some(true),
    Err(e) => {
      st.canonical_io_ok = Some(false);
      return Err(fail("canonical_io", e));
    }
  }
  if B::BACKEND.uses_retained_cache() {
    // Pre-build the deterministic code layouts for the input chunk
    // length (w and q share it). Published Brakedown results are
    // layout-warm steady state; no cold number is reported.
    let n = layout.num_vars().max(layout.num_cons());
    let len = f_chunk_len(&ie, n).map_err(|e| fail("prewarm", e))?;
    let _ = prewarm_brakedown_params(len);
  }
  eprintln!(
    "[preflight] {} H={} k={} inst={} c2^{}v2^{} log_p={} s={}",
    B::BACKEND.name(),
    input.hashes,
    input.k,
    input.instance,
    dims.log_cons,
    dims.log_vars,
    ie.log_p,
    ie.s,
  );
  Ok(Workload {
    set,
    shape,
    layout,
    ie,
    messages,
    w,
    q,
    digests,
    dims,
  })
}

fn preflight_inner<B: PoseidonBackend>(
  input: &PreflightInput<'_>,
  st: &mut PreflightState,
) -> Result<serde_json::Value, PreflightFailure> {
  let wl = build_workload::<B>(input, st)?;

  // Keys.
  let (pk, vk) =
    B::setup_with_params(wl.shape.clone(), wl.ie.clone()).map_err(|e| fail("keys", e))?;
  let pvk = PoseidonVerifierKey::new(vk, &wl.set, &wl.layout).map_err(|e| fail("keys", e))?;
  let scheduled = IntModSpartanModpSNARK::<B>::prime_sampler_invocations_for_prove(&pk)
    .map_err(|e| fail("audit_schedule", e))?;

  // Two complete fresh constructions from the same coins.
  let mut first: Option<(
    IntModR1CSInstanceModp<B>,
    IntModSpartanModpSNARK<B>,
    ConstructionBytes,
  )> = None;
  let mut second: Option<ConstructionBytes> = None;
  for pass in 0..2 {
    maybe_reset::<B>();
    let (witness, instance, proof, records) =
      match construct_from_coins::<B>(&wl, &pk, input.instance, input.k) {
        Ok(v) => v,
        Err((message, records)) => {
          st.prover_records = records;
          return Err(fail(format!("construction_{pass}"), message));
        }
      };
    drop(witness);
    let bytes = ConstructionBytes {
      commitments: instance
        .commitment_bytes()
        .map_err(|e| fail(format!("construction_{pass}"), e))?,
      eval_arg: proof
        .eval_arg_bytes()
        .map_err(|e| fail(format!("construction_{pass}"), e))?,
      remainder: proof.sumcheck_remainder_bytes(),
      remainder_counts: proof.sumcheck_remainder_counts(),
      records,
    };
    if pass == 0 {
      st.prover_records = bytes.records.clone();
      st.prover_audit_complete = bytes.records.len() == scheduled;
      st.remainder_counts = Some(bytes.remainder_counts);
      st.remainder_serialized_bytes = Some(bytes.remainder.len());
      first = Some((instance, proof, bytes));
    } else {
      second = Some(bytes);
    }
  }
  let (instance, proof, a) = first.expect("first construction");
  let b = second.expect("second construction");
  st.commitments_equal = Some(a.commitments == b.commitments);
  st.components_equal = Some(a.eval_arg == b.eval_arg && a.remainder == b.remainder);
  st.remainder_equal = Some(a.remainder_counts == b.remainder_counts && a.remainder == b.remainder);
  let prover_audits_equal = a.records == b.records;
  if !(st.commitments_equal == Some(true)
    && st.components_equal == Some(true)
    && st.remainder_equal == Some(true)
    && prover_audits_equal)
  {
    st.audits_equal = Some(false);
    return Err(fail(
      "double_construction",
      format!(
        "commitments_equal={} eval_arg_equal={} remainder_equal={} prover_audits_equal={}",
        a.commitments == b.commitments,
        a.eval_arg == b.eval_arg,
        a.remainder == b.remainder,
        prover_audits_equal
      ),
    ));
  }
  if !st.prover_audit_complete {
    st.audits_equal = Some(false);
    return Err(fail(
      "audit",
      format!(
        "the prover audit holds {} records, the schedule requires {scheduled}",
        a.records.len()
      ),
    ));
  }

  // Audited verification: the verifier's records must equal the prover's.
  match verify_poseidon_chain_with_prime_audit(&pvk, &instance, &proof) {
    PrimeAuditedOutcome::Success { records, .. } => st.verifier_records = records,
    PrimeAuditedOutcome::Failure { source, records } => {
      st.verifier_records = records;
      st.audits_equal = Some(false);
      return Err(fail("verify", source));
    }
  }
  let audits_equal = st.verifier_records == a.records;
  st.audits_equal = Some(audits_equal);
  if !audits_equal {
    return Err(fail(
      "audit",
      "the verifier's prime-sampler audit differs from the prover's",
    ));
  }
  for (i, rec) in a.records.iter().enumerate() {
    eprintln!(
      "[preflight] record {i}: {} width={} candidates={} bases={}+{} mr_rounds={}",
      rec.purpose().name(),
      rec.width_bits(),
      rec.candidates(),
      rec.bases_accepted(),
      rec.bases_rejected(),
      rec.mr_rounds_completed()
    );
  }

  // The proof-size record of the verified proof.
  let analytical = analytical_sumcheck_remainder_bytes(wl.dims.log_cons, wl.dims.log_vars);
  eprintln!(
    "[preflight] proof size: commitments={} B eval_arg={} B sumcheck remainder={} B \
     (analytical payload, no framing; serialized {} B); public digests (3) excluded",
    a.commitments.len(),
    a.eval_arg.len(),
    analytical,
    a.remainder.len()
  );
  Ok(serde_json::json!({
    "schema": PROOF_SIZE_SCHEMA,
    "metric_kind": PROOF_SIZE_METRIC_KIND,
    "config_sha256": input.config_sha256,
    "comparison_key": input.comparison_key,
    "wire_id": LIMBER_PROTOCOL_WIRE_ID,
    "backend": B::BACKEND.name(),
    "k": input.k,
    "hashes_per_field": input.hashes,
    "instance": input.instance,
    "components": {
      "commitments_bytes": a.commitments.len(),
      "eval_arg_bytes": a.eval_arg.len(),
      "commitments_sha256": sha256_hex(&a.commitments),
      "eval_arg_sha256": sha256_hex(&a.eval_arg),
    },
    "analytical_remainder": {
      "sumcheck_remainder_bytes": analytical,
      "log_cons": wl.dims.log_cons,
      "log_vars": wl.dims.log_vars,
      "formula": SUMCHECK_REMAINDER_FORMULA,
    },
    "prime_sampler_id": LIMBER_PRIME_SAMPLER_ID,
    "verified": true,
  }))
}

//
// Form 3: Criterion child
//

fn child_config_form(config: &Path, config_sha256: &str, artifact_dir: &Path) -> i32 {
  let compiled = protocol_metadata();
  let bytes = match read_hashed_file(config, config_sha256) {
    Ok(b) => b,
    Err(e) => return usage_failure(e),
  };
  if let Err(e) = validate_artifact_dir(artifact_dir) {
    return usage_failure(e);
  }
  let text = match String::from_utf8(bytes) {
    Ok(t) => t,
    Err(e) => return usage_failure(format!("child config is not UTF-8: {e}")),
  };
  let cc = match parse_child_config(&text, &compiled) {
    Ok(c) => c,
    Err(e) => return usage_failure(format!("child config: {e}")),
  };
  if let Err(e) = validate_environment(cc.parent.threads) {
    return usage_failure(e);
  }
  dispatch!(cc.parent.backend, child(&cc, artifact_dir))
}

/// The pinned Criterion object of a child (never `configure_from_args`).
fn criterion(artifact_dir: &Path) -> Criterion {
  Criterion::default()
    .sample_size(10)
    .warm_up_time(Duration::from_secs(1))
    .measurement_time(Duration::from_secs(20))
    .output_directory(&artifact_dir.join(CRITERION_SUBDIR))
}

/// A validation failure of a child before any Criterion work.
fn child_failure(message: impl std::fmt::Display) -> i32 {
  usage_failure(format!("child: {message}"))
}

/// One untimed audited iteration from the empty-cache state (Brakedown
/// only): runs `op` after a reset and reports the retained-cache stats on
/// stderr, so the measured samples' cache policy is visible in the log.
fn audit_group<B: PoseidonBackend>(group: &str, id: &str, op: impl FnOnce()) {
  if !B::BACKEND.uses_retained_cache() {
    return;
  }
  bd_retained_cache_reset();
  op();
  let stats = bd_retained_cache_stats();
  eprintln!(
    "[child] cache audit {group}/{id}: {}",
    serde_json::to_string(&stats).expect("stats serialize")
  );
}

fn child<B: PoseidonBackend>(cc: &ChildConfig, artifact_dir: &Path) -> i32 {
  let parent = &cc.parent;
  let (k, instance, token, groups): (usize, u32, String, Vec<&str>) = match &cc.coordinate {
    Coordinate::Primary { metric, block } => (
      parent.k.expect("validated: k is set"),
      parent.instance.expect("validated: instance is set"),
      coordinate_token(Some(*block)),
      vec![metric],
    ),
    Coordinate::Diagnostic => (
      parent.k.expect("validated: k is set"),
      parent.instance.expect("validated: instance is set"),
      coordinate_token(None),
      B::BACKEND.diagnostic_groups(),
    ),
    Coordinate::Sweep {
      block,
      candidate,
      instance,
      ..
    } => (
      *candidate,
      *instance,
      coordinate_token(Some(*block)),
      vec!["prove_e2e"],
    ),
  };
  let expected_dims = match &cc.coordinate {
    Coordinate::Sweep { candidate, .. } => parent
      .sweep
      .as_ref()
      .and_then(|s| s.dimensions_by_candidate.get(candidate).copied()),
    _ => parent.dimensions,
  }
  .expect("validated: the coordinate's dimensions are known");
  let input = PreflightInput {
    k,
    hashes: parent.hashes,
    instance,
    expected_dims,
    config_sha256: &cc.parent_config_sha256,
    comparison_key: None,
    rayon_threads_asserted: parent.threads,
  };
  let mut st = PreflightState::default();
  let wl = match build_workload::<B>(&input, &mut st) {
    Ok(wl) => wl,
    Err(e) => return child_failure(format!("{}: {}", e.stage, e.message)),
  };
  let id = function_id(
    B::BACKEND,
    parent.hashes,
    &wl.dims,
    k,
    instance,
    parent.threads,
    &token,
  );
  eprintln!("[child] {id} groups {groups:?}");
  let has = |g: &str| groups.contains(&g);
  let mut c = criterion(artifact_dir);

  // setup/: raw setup_with_params + checked PoseidonVerifierKey::new. No
  // other key pair exists while this group runs; the constructed pair is
  // returned so its destruction happens after the timer stops.
  if has("setup") {
    let mut g = c.benchmark_group("setup");
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        || (wl.shape.clone(), wl.ie.clone()),
        |(shape, ie)| {
          let (pk, vk) = B::setup_with_params(shape, ie).expect("setup");
          let pvk = PoseidonVerifierKey::new(vk, &wl.set, &wl.layout).expect("vk predicates");
          (pk, pvk)
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // advice/: all three blocks' compute_advice only (backend-independent;
  // only the Hyrax diagnostic child registers it).
  if has("advice") {
    let mut g = c.benchmark_group("advice");
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        || (),
        |()| compute_advice(&wl.set, &wl.layout, &wl.messages).expect("advice"),
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // The proving key pair is built only after the setup group finished, so
  // setup timing never coexists with another shape-bearing key pair.
  let (pk, pvk) = match B::setup_with_params(wl.shape.clone(), wl.ie.clone())
    .and_then(|(pk, vk)| PoseidonVerifierKey::new(vk, &wl.set, &wl.layout).map(|pvk| (pk, pvk)))
  {
    Ok(keys) => keys,
    Err(e) => return child_failure(format!("keys: {e}")),
  };

  // commit_witness/: IntModR1CSWitnessModp::new_with_rng for the combined
  // W/Q vectors, from a reset cache, with the benchmark coins.
  if has("commit_witness") {
    let mut g = c.benchmark_group("commit_witness");
    audit_group::<B>("commit_witness", &id, || {
      let mut rng = benchmark_rng(B::BACKEND, k, instance);
      let _ = IntModR1CSWitnessModp::<B>::new_with_rng(
        &wl.shape,
        pk.ck(),
        wl.w.clone(),
        wl.q.clone(),
        wl.digests.to_vec(),
        &mut rng,
      )
      .expect("commit");
    });
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        || {
          maybe_reset::<B>();
          (wl.w.clone(), wl.q.clone(), wl.digests.to_vec())
        },
        |(w, q, x)| {
          let mut rng = benchmark_rng(B::BACKEND, k, instance);
          IntModR1CSWitnessModp::<B>::new_with_rng(&wl.shape, pk.ck(), w, q, x, &mut rng)
            .expect("commit")
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // prove_after_input_commit/: untimed reset + W/Q commit (from the
  // coins), timed prove continuing the same coins — including any
  // deterministic W/Q re-encoding caused by internal commitment eviction.
  // Never described as "commit-free".
  if has("prove_after_input_commit") {
    let mut g = c.benchmark_group("prove_after_input_commit");
    audit_group::<B>("prove_after_input_commit", &id, || {
      let _ = construct_from_coins::<B>(&wl, &pk, instance, k).expect("construction");
    });
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        || {
          maybe_reset::<B>();
          let mut rng = benchmark_rng(B::BACKEND, k, instance);
          let (witness, instance) = IntModR1CSWitnessModp::<B>::new_with_rng(
            &wl.shape,
            pk.ck(),
            wl.w.clone(),
            wl.q.clone(),
            wl.digests.to_vec(),
            &mut rng,
          )
          .expect("commit");
          (witness, instance, rng)
        },
        |(witness, instance, mut rng)| {
          let proof =
            IntModSpartanModpSNARK::<B>::prove_with_rng(&pk, &instance, &witness, &mut rng)
              .expect("prove");
          (proof, witness, instance)
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // prove_e2e/: the comparable prover boundary — coins regenerated from
  // the seed, combined advice, W/Q commit and complete prove inside the
  // timed closure; the setup is the Brakedown cache reset only. The
  // witness, instance and proof are returned so their destruction happens
  // after the timer stops.
  if has("prove_e2e") {
    audit_group::<B>("prove_e2e", &id, || {
      let _ = construct_from_coins::<B>(&wl, &pk, instance, k).expect("construction");
    });
    // [boundary:prove_e2e:begin]
    let mut g = c.benchmark_group("prove_e2e");
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        maybe_reset::<B>,
        |()| -> (
          IntModR1CSWitnessModp<B>,
          IntModR1CSInstanceModp<B>,
          IntModSpartanModpSNARK<B>,
        ) {
          let mut rng: ChaCha20Rng = benchmark_rng(B::BACKEND, k, instance);
          let (w, q, digests) = compute_advice(&wl.set, &wl.layout, &wl.messages).expect("advice");
          let (witness, instance) = IntModR1CSWitnessModp::<B>::new_with_rng(
            &wl.shape,
            pk.ck(),
            w,
            q,
            digests.to_vec(),
            &mut rng,
          )
          .expect("commit");
          let proof =
            IntModSpartanModpSNARK::<B>::prove_with_rng(&pk, &instance, &witness, &mut rng)
              .expect("prove");
          (witness, instance, proof)
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
    // [boundary:prove_e2e:end]
  }

  // verify_core/: one untimed instance + proof from the coins, verified
  // once untimed; the proving key is dropped first. Each sample clones
  // exactly one proof untimed and the routine consumes it, so proof
  // destruction is inside timing; the verifier key and instance are not
  // cloned per iteration.
  if has("verify_core") {
    let (witness, instance, proof, _records) =
      match construct_from_coins::<B>(&wl, &pk, instance, k) {
        Ok(v) => v,
        Err((message, _)) => return child_failure(message),
      };
    drop(witness);
    drop(pk);
    if let Err(e) = verify_poseidon_chain(&pvk, &instance, &proof) {
      return child_failure(format!("the child's proof does not verify: {e}"));
    }
    // [boundary:verify_core:begin]
    let mut g = c.benchmark_group("verify_core");
    g.bench_function(BenchmarkId::new(&id, ""), |b| {
      b.iter_batched(
        || proof.clone(),
        |proof| {
          let verified = verify_poseidon_chain(&pvk, &instance, &proof);
          black_box(&verified);
          verified.expect("verify");
          drop(proof);
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
    // [boundary:verify_core:end]
  }

  c.final_summary();
  EXIT_OK
}
