//! Dedicated preflight child for the classic-Spartan Poseidon2 suite
//! (`plan/poseidon_spartan_bench.md` §4, stages 2–4).
//!
//! Invoked by `scripts/run_poseidon_spartan_bench.sh` after the immutable
//! run config is staged. The child re-parses the environment, runs the
//! stage-1 hard-safety precheck, re-synthesizes the exact requested shape,
//! asserts dimension/digest equality with the staged config, derives the
//! execution role, and then:
//!
//! - `resource_proposal`: writes `proposal.json` (canonical bytes whose
//!   SHA-256 is the required `POSEIDON_RESOURCE_ACK` value) and exits 20 —
//!   a successful terminal state the runner maps to a nonpublishable
//!   manifest with no Criterion artifacts;
//! - `resource_checkpoint`: validates the acknowledgment against the
//!   regenerated proposal hash, performs the one diagnostic
//!   setup + `prep_prove + prove` + typed verification, writes the bound
//!   `preflight.json`, and exits 21 (successful, nonpublishable terminal
//!   state);
//! - `timing` / `proof_size`: performs the same short-lived run-bound
//!   preflight proof, writes `preflight.json` with per-stage wall times,
//!   normalized peak RSS, per-group Criterion projections, and the
//!   pass/fail decision, then exits 0 so the runner may launch the
//!   measuring process — every preflight key/proof/prep allocation dies
//!   with this process first. Exits nonzero on any failure (fail closed).
//!
//! `preflight.json` embeds the full resolved-config SHA-256, source commit
//! and lock hash, proposal hash (when applicable), resource-policy
//! version, workload/encoding digests, real/padded dimensions, stage
//! timings, peak RSS, and the decision; the runner revalidates every
//! binding field before accepting the sidecar.

#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms
)]
#![allow(non_snake_case)]

use limber::bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS};
use limber::poseidon_bench::{
  ENV_CONFIG_PATH, ENV_CONFIG_SHA256, ENV_RUN_DIR, canonical_json_bytes,
  protocol_bytes_from_full_config,
};
use limber::poseidon2::build_all_params;
use limber::poseidon2_spartan::{
  CANONICAL_CEILINGS, EMULATED_DEP_REV, ExecutionRole, REDUCTION_SCHEDULE_VERSION,
  RESOURCE_POLICY_VERSION, SpartanResolvedConfig, SpartanRunRequest, build_circuit,
  hard_safety_precheck, resolve_spartan_run, setup_poseidon_spartan, verify_poseidon_spartan,
};
use limber::provider::T256HyraxEngine;
use limber::spartan::SpartanSNARK;
use limber::traits::snark::R1CSSNARKTrait;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

type E = T256HyraxEngine;

/// Exit code for the successful `resource_proposal` terminal state.
const EXIT_PROPOSAL: u8 = 20;
/// Exit code for the successful `resource_checkpoint` terminal state.
const EXIT_CHECKPOINT: u8 = 21;

fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
  let mut h = Sha256::new();
  h.update(bytes);
  hex(&h.finalize())
}

fn run_cmd(cmd: &str, args: &[&str]) -> Result<String, String> {
  let out = std::process::Command::new(cmd)
    .args(args)
    .output()
    .map_err(|e| format!("{cmd} {args:?}: {e}"))?;
  if !out.status.success() {
    return Err(format!("{cmd} {args:?} exited with {}", out.status));
  }
  String::from_utf8(out.stdout).map_err(|e| format!("{cmd} output not UTF-8: {e}"))
}

/// Normalized peak RSS of this process in bytes (`ru_maxrss` is bytes on
/// macOS and kilobytes on Linux).
fn peak_rss_bytes() -> u64 {
  let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
  let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
  if rc != 0 {
    return 0;
  }
  let raw = unsafe { usage.assume_init() }.ru_maxrss.max(0) as u64;
  if cfg!(target_os = "macos") {
    raw
  } else {
    raw * 1024
  }
}

/// The canonical proposal document (§4): schema/policy versions, source
/// commit and lock hash, `H`, workload/encoding digests, dependency and
/// schedule identifiers, the actual shape, candidate hard limits, and
/// conservative projections. Excludes the acknowledgment, run
/// IDs/timestamps, and the final resolved-config hash so the acknowledged
/// invocation reproduces the same digest.
fn proposal_json(
  resolved: &SpartanResolvedConfig,
  git_sha: &str,
  lock_sha: &str,
) -> serde_json::Value {
  // Naive projection model, versioned with the resource policy: scale the
  // pinned H = 1 preflight prove wall (~3 s multithreaded at 2^20) by the
  // padded-domain ratio. Projections are warnings, never admission
  // evidence (§6).
  let scale = (resolved.padded_cons as f64) / f64::from(1u32 << 20);
  serde_json::json!({
    "schema": "poseidon2-spartan-resource-proposal-v1",
    "resource_policy_version": RESOURCE_POLICY_VERSION,
    "git_sha": git_sha,
    "cargo_lock_sha256": lock_sha,
    "hashes_per_field": resolved.request.hashes,
    "total_hashes": resolved.request.total_hashes,
    "workload_digest": resolved.workload_digest,
    "encoding_digest": resolved.encoding_digest,
    "gadget_rev": EMULATED_DEP_REV,
    "reduction_schedule": REDUCTION_SCHEDULE_VERSION,
    "shape": {
      "real_cons": resolved.real_cons,
      "real_vars": resolved.real_vars,
      "padded_cons": resolved.padded_cons,
      "padded_vars": resolved.padded_vars,
    },
    "candidate_hard_limits": {
      "max_padded_domain": limber::poseidon2_spartan::config::HARD_MAX_PADDED_DOMAIN,
      "max_est_rss_bytes": limber::poseidon2_spartan::config::HARD_MAX_EST_RSS_BYTES,
    },
    "projections": {
      "model": "h1-baseline-scaled-v1",
      "prove_e2e_per_sample_s_multithreaded": 3.0 * scale,
      "note": "warnings only; the checkpoint measures reality",
    },
  })
}

struct Timings {
  setup_s: f64,
  prove_s: f64,
  verify_s: f64,
}

/// One complete short-lived preflight: setup, `prep_prove + prove`, typed
/// verification, reference-digest check. Every allocation is dropped
/// before return.
fn run_preflight_proof(resolved: &SpartanResolvedConfig) -> Result<Timings, String> {
  let set = build_all_params().map_err(|e| e.to_string())?;
  let circuit = build_circuit::<E>(&set, resolved.request.hashes).map_err(|e| e.to_string())?;

  let t = Instant::now();
  let (pk, vk) = setup_poseidon_spartan::<E>(circuit.clone()).map_err(|e| e.to_string())?;
  let setup_s = t.elapsed().as_secs_f64();

  let t = Instant::now();
  let prep =
    SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).map_err(|e| e.to_string())?;
  let (proof, _prep) =
    SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, false).map_err(|e| e.to_string())?;
  let prove_s = t.elapsed().as_secs_f64();

  let t = Instant::now();
  let digests = verify_poseidon_spartan(&vk, &proof).map_err(|e| e.to_string())?;
  let verify_s = t.elapsed().as_secs_f64();
  if &digests != circuit.digests() {
    return Err("preflight verified digests do not match the circuit's claims".to_string());
  }
  Ok(Timings {
    setup_s,
    prove_s,
    verify_s,
  })
}

/// Per-group Criterion projections under the fixed Flat protocol (§4
/// stage 4): `iters_per_sample_g = max(ceil((20 / 10) / t_g), 1)`, ten
/// samples, warm-up allowance `max(1, t_g)`.
fn group_projection_s(t_g: f64) -> f64 {
  let iters = ((20.0_f64 / 10.0) / t_g).ceil().max(1.0);
  10.0 * iters * t_g + t_g.max(1.0)
}

fn write_json(path: &PathBuf, value: &serde_json::Value) -> Result<(), String> {
  let bytes = canonical_json_bytes(value);
  let tmp = path.with_extension("json.tmp");
  std::fs::write(&tmp, &bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
  std::fs::rename(&tmp, path).map_err(|e| format!("rename {}: {e}", path.display()))?;
  Ok(())
}

fn main() -> ExitCode {
  match real_main() {
    Ok(code) => code,
    Err(e) => {
      eprintln!("poseidon_spartan_preflight: {e}");
      ExitCode::FAILURE
    }
  }
}

fn real_main() -> Result<ExitCode, String> {
  let env_map: BTreeMap<std::ffi::OsString, std::ffi::OsString> = std::env::vars_os().collect();
  let request = SpartanRunRequest::parse(&env_map).map_err(|e| e.to_string())?;
  hard_safety_precheck(request.hashes).map_err(|e| e.to_string())?;

  // Stage 2: exact requested-shape synthesis, frozen and pin-checked.
  let set = build_all_params().map_err(|e| e.to_string())?;
  let circuit = build_circuit::<E>(&set, request.hashes).map_err(|e| e.to_string())?;
  let shape = ShapeCS::r1cs_shape(&circuit).map_err(|e| e.to_string())?;
  let resolved = resolve_spartan_run(request, &set, &shape).map_err(|e| e.to_string())?;
  drop(shape);
  drop(circuit);

  // Managed handshake: the staged config's protocol subsection must equal
  // this process's own resolution byte-for-byte.
  let run_dir = std::env::var_os(ENV_RUN_DIR)
    .map(PathBuf::from)
    .ok_or("preflight requires the managed-run handshake (no POSEIDON_RUN_DIR)")?;
  let config_path =
    std::env::var(ENV_CONFIG_PATH).map_err(|_| "run dir set but no config path".to_string())?;
  let passed_hash =
    std::env::var(ENV_CONFIG_SHA256).map_err(|_| "run dir set but no config hash".to_string())?;
  let file = std::fs::read(&config_path).map_err(|e| format!("read {config_path}: {e}"))?;
  if sha256_hex(&file) != passed_hash {
    return Err("run-config file hash does not match the hash the runner passed".to_string());
  }
  let file_protocol = protocol_bytes_from_full_config(
    std::str::from_utf8(&file).map_err(|e| format!("run-config not UTF-8: {e}"))?,
  )
  .map_err(|e| e.to_string())?;
  if file_protocol != resolved.protocol_canonical_bytes() {
    return Err(
      "staged protocol subsection does not match this process's own resolution".to_string(),
    );
  }

  let git_sha = run_cmd("git", &["rev-parse", "HEAD"])?.trim().to_string();
  let lock = std::fs::read("Cargo.lock").map_err(|e| format!("Cargo.lock: {e}"))?;
  let lock_sha = sha256_hex(&lock);

  // Derive the execution role against the regenerated proposal hash.
  let proposal = proposal_json(&resolved, &git_sha, &lock_sha);
  let proposal_bytes = canonical_json_bytes(&proposal);
  let proposal_hash = sha256_hex(&proposal_bytes);
  let ack_matches = resolved
    .request
    .resource_ack
    .as_ref()
    .map(|ack| *ack == proposal_hash);
  let role = resolved
    .request
    .execution_role(ack_matches)
    .map_err(|e| e.to_string())?;

  match role {
    ExecutionRole::ResourceProposal => {
      write_json(&run_dir.join("proposal.json"), &proposal)?;
      println!(
        "resource proposal written; to run the one diagnostic checkpoint:\n  \
         POSEIDON_RESOURCE_ACK={proposal_hash} scripts/run_poseidon_spartan_bench.sh"
      );
      Ok(ExitCode::from(EXIT_PROPOSAL))
    }
    ExecutionRole::ResourceCheckpoint => {
      write_json(&run_dir.join("proposal.json"), &proposal)?;
      let timings = run_preflight_proof(&resolved)?;
      let doc = preflight_doc(
        &resolved,
        &git_sha,
        &lock_sha,
        &passed_hash,
        Some(&proposal_hash),
        &timings,
        "checkpoint (nonpublishable; review and pin the §4 ceilings)",
        true,
      );
      write_json(&run_dir.join("preflight.json"), &doc)?;
      println!(
        "resource checkpoint complete: setup {:.1}s, prove_e2e {:.1}s, verify {:.3}s, \
         peak RSS {} MiB — review and pin the §4 ceilings, then rerun without the ack",
        timings.setup_s,
        timings.prove_s,
        timings.verify_s,
        peak_rss_bytes() >> 20,
      );
      Ok(ExitCode::from(EXIT_CHECKPOINT))
    }
    ExecutionRole::Timing | ExecutionRole::ProofSize => {
      let timings = run_preflight_proof(&resolved)?;
      let rss = peak_rss_bytes();
      // Stage 4: ceiling admission. Canonical H requires pinned ceilings
      // (unreachable here while they are unset — the role derivation fails
      // closed first); noncanonical H records without ceiling admission.
      let mut pass = true;
      let mut reasons: Vec<String> = Vec::new();
      if let Some(c) = CANONICAL_CEILINGS
        && resolved.request.hashes == 10
      {
        let projected = group_projection_s(timings.setup_s)
          + group_projection_s(timings.prove_s)
          + group_projection_s(timings.verify_s);
        if resolved.padded_cons.max(resolved.padded_vars) > c.max_padded_domain {
          pass = false;
          reasons.push("padded domain exceeds the pinned ceiling".to_string());
        }
        if rss > c.max_rss_bytes {
          pass = false;
          reasons.push("preflight peak RSS exceeds the pinned ceiling".to_string());
        }
        if projected > c.max_projected_wall_s as f64 {
          pass = false;
          reasons.push("projected wall-clock exceeds the pinned ceiling".to_string());
        }
      }
      let doc = preflight_doc(
        &resolved,
        &git_sha,
        &lock_sha,
        &passed_hash,
        None,
        &timings,
        if pass { "pass" } else { "fail" },
        pass,
      );
      write_json(&run_dir.join("preflight.json"), &doc)?;
      if !pass {
        return Err(format!("preflight ceilings failed: {}", reasons.join("; ")));
      }
      Ok(ExitCode::SUCCESS)
    }
  }
}

/// The run-bound `preflight.json` document (§4): every binding field the
/// runner revalidates against the current resolved config.
#[allow(clippy::too_many_arguments)]
fn preflight_doc(
  resolved: &SpartanResolvedConfig,
  git_sha: &str,
  lock_sha: &str,
  config_sha: &str,
  proposal_hash: Option<&str>,
  timings: &Timings,
  decision: &str,
  pass: bool,
) -> serde_json::Value {
  serde_json::json!({
    "schema": "poseidon2-spartan-preflight-v1",
    "resolved_config_sha256": config_sha,
    "git_sha": git_sha,
    "cargo_lock_sha256": lock_sha,
    "proposal_sha256": proposal_hash,
    "resource_policy_version": RESOURCE_POLICY_VERSION,
    "workload_digest": resolved.workload_digest,
    "encoding_digest": resolved.encoding_digest,
    "hashes_per_field": resolved.request.hashes,
    "shape": {
      "real_cons": resolved.real_cons,
      "real_vars": resolved.real_vars,
      "padded_cons": resolved.padded_cons,
      "padded_vars": resolved.padded_vars,
    },
    "stages_s": {
      "setup": timings.setup_s,
      "prove_e2e": timings.prove_s,
      "verify": timings.verify_s,
    },
    "projections_s": {
      "setup": group_projection_s(timings.setup_s),
      "prove_e2e": group_projection_s(timings.prove_s),
      "verify": group_projection_s(timings.verify_s),
    },
    "peak_rss_bytes": peak_rss_bytes(),
    "decision": decision,
    "pass": pass,
  })
}
