//! Run-configuration lifecycle for the classic-Spartan Poseidon2 suite
//! (`plan/poseidon_spartan_bench.md` §4, §6): the pure
//! [`SpartanRunRequest::parse`] syntax/semantic stage, the
//! [`resolve_spartan_run`] stage that freezes one actual shape synthesis
//! into an immutable [`SpartanResolvedConfig`], the fail-closed resource
//! policy, and the derived execution role. Shares the Unicode/0-or-1/
//! numeric validation helpers with the ModP suite's
//! [`crate::poseidon_bench::RunConfig`].

use super::circuit::NUM_PUBLIC_LIMBS;
use crate::errors::SpartanError;
use crate::poseidon_bench::{
  FIELD_ORDER, HIDDEN_KNOBS, canonical_json_bytes, get_unicode, is_present, parse_bool, parse_usize,
};
use std::collections::BTreeMap;
use std::ffi::OsString;

/// Default chain length per field.
pub const DEFAULT_HASHES: usize = 10;

/// Runner resource cap on `H`. Initialized to the canonical workload size;
/// raising it requires exact shape synthesis and a resource test at the
/// proposed new cap (plan §4) because lazy-reduction thresholds and padding
/// need not extrapolate linearly.
pub const SPARTAN_MAX_HASHES: usize = 10;

/// The exact upstream revision of `bellpepper-emulated` this suite is
/// pinned to; recorded in every protocol JSON and manifest.
pub const EMULATED_DEP_REV: &str = "2f90c6b90402128dd06b44d5d7f4cf1d9785a1ed";

/// Version identifier of the explicit reduction schedule (two S-box
/// reductions plus the lane-1/2 stride-8 schedule). Any schedule change
/// bumps this and reopens the §3 decision gate.
pub const REDUCTION_SCHEDULE_VERSION: &str = "sbox2-lane8-v1";

/// Version of the resource-policy formula in [`hard_safety_precheck`].
pub const RESOURCE_POLICY_VERSION: u32 = 1;

/// Hard process-safety ceiling on the padded domain (stage 1 of the §4
/// resource gate). `POSEIDON_ALLOW_LARGE` bypasses the ordinary hash cap
/// but never this bound.
pub const HARD_MAX_PADDED_DOMAIN: usize = 1 << 26;

/// Hard process-safety ceiling on the conservatively estimated peak RSS in
/// bytes (stage 1 of the §4 resource gate).
pub const HARD_MAX_EST_RSS_BYTES: u64 = 48 << 30;

/// Version-controlled canonical ceilings for `H = 10` (stage 4 of the §4
/// resource gate). `None` until the reviewed resource checkpoint pins
/// them; while unset, managed timing/proof-size runs fail closed into the
/// `resource_proposal`/`resource_checkpoint` roles.
pub const CANONICAL_CEILINGS: Option<ResourceCeilings> = None;

/// Reviewed resource ceilings for the canonical `H = 10` configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceCeilings {
  /// Maximum accepted padded domain (constraints and variables).
  pub max_padded_domain: usize,
  /// Maximum accepted preflight peak RSS in bytes.
  pub max_rss_bytes: u64,
  /// Maximum accepted total projected wall-clock in seconds across the
  /// three groups.
  pub max_projected_wall_s: u64,
}

/// Per-permutation pilot constraint count (upper of the three fields,
/// pinned by `pilot_counts_pinned`), used only by the conservative stage-1
/// estimate.
const PILOT_PERM_CONS_UPPER: usize = 309_414;

/// Conservative slack multiplier numerator/denominator for the stage-1
/// estimate (25% headroom over the pilot-derived projection).
const EST_SLACK_NUM: usize = 5;
/// Denominator of the stage-1 slack multiplier.
const EST_SLACK_DEN: usize = 4;

/// Benchmark mode, resolved from `PSIZE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpartanBenchMode {
  /// The ordinary Criterion groups (`setup`, `prove_e2e`, `verify`).
  Normal,
  /// The labelled proof-size block, then exit; requires `H = 10`.
  ProofSize,
}

impl SpartanBenchMode {
  /// Stable lowercase name for JSON/IDs.
  pub fn name(&self) -> &'static str {
    match self {
      SpartanBenchMode::Normal => "normal",
      SpartanBenchMode::ProofSize => "proof_size",
    }
  }
}

/// Execution role derived by the pre-run protocol (§6): proposal and
/// checkpoint invocations terminate as explicit successful states rather
/// than masquerading as incomplete timing runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionRole {
  /// `H = 10`, ceilings unset, no acknowledgment: emit the resource
  /// proposal and terminate (nonpublishable).
  ResourceProposal,
  /// `H = 10`, ceilings unset, matching acknowledgment: run the one
  /// diagnostic preflight proof and terminate (nonpublishable).
  ResourceCheckpoint,
  /// Ordinary Criterion timing run.
  Timing,
  /// Proof-size run.
  ProofSize,
}

impl ExecutionRole {
  /// Stable lowercase name for JSON/manifests.
  pub fn name(&self) -> &'static str {
    match self {
      ExecutionRole::ResourceProposal => "resource_proposal",
      ExecutionRole::ResourceCheckpoint => "resource_checkpoint",
      ExecutionRole::Timing => "timing",
      ExecutionRole::ProofSize => "proof_size",
    }
  }
}

/// Stage-1 hard-safety precheck (§4): a conservative padded-domain/RSS
/// upper bound from `H` and the pinned pilot counts, under the versioned
/// policy formula. Checked arithmetic throughout; runs BEFORE any
/// potentially large allocation and is never bypassed.
pub fn hard_safety_precheck(hashes_per_field: usize) -> Result<(), SpartanError> {
  let err = |reason: String| SpartanError::InvalidInputLength { reason };
  let est_real = hashes_per_field
    .checked_mul(3)
    .and_then(|p| p.checked_mul(PILOT_PERM_CONS_UPPER))
    .and_then(|c| c.checked_mul(EST_SLACK_NUM))
    .map(|c| c / EST_SLACK_DEN)
    .ok_or_else(|| err("resource precheck: estimate arithmetic overflow".to_string()))?;
  let est_padded = est_real
    .checked_next_power_of_two()
    .ok_or_else(|| err("resource precheck: padded estimate overflow".to_string()))?;
  if est_padded > HARD_MAX_PADDED_DOMAIN {
    return Err(err(format!(
      "resource precheck (policy v{RESOURCE_POLICY_VERSION}): estimated padded domain \
       {est_padded} exceeds the hard safety ceiling {HARD_MAX_PADDED_DOMAIN}"
    )));
  }
  // Conservative RSS model: ~12 domain-sized 32-byte scalar vectors (z,
  // witness, scratch, sumcheck state) plus ~6 sparse-matrix entries per
  // real row at ~24 bytes across A/B/C and their precomputations.
  let vectors = (est_padded as u64)
    .checked_mul(32 * 12)
    .ok_or_else(|| err("resource precheck: RSS arithmetic overflow".to_string()))?;
  let matrices = (est_real as u64)
    .checked_mul(6 * 24)
    .ok_or_else(|| err("resource precheck: RSS arithmetic overflow".to_string()))?;
  let est_rss = vectors
    .checked_add(matrices)
    .ok_or_else(|| err("resource precheck: RSS arithmetic overflow".to_string()))?;
  if est_rss > HARD_MAX_EST_RSS_BYTES {
    return Err(err(format!(
      "resource precheck (policy v{RESOURCE_POLICY_VERSION}): estimated peak RSS \
       {est_rss} B exceeds the hard safety ceiling {HARD_MAX_EST_RSS_BYTES} B"
    )));
  }
  Ok(())
}

/// A parsed, syntax/semantics-validated benchmark-run request. Pure result
/// of [`SpartanRunRequest::parse`]; carries NO shape-derived data —
/// benchmark IDs, manifests, and admission decisions must come from the
/// resolved config, never from this request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanRunRequest {
  /// Resolved mode.
  pub mode: SpartanBenchMode,
  /// Requested chain length `H` per field.
  pub hashes: usize,
  /// Total compressions in the combined circuit: `3H`.
  pub total_hashes: usize,
  /// `POSEIDON_ALLOW_LARGE=1`.
  pub allow_large: bool,
  /// `POSEIDON_ALLOW_DIRTY=1`.
  pub allow_dirty: bool,
  /// `POSEIDON_RESOURCE_ACK=<sha256 hex>`, if present.
  pub resource_ack: Option<String>,
  /// The environment-side half of the §6 `canonical_common` predicate
  /// (`H`, overrides, ack absence, `RAYON_NUM_THREADS`, `RUSTFLAGS`,
  /// knob-freedom). Tree cleanliness and artifact validation are judged by
  /// the config binary, runner, and publisher.
  pub canonical_env_common: bool,
}

fn cfg_err(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: format!("poseidon spartan bench config: {}", reason.into()),
  }
}

/// Flags whose PRESENCE is an error in this suite: there are no backends,
/// no `k`, no sweep, and no permitted repo knobs to configure.
pub const FORBIDDEN_FLAGS: [&str; 5] = ["BDPCS", "IMOD_K", "BDK", "KSWEEP", "POSEIDON_ALLOW_KNOBS"];

impl SpartanRunRequest {
  /// Parse a benchmark-run request from an environment map. Pure: consults
  /// nothing but `env`. Conflicts are errors, never silently ignored; the
  /// presence of any [`FORBIDDEN_FLAGS`] entry or any of the seven ModP
  /// repo knobs (including `RUST_LOG`) is an error regardless of value.
  /// `POSEIDON_ALLOW_DIRTY` changes only the runner's clean-tree policy;
  /// it never permits logging or tuning knobs.
  pub fn parse(env: &BTreeMap<OsString, OsString>) -> Result<Self, SpartanError> {
    for key in FORBIDDEN_FLAGS {
      if is_present(env, key) {
        return Err(cfg_err(format!(
          "{key} is not a flag of the classic-Spartan suite; its presence is an error"
        )));
      }
    }
    for key in HIDDEN_KNOBS {
      if is_present(env, key) {
        return Err(cfg_err(format!(
          "{key} is set; this suite permits no repo knobs (RUST_LOG included — \
           span logging inside prove is not free)"
        )));
      }
    }

    let psize = parse_bool(env, "PSIZE")?;
    let allow_large = parse_bool(env, "POSEIDON_ALLOW_LARGE")?;
    let allow_dirty = parse_bool(env, "POSEIDON_ALLOW_DIRTY")?;

    let resource_ack = match get_unicode(env, "POSEIDON_RESOURCE_ACK")? {
      None => None,
      Some(s) => {
        if s.len() != 64
          || !s
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
          return Err(cfg_err(
            "POSEIDON_RESOURCE_ACK must be a 64-character lowercase hex SHA-256",
          ));
        }
        Some(s)
      }
    };

    let hashes = parse_usize(env, "HASHES")?.unwrap_or(DEFAULT_HASHES);
    if hashes == 0 {
      return Err(cfg_err("HASHES must be at least 1"));
    }
    if hashes > SPARTAN_MAX_HASHES && !allow_large {
      return Err(cfg_err(format!(
        "HASHES = {hashes} exceeds the resource cap {SPARTAN_MAX_HASHES}; \
         POSEIDON_ALLOW_LARGE=1 admits a noncanonical run after the hard-safety \
         precheck and requested-shape synthesis"
      )));
    }
    if u32::try_from(hashes).is_err() {
      return Err(cfg_err("HASHES exceeds u32::MAX"));
    }
    let total_hashes = hashes
      .checked_mul(3)
      .ok_or_else(|| cfg_err("3H overflows usize"))?;

    let mode = if psize {
      if hashes != DEFAULT_HASHES {
        return Err(cfg_err(format!(
          "PSIZE=1 requires H = {DEFAULT_HASHES}, got {hashes}"
        )));
      }
      SpartanBenchMode::ProofSize
    } else {
      SpartanBenchMode::Normal
    };

    let canonical_env_common = hashes == DEFAULT_HASHES
      && !allow_large
      && !allow_dirty
      && resource_ack.is_none()
      && get_unicode(env, "RAYON_NUM_THREADS")?.as_deref() == Some("1")
      && get_unicode(env, "RUSTFLAGS")?.as_deref() == Some("-C target-cpu=native");

    Ok(Self {
      mode,
      hashes,
      total_hashes,
      allow_large,
      allow_dirty,
      resource_ack,
      canonical_env_common,
    })
  }

  /// Derive the §6 execution role for this request under the compiled-in
  /// ceilings and a proposal digest computed by the caller. `ack_matches`
  /// must be the result of comparing [`Self::resource_ack`] against the
  /// canonical proposal hash; a present-but-mismatched acknowledgment is
  /// an error, never a silent downgrade.
  pub fn execution_role(&self, ack_matches: Option<bool>) -> Result<ExecutionRole, SpartanError> {
    let ceilings_set = CANONICAL_CEILINGS.is_some();
    if self.hashes == DEFAULT_HASHES && !ceilings_set {
      return match (&self.resource_ack, ack_matches) {
        (None, _) => Ok(ExecutionRole::ResourceProposal),
        (Some(_), Some(true)) => Ok(ExecutionRole::ResourceCheckpoint),
        (Some(_), Some(false)) => Err(cfg_err(
          "POSEIDON_RESOURCE_ACK does not match the canonical proposal hash",
        )),
        (Some(_), None) => Err(cfg_err(
          "POSEIDON_RESOURCE_ACK is set but no proposal hash was computed",
        )),
      };
    }
    if self.resource_ack.is_some() {
      return Err(cfg_err(
        "POSEIDON_RESOURCE_ACK authorizes only the first H = 10 resource checkpoint \
         while ceilings are unset",
      ));
    }
    Ok(match self.mode {
      SpartanBenchMode::Normal => ExecutionRole::Timing,
      SpartanBenchMode::ProofSize => ExecutionRole::ProofSize,
    })
  }
}

/// An immutable resolved configuration: the request plus the frozen result
/// of exactly one actual shape synthesis. `config12` (and every benchmark
/// ID, manifest, and admission decision) derives from this object's
/// canonical bytes, never from the unresolved request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpartanResolvedConfig {
  /// The parsed request.
  pub request: SpartanRunRequest,
  /// Real (unpadded) constraint count from synthesis.
  pub real_cons: usize,
  /// Real (unpadded) witness-variable count from synthesis.
  pub real_vars: usize,
  /// Padded constraint count.
  pub padded_cons: usize,
  /// Padded witness-variable count.
  pub padded_vars: usize,
  /// Semantic workload digest (lowercase hex of the §3 descriptor).
  pub workload_digest: String,
  /// Suite-specific statement-encoding digest (lowercase hex).
  pub encoding_digest: String,
}

/// Pinned combined dimensions per `H` (measured by real synthesis, §4).
/// After a value is pinned, every subsequent resolution must match it
/// exactly; unpinned `H` values resolve without a pin comparison.
const PINNED_DIMENSIONS: [(usize, usize, usize, usize, usize); 3] = [
  (1, 935_187, 930_420, 1 << 20, 1 << 20),
  (2, 1_881_735, 1_872_162, 1 << 21, 1 << 21),
  (10, 9_454_119, 9_406_098, 1 << 24, 1 << 24),
];

/// The statement-encoding digest: a BLAKE3 hash of the public-IO layout
/// (12 limb scalars, field-major, limb-minor little-endian), limb geometry,
/// gadget revision, and reduction-schedule version.
fn encoding_digest() -> String {
  let mut hasher = blake3::Hasher::new();
  hasher.update(b"limber/poseidon2-spartan-v1/encoding");
  hasher.update(&[NUM_PUBLIC_LIMBS as u8, 4, 64]);
  hasher.update(b"field-major-limb-minor-le");
  hasher.update(EMULATED_DEP_REV.as_bytes());
  hasher.update(REDUCTION_SCHEDULE_VERSION.as_bytes());
  hex(hasher.finalize().as_bytes())
}

fn hex(bytes: &[u8]) -> String {
  bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Freeze one actual shape synthesis into an immutable resolved config
/// (§6): records real/padded dimensions, the semantic workload digest, and
/// the statement-encoding digest, and — for pinned `H` — asserts exact
/// dimension equality with the §4 table.
pub fn resolve_spartan_run<E: crate::traits::Engine>(
  request: SpartanRunRequest,
  set: &crate::poseidon2::Poseidon2ParamsSet,
  shape: &crate::r1cs::SplitR1CSShape<E>,
) -> Result<SpartanResolvedConfig, SpartanError> {
  let sizes = shape.sizes();
  let (real_cons, real_vars) = (sizes[0], sizes[1] + sizes[2] + sizes[3]);
  let (padded_cons, padded_vars) = (sizes[4], sizes[5] + sizes[6] + sizes[7]);
  if sizes[8] != NUM_PUBLIC_LIMBS || sizes[9] != 0 {
    return Err(cfg_err(format!(
      "resolved shape has num_public = {}, num_challenges = {}; expected {NUM_PUBLIC_LIMBS}, 0",
      sizes[8], sizes[9]
    )));
  }
  for (h, cons_u, rest_u, cons, rest) in PINNED_DIMENSIONS {
    if request.hashes == h
      && (real_cons, real_vars, padded_cons, padded_vars) != (cons_u, rest_u, cons, rest)
    {
      return Err(cfg_err(format!(
        "resolved H = {h} dimensions ({real_cons}, {real_vars}, {padded_cons}, {padded_vars}) \
         do not match the pinned table ({cons_u}, {rest_u}, {cons}, {rest}); \
         the schedule or dependency changed — reopen the §3 gate"
      )));
    }
  }
  let workload = super::snark::workload_digest(set)?;
  Ok(SpartanResolvedConfig {
    request,
    real_cons,
    real_vars,
    padded_cons,
    padded_vars,
    workload_digest: hex(&workload),
    encoding_digest: encoding_digest(),
  })
}

impl SpartanResolvedConfig {
  /// The canonical protocol subsection: every immutable, result-affecting
  /// resolved input. `serde_json`'s default map is BTree-backed, so
  /// serialization is key-sorted and deterministic.
  pub fn protocol_json(&self) -> serde_json::Value {
    serde_json::json!({
      "workload": "limber-poseidon2-v1",
      "suite": "spartan",
      "circuit": "mixed3-emulated",
      "proof_system": "classic-spartan",
      "engine": "T256HyraxEngine",
      "mode": self.request.mode.name(),
      "hashes_per_field": self.request.hashes,
      "total_hashes": self.request.total_hashes,
      "num_public": NUM_PUBLIC_LIMBS,
      "field_order": FIELD_ORDER.to_vec(),
      "dims": {
        "real_cons": self.real_cons,
        "real_vars": self.real_vars,
        "padded_cons": self.padded_cons,
        "padded_vars": self.padded_vars,
      },
      "workload_digest": self.workload_digest,
      "encoding_digest": self.encoding_digest,
      "emulation": {
        "gadget": "bellpepper-emulated",
        "rev": EMULATED_DEP_REV,
        "num_limbs": 4,
        "bits_per_limb": 64,
        "limb_order": "le",
        "reduction_schedule": REDUCTION_SCHEDULE_VERSION,
      },
      "resource_policy": {
        "version": RESOURCE_POLICY_VERSION,
        "hard_max_padded_domain": HARD_MAX_PADDED_DOMAIN,
        "hard_max_est_rss_bytes": HARD_MAX_EST_RSS_BYTES,
        "canonical_ceilings": CANONICAL_CEILINGS.map(|c| serde_json::json!({
          "max_padded_domain": c.max_padded_domain,
          "max_rss_bytes": c.max_rss_bytes,
          "max_projected_wall_s": c.max_projected_wall_s,
        })),
      },
      "is_small": false,
      "allow_large": self.request.allow_large,
      "allow_dirty": self.request.allow_dirty,
      "resource_ack_present": self.request.resource_ack.is_some(),
      "canonical_env_common": self.request.canonical_env_common,
      "criterion": {
        "sample_size": 10,
        "warm_up_time_s": 1,
        "measurement_time_s": 20,
        "sampling_mode": "flat",
      },
    })
  }

  /// Canonical bytes of the protocol subsection (pretty, sorted keys,
  /// trailing newline) — same convention as the ModP suite.
  pub fn protocol_canonical_bytes(&self) -> Vec<u8> {
    canonical_json_bytes(&self.protocol_json())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::ffi::OsStr;

  fn env(pairs: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
    pairs
      .iter()
      .map(|(k, v)| (OsStr::new(k).to_os_string(), OsStr::new(v).to_os_string()))
      .collect()
  }

  const CANON_ENV: [(&str, &str); 2] = [
    ("RAYON_NUM_THREADS", "1"),
    ("RUSTFLAGS", "-C target-cpu=native"),
  ];

  #[test]
  fn defaults_parse_and_env_common_requires_thread_pins() {
    let cfg = SpartanRunRequest::parse(&env(&[])).unwrap();
    assert_eq!(cfg.mode, SpartanBenchMode::Normal);
    assert_eq!(cfg.hashes, 10);
    assert_eq!(cfg.total_hashes, 30);
    // canonical_common requires the pinned environment protocol too.
    assert!(!cfg.canonical_env_common);
    let cfg = SpartanRunRequest::parse(&env(&CANON_ENV)).unwrap();
    assert!(cfg.canonical_env_common);
  }

  #[test]
  fn hashes_grammar_and_cap() {
    assert_eq!(
      SpartanRunRequest::parse(&env(&[("HASHES", "1")]))
        .unwrap()
        .hashes,
      1
    );
    assert!(SpartanRunRequest::parse(&env(&[("HASHES", "0")])).is_err());
    assert!(SpartanRunRequest::parse(&env(&[("HASHES", "11")])).is_err());
    assert!(SpartanRunRequest::parse(&env(&[("HASHES", "x")])).is_err());
    let cfg =
      SpartanRunRequest::parse(&env(&[("HASHES", "11"), ("POSEIDON_ALLOW_LARGE", "1")])).unwrap();
    assert_eq!(cfg.hashes, 11);
    assert!(!cfg.canonical_env_common);
  }

  #[test]
  fn strict_booleans_and_ack_grammar() {
    for bad in ["2", "true", "yes", ""] {
      assert!(
        SpartanRunRequest::parse(&env(&[("PSIZE", bad)])).is_err(),
        "{bad:?}"
      );
    }
    for bad in ["", "abc", "ABCDEF", &"a".repeat(63), &"g".repeat(64)] {
      assert!(
        SpartanRunRequest::parse(&env(&[("POSEIDON_RESOURCE_ACK", bad)])).is_err(),
        "{bad:?}"
      );
    }
    let ok = "a".repeat(64);
    let cfg = SpartanRunRequest::parse(&env(&[("POSEIDON_RESOURCE_ACK", &ok)])).unwrap();
    assert_eq!(cfg.resource_ack.as_deref(), Some(ok.as_str()));
  }

  #[test]
  fn psize_requires_canonical_h() {
    let cfg = SpartanRunRequest::parse(&env(&[("PSIZE", "1")])).unwrap();
    assert_eq!(cfg.mode, SpartanBenchMode::ProofSize);
    assert!(SpartanRunRequest::parse(&env(&[("PSIZE", "1"), ("HASHES", "2")])).is_err());
  }

  #[test]
  fn forbidden_flags_and_knobs_error_on_presence() {
    for key in FORBIDDEN_FLAGS {
      assert!(
        SpartanRunRequest::parse(&env(&[(key, "0")])).is_err(),
        "{key} presence must be rejected"
      );
    }
    for key in HIDDEN_KNOBS {
      assert!(
        SpartanRunRequest::parse(&env(&[(key, "1")])).is_err(),
        "{key} presence must be rejected"
      );
    }
    assert!(
      SpartanRunRequest::parse(&env(&[("POSEIDON_ALLOW_DIRTY", "1"), ("RUST_LOG", "info")]))
        .is_err()
    );
  }

  #[test]
  fn execution_roles_fail_closed_while_ceilings_unset() {
    assert!(
      CANONICAL_CEILINGS.is_none(),
      "test written for unset ceilings"
    );
    // H = 10 without acknowledgment: proposal role, regardless of mode.
    let cfg = SpartanRunRequest::parse(&env(&CANON_ENV)).unwrap();
    assert_eq!(
      cfg.execution_role(None).unwrap(),
      ExecutionRole::ResourceProposal
    );
    // Matching acknowledgment: checkpoint; mismatched: error.
    let ack = "b".repeat(64);
    let cfg = SpartanRunRequest::parse(&env(&[("POSEIDON_RESOURCE_ACK", &ack)])).unwrap();
    assert_eq!(
      cfg.execution_role(Some(true)).unwrap(),
      ExecutionRole::ResourceCheckpoint
    );
    assert!(cfg.execution_role(Some(false)).is_err());
    assert!(cfg.execution_role(None).is_err());
    // Non-canonical H: ordinary roles, and an acknowledgment is an error.
    let cfg = SpartanRunRequest::parse(&env(&[("HASHES", "1")])).unwrap();
    assert_eq!(cfg.execution_role(None).unwrap(), ExecutionRole::Timing);
    let cfg =
      SpartanRunRequest::parse(&env(&[("HASHES", "1"), ("POSEIDON_RESOURCE_ACK", &ack)])).unwrap();
    assert!(cfg.execution_role(None).is_err());
  }

  #[test]
  fn hard_safety_precheck_policy() {
    // The canonical workload passes with wide margin.
    hard_safety_precheck(10).unwrap();
    // The 2^26-domain ceiling binds: ~57 pilot-permutation triples exceed it.
    assert!(hard_safety_precheck(60).is_err());
    // Absurd requests fail through checked arithmetic, not overflow.
    assert!(hard_safety_precheck(usize::MAX / 2).is_err());
  }

  #[test]
  fn resolver_freezes_dimensions_and_digests() {
    use crate::bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS};
    use crate::poseidon2::build_all_params;
    use crate::poseidon2_spartan::circuit::build_circuit;
    use crate::provider::T256HyraxEngine;

    let set = build_all_params().unwrap();
    let circuit = build_circuit::<T256HyraxEngine>(&set, 1).unwrap();
    let shape = ShapeCS::r1cs_shape(&circuit).unwrap();
    let request = SpartanRunRequest::parse(&env(&[("HASHES", "1")])).unwrap();
    let resolved = resolve_spartan_run(request, &set, &shape).unwrap();
    assert_eq!(resolved.real_cons, 935_187);
    assert_eq!(resolved.padded_cons, 1 << 20);
    assert_eq!(resolved.workload_digest.len(), 64);
    assert_eq!(resolved.encoding_digest.len(), 64);
    let b1 = resolved.protocol_canonical_bytes();
    assert_eq!(b1, resolved.protocol_canonical_bytes());
    let text = String::from_utf8(b1).unwrap();
    assert!(text.contains(EMULATED_DEP_REV));
    assert!(text.contains(&resolved.workload_digest));
    assert!(text.ends_with('\n'));
  }
}
