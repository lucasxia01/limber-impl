//! Crate-side support for the Poseidon2 cross-system benchmark harness
//! (`benches/poseidon_modp.rs`; Zinc plan v10 §9, limber contract §§1, 4,
//! 5): the compiled identifiers, the pinned IntEval bounds and the TUNE-1
//! candidate order, the persisted default `k` (from the generated
//! `poseidon_tuned_defaults` module, falling back to the v9 default), the
//! deterministic benchmark-coins seed derivation, the canonical `H = 10`
//! circuit dimensions, and the pure environment helpers the
//! classic-Spartan suite's own run-configuration parser still shares.
//!
//! The v9 environment-driven `RunConfig` (the `BDPCS`/`KSWEEP`/`PSIZE`/
//! `HASHES`/`IMOD_K`/`BDK`/`POSEIDON_*` knobs) and the `POSEIDON_RUN_DIR`
//! handshake are gone from the ModP benchmark: every input of the v10
//! bench comes from its argv forms and the runner-written config files
//! (`scripts/NOTES-for-rust.md`). The handshake constants below remain
//! only for the classic-Spartan suite (`benches/poseidon_spartan.rs`).

use crate::{errors::SpartanError, poseidon_tuned_defaults, poseidon2};
use std::{
  collections::BTreeMap,
  ffi::{OsStr, OsString},
};

pub use crate::{
  poseidon2::ids::{BENCHMARK_COINS_DOMAIN, BENCHMARK_COINS_FRAMING, BENCHMARK_COINS_ID, K_ORDER},
  prime_sampler::{LIMBER_PRIME_SAMPLER_ID, LIMBER_PROTOCOL_WIRE_ID, LIMBER_TRANSCRIPT_ID},
};

/// `log_2(T_f)` for the Poseidon2 workload: committed values are
/// canonical residues below the ~256-bit target moduli.
pub const POSEIDON_LOG_T_F: usize = 256;
/// Limb bound (bits) for the IntEval range checks.
pub const POSEIDON_LOG_T: usize = 64;
/// The pinned workload: compressions per field (`3H = 30` in the combined
/// circuit).
pub const CANONICAL_HASHES_PER_FIELD: usize = 10;
/// The persisted v9 default IntEval `k` for both backends on the combined
/// mixed-modulus circuit (2026-09-02 combined-circuit tuning sweep, Apple
/// M2, `RAYON_NUM_THREADS=1`: tie band `{9, 11}` (Hyrax) / `{9, 10, 11}`
/// (Brakedown), winner = smallest in band). Used only while the generated
/// tuned-defaults module carries no epoch.
pub const V9_DEFAULT_K: usize = 9;
/// Inclusive candidate range of `k` (the tuning protocol's `k_range`).
pub const K_RANGE: (usize, usize) = (7, 13);

/// The circuit's semantic field-block/public-IO order (layout metadata;
/// benchmark groups have no field dimension). Must match
/// `poseidon2::FIELD_ORDER` — a unit test binds the two.
pub const FIELD_ORDER: [&str; 3] = ["bn254", "bls12_381", "secp256k1"];

/// The comparable groups (aligned cross-system boundaries), in literal
/// order.
pub const PRIMARY_GROUPS: [&str; 2] = ["prove_e2e", "verify_core"];
/// The diagnostic groups of the diagnostic child, in literal registration
/// order (`advice` is Hyrax-only: it is backend-independent and only the
/// Hyrax process registers it).
pub const DIAGNOSTIC_GROUPS: [&str; 4] = [
  "setup",
  "advice",
  "commit_witness",
  "prove_after_input_commit",
];
/// Diagnostic groups only the Hyrax process registers.
pub const HYRAX_ONLY_DIAGNOSTICS: [&str; 1] = ["advice"];
/// The backend names (the metadata `variants`), in literal order.
pub const BACKEND_NAMES: [&str; 2] = ["hyrax", "brakedown"];

/// Runner-handshake variable of the classic-Spartan suite: absolute run
/// directory. Not read by the ModP benchmark.
pub const ENV_RUN_DIR: &str = "POSEIDON_RUN_DIR";
/// Runner-handshake variable of the classic-Spartan suite: absolute path
/// of the immutable `run-config.json`. Not read by the ModP benchmark.
pub const ENV_CONFIG_PATH: &str = "POSEIDON_CONFIG_PATH";
/// Runner-handshake variable of the classic-Spartan suite: full SHA-256
/// (lowercase hex) of the config's exact bytes. Not read by the ModP
/// benchmark.
pub const ENV_CONFIG_SHA256: &str = "POSEIDON_CONFIG_SHA256";

/// The seven repository knobs that silently change Brakedown layout or
/// prover work. The v10 ModP benchmark requires all seven absent (fatal
/// otherwise); the classic-Spartan suite still parses them.
pub const HIDDEN_KNOBS: [&str; 7] = [
  "BDDIRECT",
  "BDSPEC",
  "BDROWLEN",
  "BDSPLIT",
  "CHAIN_BITS",
  "GKRSKIP",
  "RUST_LOG",
];

/// The analytical sumcheck-remainder formula of the proof-size record:
/// cubic outer rounds (3 coefficients each), quadratic inner rounds (2
/// coefficients each, `log_vars + 1` rounds), plus 6 claimed evaluations,
/// at 16 bytes per two-limb runtime-prime scalar. Analytical payload, no
/// framing.
pub const SUMCHECK_REMAINDER_FORMULA: &str = "16 * (3 * log_cons + 2 * (log_vars + 1) + 6)";

/// Commitment backend of one benchmark process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchBackend {
  /// Hyrax (curve) Mod-PCS: hiding commitments, prover coins.
  Hyrax,
  /// Brakedown (hash) Mod-PCS: non-hiding, no prover coins.
  Brakedown,
}

impl BenchBackend {
  /// Both backends in `BACKEND_NAMES` order.
  pub const ALL: [BenchBackend; 2] = [BenchBackend::Hyrax, BenchBackend::Brakedown];

  /// Stable lowercase name for JSON/IDs.
  pub fn name(self) -> &'static str {
    match self {
      BenchBackend::Hyrax => "hyrax",
      BenchBackend::Brakedown => "brakedown",
    }
  }

  /// The benchmark-coins backend tag (the tuning protocol's
  /// `backend_tags`: hyrax 0, brakedown 1).
  pub fn tag(self) -> u8 {
    match self {
      BenchBackend::Hyrax => 0,
      BenchBackend::Brakedown => 1,
    }
  }

  /// Whether this backend keeps the Brakedown retained cache (and thus
  /// the deterministic empty-cache reset policy before timed samples).
  pub fn uses_retained_cache(self) -> bool {
    matches!(self, BenchBackend::Brakedown)
  }

  /// Parse a backend name.
  pub fn parse(name: &str) -> Option<Self> {
    Self::ALL.into_iter().find(|b| b.name() == name)
  }

  /// The diagnostic groups this backend's diagnostic child registers, in
  /// literal order.
  pub fn diagnostic_groups(self) -> Vec<&'static str> {
    DIAGNOSTIC_GROUPS
      .iter()
      .copied()
      .filter(|g| self == BenchBackend::Hyrax || !HYRAX_ONLY_DIAGNOSTICS.contains(g))
      .collect()
  }
}

/// Whether `k` is a TUNE-1 candidate (`K_RANGE`, equivalently a member of
/// `K_ORDER`).
pub fn is_candidate_k(k: usize) -> bool {
  (K_RANGE.0..=K_RANGE.1).contains(&k)
}

/// Zero-based position of `k` in the pinned candidate order `K_ORDER`
/// (the benchmark-coins `candidate_index`).
pub fn candidate_index(k: usize) -> Option<u32> {
  K_ORDER
    .iter()
    .position(|c| *c == k)
    .map(|i| u32::try_from(i).expect("seven candidates"))
}

/// The compiled tuned default of `backend`, `(k, tuning_id)`, from the
/// generated `poseidon_tuned_defaults` module; `None` before the first
/// tuning epoch.
pub fn tuned_default(backend: BenchBackend) -> Option<(usize, &'static str)> {
  poseidon_tuned_defaults::TUNED_DEFAULTS
    .iter()
    .find(|(name, ..)| *name == backend.name())
    .map(|(_, k, tuning_id)| (*k, *tuning_id))
}

/// The default `k` of `backend`: the compiled tuned default, else the
/// persisted v9 default [`V9_DEFAULT_K`].
pub fn default_k(backend: BenchBackend) -> usize {
  tuned_default(backend).map_or(V9_DEFAULT_K, |(k, _)| k)
}

/// The padded combined-circuit dimensions of `H` compressions per field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dimensions {
  /// Padded constraint rows (`2^log_cons`).
  pub num_cons: usize,
  /// Padded witness columns (`2^log_vars`).
  pub num_vars: usize,
  /// `log_2(num_cons)`.
  pub log_cons: usize,
  /// `log_2(num_vars)`.
  pub log_vars: usize,
  /// `log_2(max(num_cons, num_vars))`: the IntEval `log_n`.
  pub log_n: usize,
}

/// The padded dimensions of the combined mixed-modulus circuit for
/// `hashes_per_field` compressions per field (checked arithmetic, no
/// shape synthesis). Independent of `k`.
pub fn poseidon_bench_dims(hashes_per_field: usize) -> Result<Dimensions, SpartanError> {
  let (_, _, _, _, num_cons, num_vars, log_n) = poseidon2::checked_dims(hashes_per_field)?;
  Ok(Dimensions {
    num_cons,
    num_vars,
    log_cons: num_cons.ilog2() as usize,
    log_vars: num_vars.ilog2() as usize,
    log_n,
  })
}

/// `dimensions_by_k` of the compiled metadata: the padded dimensions for
/// every candidate `k` in `K_RANGE` at `hashes_per_field` (the canonical
/// `H = 10` in the metadata). The dimensions do not depend on `k`; the
/// map form is the contract's shape.
pub fn dimensions_by_k(
  hashes_per_field: usize,
) -> Result<BTreeMap<usize, Dimensions>, SpartanError> {
  let dims = poseidon_bench_dims(hashes_per_field)?;
  Ok((K_RANGE.0..=K_RANGE.1).map(|k| (k, dims)).collect())
}

/// The deterministic benchmark-coins seed (`BENCHMARK_COINS_ID`): the
/// unkeyed 32-byte BLAKE3 hash of
/// `BENCHMARK_COINS_DOMAIN || backend_u8 || candidate_index_le32 || instance_le32`
/// (`BENCHMARK_COINS_FRAMING`). The benchmark seeds `ChaCha20Rng`
/// (`rand_chacha` 0.3.1) from it inside every timed prover sample.
pub fn benchmark_coins_seed(
  backend: BenchBackend,
  candidate_index: u32,
  instance: u32,
) -> [u8; 32] {
  let mut hasher = blake3::Hasher::new();
  hasher.update(BENCHMARK_COINS_DOMAIN);
  hasher.update(&[backend.tag()]);
  hasher.update(&candidate_index.to_le_bytes());
  hasher.update(&instance.to_le_bytes());
  *hasher.finalize().as_bytes()
}

/// The analytical sumcheck-remainder byte count of the proof-size record
/// ([`SUMCHECK_REMAINDER_FORMULA`]).
pub fn analytical_sumcheck_remainder_bytes(log_cons: usize, log_vars: usize) -> usize {
  16 * (3 * log_cons + 2 * (log_vars + 1) + 6)
}

pub(crate) fn cfg_err(reason: impl Into<String>) -> SpartanError {
  SpartanError::InvalidInputLength {
    reason: format!("poseidon bench config: {}", reason.into()),
  }
}

/// Fetch a recognized, result-affecting environment value. Present
/// non-Unicode values are rejected rather than platform-dependently
/// encoded.
pub(crate) fn get_unicode(
  env: &BTreeMap<OsString, OsString>,
  key: &str,
) -> Result<Option<String>, SpartanError> {
  match env.get(OsStr::new(key)) {
    None => Ok(None),
    Some(v) => v
      .to_str()
      .map(|s| Some(s.to_string()))
      .ok_or_else(|| cfg_err(format!("{key} is not valid Unicode"))),
  }
}

/// Strict boolean flag: exactly `"0"` or `"1"`; anything else errors.
pub(crate) fn parse_bool(
  env: &BTreeMap<OsString, OsString>,
  key: &str,
) -> Result<bool, SpartanError> {
  match get_unicode(env, key)?.as_deref() {
    None => Ok(false),
    Some("0") => Ok(false),
    Some("1") => Ok(true),
    Some(other) => Err(cfg_err(format!(
      "{key} must be exactly 0 or 1, got {other:?}"
    ))),
  }
}

/// Numeric flag: `usize`, present-or-absent.
pub(crate) fn parse_usize(
  env: &BTreeMap<OsString, OsString>,
  key: &str,
) -> Result<Option<usize>, SpartanError> {
  match get_unicode(env, key)? {
    None => Ok(None),
    Some(s) => s
      .parse::<usize>()
      .map(Some)
      .map_err(|_| cfg_err(format!("{key} must be a usize, got {s:?}"))),
  }
}

pub(crate) fn is_present(env: &BTreeMap<OsString, OsString>, key: &str) -> bool {
  env.contains_key(OsStr::new(key))
}

/// Canonical JSON bytes of the classic-Spartan suite's sidecars: two-space
/// pretty-printing over BTree-backed maps (sorted keys) plus one trailing
/// newline. (The v10 ModP benchmark writes compact canonical JSON of its
/// own; see `benches/poseidon_modp_support.rs`.)
pub fn canonical_json_bytes(value: &serde_json::Value) -> Vec<u8> {
  let mut bytes =
    serde_json::to_vec_pretty(value).expect("serde_json::Value serialization cannot fail");
  bytes.push(b'\n');
  bytes
}

/// Extract and canonicalize the `protocol` subsection of a full
/// run-config JSON document of the classic-Spartan suite (its benchmark
/// byte-compares this against its own re-parse of the environment).
pub fn protocol_bytes_from_full_config(full: &str) -> Result<Vec<u8>, SpartanError> {
  let doc: serde_json::Value = serde_json::from_str(full)
    .map_err(|e| cfg_err(format!("run-config is not valid JSON: {e}")))?;
  let protocol = doc
    .get("protocol")
    .ok_or_else(|| cfg_err("run-config has no protocol subsection"))?;
  Ok(canonical_json_bytes(protocol))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn env(pairs: &[(&str, &str)]) -> BTreeMap<OsString, OsString> {
    pairs
      .iter()
      .map(|(k, v)| (OsString::from(k), OsString::from(v)))
      .collect()
  }

  #[test]
  fn k_order_is_a_permutation_of_7_to_13() {
    // The pinned candidate order is a duplicate-free subset of the admissible
    // range (the single-candidate epoch pins exactly `[9]`; a full sweep would
    // pin a permutation of 7..=13).
    let mut sorted = K_ORDER.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), K_ORDER.len());
    assert!(!K_ORDER.is_empty());
    assert!(sorted.iter().all(|k| (K_RANGE.0..=K_RANGE.1).contains(k)));
    for (i, k) in K_ORDER.iter().enumerate() {
      assert!(is_candidate_k(*k));
      assert_eq!(candidate_index(*k), Some(i as u32));
    }
    assert_eq!(candidate_index(6), None);
    assert_eq!(candidate_index(14), None);
  }

  #[test]
  fn field_order_matches_the_circuit_module() {
    let circuit: Vec<&str> = crate::poseidon2::FIELD_ORDER
      .iter()
      .map(|f| f.name())
      .collect();
    assert_eq!(FIELD_ORDER.to_vec(), circuit);
  }

  #[test]
  fn backends_tags_and_groups() {
    assert_eq!(BenchBackend::Hyrax.tag(), 0);
    assert_eq!(BenchBackend::Brakedown.tag(), 1);
    assert_eq!(BenchBackend::ALL.map(BenchBackend::name), BACKEND_NAMES);
    assert_eq!(BenchBackend::parse("hyrax"), Some(BenchBackend::Hyrax));
    assert_eq!(BenchBackend::parse("Hyrax"), None);
    assert_eq!(
      BenchBackend::Hyrax.diagnostic_groups(),
      DIAGNOSTIC_GROUPS.to_vec()
    );
    assert_eq!(
      BenchBackend::Brakedown.diagnostic_groups(),
      vec!["setup", "commit_witness", "prove_after_input_commit"]
    );
    assert!(!BenchBackend::Hyrax.uses_retained_cache());
    assert!(BenchBackend::Brakedown.uses_retained_cache());
  }

  #[test]
  fn default_k_follows_the_generated_module() {
    for backend in BenchBackend::ALL {
      match tuned_default(backend) {
        None => assert_eq!(default_k(backend), V9_DEFAULT_K),
        Some((k, id)) => {
          assert_eq!(default_k(backend), k);
          assert!(is_candidate_k(k));
          assert_eq!(id.len(), 64);
        }
      }
    }
  }

  #[test]
  fn canonical_dimensions() {
    // 1299H rows, 1302H − 3 columns, one 2^14 × 2^14 padded domain at H = 10.
    let d = poseidon_bench_dims(CANONICAL_HASHES_PER_FIELD).unwrap();
    assert_eq!(
      (d.num_cons, d.num_vars, d.log_cons, d.log_vars, d.log_n),
      (16384, 16384, 14, 14, 14)
    );
    let by_k = dimensions_by_k(CANONICAL_HASHES_PER_FIELD).unwrap();
    assert_eq!(
      by_k.keys().copied().collect::<Vec<_>>(),
      (7..=13).collect::<Vec<_>>()
    );
    assert!(by_k.values().all(|v| *v == d));
    assert!(poseidon_bench_dims(0).is_err());
    assert_eq!(
      analytical_sumcheck_remainder_bytes(14, 14),
      16 * (42 + 30 + 6)
    );
  }

  #[test]
  fn coins_seed_is_the_framed_blake3() {
    let mut framed = BENCHMARK_COINS_DOMAIN.to_vec();
    framed.push(1);
    framed.extend_from_slice(&3u32.to_le_bytes());
    framed.extend_from_slice(&9u32.to_le_bytes());
    assert_eq!(
      benchmark_coins_seed(BenchBackend::Brakedown, 3, 9),
      *blake3::hash(&framed).as_bytes()
    );
    assert_ne!(
      benchmark_coins_seed(BenchBackend::Hyrax, 0, 0),
      benchmark_coins_seed(BenchBackend::Brakedown, 0, 0)
    );
    assert_ne!(
      benchmark_coins_seed(BenchBackend::Hyrax, 0, 0),
      benchmark_coins_seed(BenchBackend::Hyrax, 0, 1)
    );
  }

  #[test]
  fn env_helpers_are_strict() {
    assert!(!parse_bool(&env(&[]), "X").unwrap());
    assert!(parse_bool(&env(&[("X", "1")]), "X").unwrap());
    assert!(!parse_bool(&env(&[("X", "0")]), "X").unwrap());
    for bad in ["true", "yes", "2", "", " 1"] {
      assert!(parse_bool(&env(&[("X", bad)]), "X").is_err(), "{bad:?}");
    }
    assert_eq!(parse_usize(&env(&[("H", "10")]), "H").unwrap(), Some(10));
    assert!(parse_usize(&env(&[("H", "abc")]), "H").is_err());
    assert!(is_present(&env(&[("BDSPLIT", "0")]), "BDSPLIT"));
    #[cfg(unix)]
    {
      use std::os::unix::ffi::OsStringExt;
      let mut e = BTreeMap::new();
      e.insert(
        OsString::from("HASHES"),
        OsString::from_vec(vec![0x66, 0xff, 0xfe]),
      );
      assert!(get_unicode(&e, "HASHES").is_err());
    }
  }

  #[test]
  fn protocol_bytes_are_extractable() {
    let full = serde_json::json!({ "protocol": { "b": 1, "a": [1, 2] }, "environment": {} });
    let extracted =
      protocol_bytes_from_full_config(&serde_json::to_string(&full).unwrap()).unwrap();
    assert_eq!(extracted, canonical_json_bytes(&full["protocol"]));
    assert!(protocol_bytes_from_full_config("{}").is_err());
  }
}
