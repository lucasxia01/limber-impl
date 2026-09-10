//! Typed setup and verification for the Poseidon2 classic-Spartan baseline
//! (`plan/poseidon_spartan_bench.md` §3, §5): a Poseidon-specific verifier
//! key that integrity-binds the semantic workload metadata (chain length,
//! ordered trusted moduli, workload descriptor, recorded key sizes) to the
//! raw Spartan key, and the only supported verification path, which
//! recomposes the 12 public limb scalars into three canonical digests.

use super::circuit::{NUM_PUBLIC_LIMBS, Poseidon2SpartanCircuit};
use super::emulated::{BITS_PER_LIMB, NUM_LIMBS};
use crate::errors::SpartanError;
use crate::poseidon2::{FIELD_ORDER, Poseidon2ParamsSet, chain_iv};
use crate::spartan::{SpartanProverKey, SpartanSNARK, SpartanVerifierKey};
use crate::traits::{
  Engine,
  snark::{DigestHelperTrait, R1CSSNARKTrait, SpartanDigest},
};
use ff::PrimeField;
use num_bigint::BigUint;

/// Number of field blocks (and public digests).
const NUM_FIELDS: usize = 3;

/// Domain separator for the workload descriptor hash.
const WORKLOAD_DOMAIN: &[u8] = b"limber/poseidon2-spartan-v1/workload";

/// Domain separator for the verifier-key binding hash.
const BINDING_DOMAIN: &[u8] = b"limber/poseidon2-spartan-v1/vk";

/// Poseidon-specific verifier key: the raw Spartan key plus the semantic
/// metadata needed to interpret a proof's public values, integrity-bound by
/// [`binding digest`](Self::binding_digest) so the label cannot be detached
/// from the key. Fields and constructor are private; the only builder is
/// [`setup_poseidon_spartan`], and there is no unvalidated `Deserialize`.
pub struct PoseidonSpartanVerifierKey<E: Engine> {
  /// The raw Spartan verifier key.
  inner: SpartanVerifierKey<E>,
  /// Chain length per field block.
  hashes_per_field: u32,
  /// Ordered trusted moduli, in [`FIELD_ORDER`].
  moduli: [BigUint; NUM_FIELDS],
  /// Digest of `inner` recorded at setup.
  inner_digest: SpartanDigest,
  /// Recorded raw-key sizes (`SpartanProverKey::sizes` layout).
  sizes: [usize; 10],
  /// Hash of the complete workload description (§3).
  workload_digest: [u8; 32],
  /// Binding of all of the above under [`BINDING_DOMAIN`].
  binding_digest: [u8; 32],
}

impl<E: Engine> PoseidonSpartanVerifierKey<E> {
  /// The ordered trusted moduli, in [`FIELD_ORDER`].
  pub fn moduli(&self) -> &[BigUint; NUM_FIELDS] {
    &self.moduli
  }

  /// Chain length per field block.
  pub fn hashes_per_field(&self) -> u32 {
    self.hashes_per_field
  }
}

/// A big-endian, left-zero-padded 32-byte encoding of a `< 2^256` value.
fn be32(v: &BigUint) -> Result<[u8; 32], SpartanError> {
  let bytes = v.to_bytes_be();
  if bytes.len() > 32 {
    return Err(SpartanError::InvalidInputLength {
      reason: "poseidon2_spartan: value exceeds 32 bytes".to_string(),
    });
  }
  let mut out = [0u8; 32];
  out[32 - bytes.len()..].copy_from_slice(&bytes);
  Ok(out)
}

/// Hash the complete workload description: version domain, field order and
/// moduli, both matrices, all round constants, the chain IV, and the
/// public-limb layout. Every non-byte field uses a fixed-order,
/// length-delimited canonical encoding. Crate-side: the run resolver
/// freezes this same digest into the resolved config.
pub(crate) fn workload_digest(set: &Poseidon2ParamsSet) -> Result<[u8; 32], SpartanError> {
  let mut hasher = blake3::Hasher::new();
  hasher.update(WORKLOAD_DOMAIN);
  hasher.update(&[NUM_FIELDS as u8]);
  for field in FIELD_ORDER {
    let params = set.get(field);
    let name = field.name().as_bytes();
    hasher.update(&(name.len() as u32).to_be_bytes());
    hasher.update(name);
    hasher.update(&be32(params.modulus())?);
    for m in [params.m_e(), params.m_i()] {
      for row in m {
        for entry in row {
          hasher.update(&entry.to_be_bytes());
        }
      }
    }
    let rc = params.round_constants();
    hasher.update(&(rc.len() as u32).to_be_bytes());
    for round in rc {
      hasher.update(&[round.len() as u8]);
      for c in round {
        hasher.update(&be32(c)?);
      }
    }
  }
  hasher.update(&be32(&chain_iv())?);
  hasher.update(&[NUM_LIMBS as u8, BITS_PER_LIMB as u8]);
  hasher.update(b"le");
  Ok(*hasher.finalize().as_bytes())
}

/// The verifier-key binding hash over the recorded metadata (§3): inner key
/// digest, chain length, ordered moduli, workload digest, and sizes (each
/// `usize` as a checked `u64`).
fn binding_digest(
  inner_digest: &SpartanDigest,
  hashes_per_field: u32,
  moduli: &[BigUint; NUM_FIELDS],
  workload: &[u8; 32],
  sizes: &[usize; 10],
) -> Result<[u8; 32], SpartanError> {
  let mut hasher = blake3::Hasher::new();
  hasher.update(BINDING_DOMAIN);
  hasher.update(inner_digest);
  hasher.update(&u64::from(hashes_per_field).to_be_bytes());
  for m in moduli {
    hasher.update(&be32(m)?);
  }
  hasher.update(workload);
  for s in sizes {
    let s64 = u64::try_from(*s).map_err(|_| SpartanError::InvalidInputLength {
      reason: "poseidon2_spartan: recorded size exceeds u64".to_string(),
    })?;
    hasher.update(&s64.to_be_bytes());
  }
  Ok(*hasher.finalize().as_bytes())
}

/// Indices of `num_public` and `num_challenges` in the raw-key sizes array.
const SIZES_NUM_PUBLIC: usize = 8;
/// Index of `num_challenges` in the raw-key sizes array.
const SIZES_NUM_CHALLENGES: usize = 9;

/// The only supported Poseidon setup path: extract the semantic metadata
/// from the circuit, run generic Spartan setup exactly once, validate the
/// resulting key dimensions (`num_public = 12`, `num_challenges = 0`,
/// prover/verifier sizes agree), and construct the bound typed verifier
/// key atomically.
pub fn setup_poseidon_spartan<E: Engine>(
  circuit: Poseidon2SpartanCircuit<E>,
) -> Result<(SpartanProverKey<E>, PoseidonSpartanVerifierKey<E>), SpartanError> {
  let hashes_per_field =
    u32::try_from(circuit.hashes_per_field()).map_err(|_| SpartanError::InvalidInputLength {
      reason: "poseidon2_spartan: H exceeds u32::MAX".to_string(),
    })?;
  let moduli: [BigUint; NUM_FIELDS] =
    core::array::from_fn(|f| circuit.params().get(FIELD_ORDER[f]).modulus().clone());
  let workload = workload_digest(circuit.params())?;

  let (pk, vk) = SpartanSNARK::<E>::setup(circuit)?;

  let sizes = pk.sizes();
  if sizes != vk.sizes() {
    return Err(SpartanError::ProofVerifyError {
      reason: "poseidon2_spartan setup: prover/verifier key sizes disagree".to_string(),
    });
  }
  if sizes[SIZES_NUM_PUBLIC] != NUM_PUBLIC_LIMBS {
    return Err(SpartanError::ProofVerifyError {
      reason: format!(
        "poseidon2_spartan setup: num_public = {}, expected {NUM_PUBLIC_LIMBS}",
        sizes[SIZES_NUM_PUBLIC]
      ),
    });
  }
  if sizes[SIZES_NUM_CHALLENGES] != 0 {
    return Err(SpartanError::ProofVerifyError {
      reason: format!(
        "poseidon2_spartan setup: num_challenges = {}, expected 0",
        sizes[SIZES_NUM_CHALLENGES]
      ),
    });
  }
  let inner_digest = vk.digest()?;
  let binding = binding_digest(&inner_digest, hashes_per_field, &moduli, &workload, &sizes)?;
  Ok((
    pk,
    PoseidonSpartanVerifierKey {
      inner: vk,
      hashes_per_field,
      moduli,
      inner_digest,
      sizes,
      workload_digest: workload,
      binding_digest: binding,
    },
  ))
}

/// Convert one public scalar to a `u64` limb, rejecting any canonical
/// scalar integer above `u64::MAX` without truncation.
fn scalar_to_limb<F: PrimeField>(scalar: &F) -> Result<u64, SpartanError> {
  // The `ff`-derive fields used here expose a little-endian canonical repr;
  // pinned by the `scalar_repr_is_little_endian` unit test.
  let value = BigUint::from_bytes_le(scalar.to_repr().as_ref());
  u64::try_from(&value).map_err(|_| SpartanError::ProofVerifyError {
    reason: "poseidon2_spartan verify: public scalar exceeds a 64-bit limb".to_string(),
  })
}

/// Recompose the 12 public limb scalars into three digests and enforce the
/// canonicality policy `digest_f < moduli[f]`. Separate from the proof
/// argument so the policy is directly testable.
fn check_canonical_digests<F: PrimeField>(
  scalars: &[F],
  moduli: &[BigUint; NUM_FIELDS],
) -> Result<[BigUint; NUM_FIELDS], SpartanError> {
  if scalars.len() != NUM_PUBLIC_LIMBS {
    return Err(SpartanError::ProofVerifyError {
      reason: format!(
        "poseidon2_spartan verify: {} public values, expected {NUM_PUBLIC_LIMBS}",
        scalars.len()
      ),
    });
  }
  let mut out = Vec::with_capacity(NUM_FIELDS);
  for (f, modulus) in moduli.iter().enumerate() {
    let mut digest = BigUint::default();
    for l in (0..NUM_LIMBS).rev() {
      let limb = scalar_to_limb(&scalars[NUM_LIMBS * f + l])?;
      digest = (digest << BITS_PER_LIMB) + BigUint::from(limb);
    }
    if &digest >= modulus {
      return Err(SpartanError::ProofVerifyError {
        reason: format!("poseidon2_spartan verify: digest {f} is not a canonical residue (>= p)"),
      });
    }
    out.push(digest);
  }
  match out.try_into() {
    Ok(d) => Ok(d),
    Err(_) => unreachable!("exactly three digests"),
  }
}

/// Verify with the bound key, then recompose and canonicality-check the
/// digest IO. The only supported verification path: it revalidates the
/// binding digest, the recorded inner-key digest, and the recorded sizes
/// before using any metadata, so a proof cannot be reinterpreted under
/// unrelated metadata.
pub fn verify_poseidon_spartan<E: Engine>(
  vk: &PoseidonSpartanVerifierKey<E>,
  proof: &SpartanSNARK<E>,
) -> Result<[BigUint; NUM_FIELDS], SpartanError> {
  if vk.inner.digest()? != vk.inner_digest {
    return Err(SpartanError::ProofVerifyError {
      reason: "poseidon2_spartan verify: inner verifier-key digest mismatch".to_string(),
    });
  }
  if vk.inner.sizes() != vk.sizes {
    return Err(SpartanError::ProofVerifyError {
      reason: "poseidon2_spartan verify: recorded key sizes mismatch".to_string(),
    });
  }
  if vk.sizes[SIZES_NUM_PUBLIC] != NUM_PUBLIC_LIMBS || vk.sizes[SIZES_NUM_CHALLENGES] != 0 {
    return Err(SpartanError::ProofVerifyError {
      reason: "poseidon2_spartan verify: recorded public/challenge counts are wrong".to_string(),
    });
  }
  let recomputed = binding_digest(
    &vk.inner_digest,
    vk.hashes_per_field,
    &vk.moduli,
    &vk.workload_digest,
    &vk.sizes,
  )?;
  if recomputed != vk.binding_digest {
    return Err(SpartanError::ProofVerifyError {
      reason: "poseidon2_spartan verify: verifier-key binding digest mismatch".to_string(),
    });
  }
  let public_values = proof.verify(&vk.inner)?;
  check_canonical_digests(&public_values, &vk.moduli)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::poseidon2::{build_all_params, build_inputs, expected_chain};
  use crate::provider::T256HyraxEngine;
  use crate::spartan::SpartanSNARK;
  use ff::Field as FfField;

  type E = T256HyraxEngine;
  type S = <E as Engine>::Scalar;

  #[test]
  fn scalar_repr_is_little_endian() {
    // scalar_to_limb depends on to_repr being canonical little-endian.
    let x = S::from(0x0102_0304_0506_0708u64);
    let repr = x.to_repr();
    let bytes = repr.as_ref();
    assert_eq!(
      &bytes[..8],
      &[0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01]
    );
    assert!(bytes[8..].iter().all(|b| *b == 0));
    assert_eq!(scalar_to_limb(&x).unwrap(), 0x0102_0304_0506_0708);
  }

  #[test]
  fn scalar_to_limb_rejects_wide_scalars() {
    let wide = S::from_u128(1u128 << 64);
    assert!(matches!(
      scalar_to_limb(&wide),
      Err(SpartanError::ProofVerifyError { .. })
    ));
    assert_eq!(scalar_to_limb(&S::from(u64::MAX)).unwrap(), u64::MAX);
  }

  fn limbs_of(v: &BigUint) -> [S; NUM_LIMBS] {
    let digits = v.to_u64_digits();
    core::array::from_fn(|i| S::from(digits.get(i).copied().unwrap_or(0)))
  }

  #[test]
  fn canonicality_policy_per_position() {
    let set = build_all_params().unwrap();
    let moduli: [BigUint; NUM_FIELDS] =
      core::array::from_fn(|f| set.get(FIELD_ORDER[f]).modulus().clone());

    // Wrong length is rejected.
    assert!(check_canonical_digests(&vec![S::ZERO; 11], &moduli).is_err());

    // p_f − 1 accepted, p_f and p_f + 1 rejected, independently per slot.
    for f in 0..NUM_FIELDS {
      let ok: Vec<S> = (0..NUM_FIELDS)
        .flat_map(|g| {
          let v = if g == f {
            &moduli[g] - 1u8
          } else {
            BigUint::from(1u8)
          };
          limbs_of(&v)
        })
        .collect();
      let digests = check_canonical_digests(&ok, &moduli).unwrap();
      assert_eq!(digests[f], &moduli[f] - 1u8);

      for bump in [0u8, 1u8] {
        let bad: Vec<S> = (0..NUM_FIELDS)
          .flat_map(|g| {
            let v = if g == f {
              &moduli[g] + bump
            } else {
              BigUint::from(1u8)
            };
            limbs_of(&v)
          })
          .collect();
        assert!(
          matches!(
            check_canonical_digests(&bad, &moduli),
            Err(SpartanError::ProofVerifyError { .. })
          ),
          "slot {f} accepted p + {bump}"
        );
      }
    }
  }

  #[test]
  fn workload_digest_is_stable_and_binding_varies() {
    let set = build_all_params().unwrap();
    let w1 = workload_digest(&set).unwrap();
    let w2 = workload_digest(&set).unwrap();
    assert_eq!(w1, w2);

    let moduli: [BigUint; NUM_FIELDS] =
      core::array::from_fn(|f| set.get(FIELD_ORDER[f]).modulus().clone());
    let sizes = [1usize, 2, 3, 4, 5, 6, 7, 8, NUM_PUBLIC_LIMBS, 0];
    let d = [7u8; 32];
    let b1 = binding_digest(&d, 1, &moduli, &w1, &sizes).unwrap();
    let b2 = binding_digest(&d, 2, &moduli, &w1, &sizes).unwrap();
    assert_ne!(b1, b2, "binding must depend on H");
    let mut other_sizes = sizes;
    other_sizes[0] += 1;
    let b3 = binding_digest(&d, 1, &moduli, &w1, &other_sizes).unwrap();
    assert_ne!(b1, b3, "binding must depend on sizes");
  }

  /// Canonical combined H = 10 round-trip: one real proof and typed
  /// verification at the padded 2^24 benchmark shape. Run pre-publication,
  /// never in ordinary CI:
  /// `cargo test --release round_trip_h10 -- --ignored --nocapture`.
  #[test]
  #[ignore]
  fn round_trip_h10() {
    let set = build_all_params().unwrap();
    let circuit = super::super::circuit::build_circuit::<E>(&set, 10).unwrap();
    let (pk, vk) = setup_poseidon_spartan::<E>(circuit.clone()).unwrap();
    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let (proof, _prep) = SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, false).unwrap();
    let digests = verify_poseidon_spartan(&vk, &proof).unwrap();
    let msgs = build_inputs(10).unwrap();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let chain = expected_chain(set.get(*field), &msgs).unwrap();
      assert_eq!(&digests[f], chain.last().unwrap(), "{}", field.name());
    }
    let sizes = proof.component_sizes().unwrap();
    assert_eq!(
      sizes.instance_commitments
        + sizes.protocol_challenges
        + sizes.public_values
        + sizes.sumchecks_and_claims
        + sizes.evaluation_opening,
      sizes.wire_total
    );
    println!("H=10 proof sizes: {sizes:?}");
  }

  /// Full H = 1 round-trip through the typed path, plus metadata-tamper
  /// negatives on the resulting key. A real Hyrax proof at ~2^20; run with
  /// `cargo test --release round_trip_h1 -- --ignored`.
  #[test]
  #[ignore]
  fn round_trip_h1() {
    let set = build_all_params().unwrap();
    let circuit = super::super::circuit::build_circuit::<E>(&set, 1).unwrap();
    let (pk, vk) = setup_poseidon_spartan::<E>(circuit.clone()).unwrap();

    let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).unwrap();
    let (proof, _prep) = SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, false).unwrap();

    let digests = verify_poseidon_spartan(&vk, &proof).unwrap();
    let msgs = build_inputs(1).unwrap();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let chain = expected_chain(set.get(*field), &msgs).unwrap();
      assert_eq!(&digests[f], chain.last().unwrap(), "{}", field.name());
    }

    // Proof-size accounting: the five components sum exactly to the wire
    // total, and the comparison payload excludes exactly the public values.
    let sizes = proof.component_sizes().unwrap();
    assert_eq!(
      sizes.instance_commitments
        + sizes.protocol_challenges
        + sizes.public_values
        + sizes.sumchecks_and_claims
        + sizes.evaluation_opening,
      sizes.wire_total
    );
    assert_eq!(
      sizes.comparison_payload,
      sizes.wire_total - sizes.public_values
    );
    // Measured H = 1 values, pinned (plan §6): a serialization or shape
    // change shows up as a size drift here.
    assert_eq!(sizes.instance_commitments, 16_906);
    assert_eq!(sizes.protocol_challenges, 8);
    assert_eq!(sizes.public_values, 392);
    assert_eq!(sizes.sumchecks_and_claims, 3_704);
    assert_eq!(sizes.evaluation_opening, 65_746);
    assert_eq!(sizes.wire_total, 86_756);
    println!("H=1 proof sizes: {sizes:?}");

    // Metadata tampering is rejected before any proof work.
    let mut bad = PoseidonSpartanVerifierKey {
      inner: vk.inner,
      hashes_per_field: vk.hashes_per_field,
      moduli: vk.moduli.clone(),
      inner_digest: vk.inner_digest,
      sizes: vk.sizes,
      workload_digest: vk.workload_digest,
      binding_digest: vk.binding_digest,
    };
    bad.hashes_per_field += 1;
    assert!(matches!(
      verify_poseidon_spartan(&bad, &proof),
      Err(SpartanError::ProofVerifyError { .. })
    ));
    bad.hashes_per_field -= 1;
    bad.sizes[0] += 1;
    assert!(matches!(
      verify_poseidon_spartan(&bad, &proof),
      Err(SpartanError::ProofVerifyError { .. })
    ));
    bad.sizes[0] -= 1;
    bad.binding_digest[0] ^= 1;
    assert!(matches!(
      verify_poseidon_spartan(&bad, &proof),
      Err(SpartanError::ProofVerifyError { .. })
    ));
  }
}
