//! The combined three-field Poseidon2 chain circuit for classic Spartan:
//! three independent emulated-field blocks in [`FIELD_ORDER`], each proving
//! `H` fixed-IV chain compressions and linking its terminal state to four
//! public 64-bit digest limbs (`plan/poseidon_spartan_bench.md` §3, §5).

use super::emulated::{
  Bls12381FrParams, Bn254FrParams, Secp256k1FrParams, alloc_private_checked,
  alloc_public_digest_checked, biguint_to_bigint, le_u64_limbs, synthesize_permutation,
};
use crate::errors::SpartanError;
use crate::poseidon2::{
  FIELD_ORDER, Field, Poseidon2Params, Poseidon2ParamsSet, build_inputs, chain_iv, expected_chain,
};
use crate::traits::{Engine, circuit::SpartanCircuit};
use bellpepper_core::{ConstraintSystem, SynthesisError, num::AllocatedNum};
use bellpepper_emulated::field_element::{EmulatedFieldElement, EmulatedFieldParams};
use ff::PrimeFieldBits;
use num_bigint::{BigInt, BigUint};
use std::marker::PhantomData;
use std::sync::Arc;

/// Number of field blocks (and public digests).
const NUM_FIELDS: usize = 3;

/// Number of public limb scalars: three digests × four limbs.
pub const NUM_PUBLIC_LIMBS: usize = 12;

/// One combined circuit: three independent emulated-field chain blocks in
/// [`FIELD_ORDER`]. `shared` and `precommitted` are empty and there are no
/// verifier challenges; every witness variable is "rest" witness, so the
/// commitment cost lands inside `prove` as the plan's timing model
/// requires. Construct via [`build_circuit`]; fields stay private so a
/// mislabeled parameter set or digest cannot be smuggled in.
#[derive(Clone)]
pub struct Poseidon2SpartanCircuit<E: Engine> {
  /// Fixed-order parameters for the three blocks.
  params: Arc<Poseidon2ParamsSet>,
  /// Chain length per field block.
  hashes_per_field: usize,
  /// Per-block message vectors, in [`FIELD_ORDER`]; the public builder
  /// duplicates one pinned vector, the test-only core constructor may not.
  messages: Arc<[Vec<BigUint>; NUM_FIELDS]>,
  /// Claimed canonical digests, in [`FIELD_ORDER`].
  digests: Arc<[BigUint; NUM_FIELDS]>,
  /// Binds the circuit to its engine.
  _marker: PhantomData<E>,
}

impl<E: Engine> Poseidon2SpartanCircuit<E> {
  /// Chain length per field block.
  pub fn hashes_per_field(&self) -> usize {
    self.hashes_per_field
  }

  /// The fixed-order parameter set the circuit was built with (crate-side:
  /// the typed setup extracts trusted moduli and the workload descriptor
  /// from it before generic setup consumes the circuit).
  pub(crate) fn params(&self) -> &Poseidon2ParamsSet {
    &self.params
  }

  /// The claimed canonical digests, in [`FIELD_ORDER`].
  pub fn digests(&self) -> &[BigUint; NUM_FIELDS] {
    &self.digests
  }
}

/// Semantic validity checks shared by both constructors: `H` bounds,
/// checked `3H`, message-vector lengths, four-limb message representatives,
/// and canonical claimed digests.
fn validate_workload(
  set: &Poseidon2ParamsSet,
  hashes_per_field: usize,
  messages: &[Vec<BigUint>; NUM_FIELDS],
  digests: &[BigUint; NUM_FIELDS],
) -> Result<(), SpartanError> {
  if hashes_per_field == 0 {
    return Err(SpartanError::InvalidInputLength {
      reason: "poseidon2_spartan: per-field hash count H must be at least 1".to_string(),
    });
  }
  if u32::try_from(hashes_per_field).is_err() {
    return Err(SpartanError::InvalidInputLength {
      reason: format!(
        "poseidon2_spartan: per-field hash count H = {hashes_per_field} exceeds u32::MAX"
      ),
    });
  }
  hashes_per_field
    .checked_mul(NUM_FIELDS)
    .ok_or_else(|| SpartanError::InvalidInputLength {
      reason: format!("poseidon2_spartan: 3H overflows for H = {hashes_per_field}"),
    })?;
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let p = set.get(*field).modulus();
    if messages[f].len() != hashes_per_field {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "poseidon2_spartan: block {} has {} messages, expected {hashes_per_field}",
          field.name(),
          messages[f].len()
        ),
      });
    }
    // Private messages are four-limb residue representatives, exactly as
    // private variables in the ModP relation: the validator enforces only
    // `message < 2^256` and must NOT reject a message merely for being
    // `≥ p_f` (§3). The benchmark builder's fixture messages are canonical
    // by construction; the test-only constructor admits any representative.
    for (j, m) in messages[f].iter().enumerate() {
      if m.bits() > 256 {
        return Err(SpartanError::InvalidInputLength {
          reason: format!(
            "poseidon2_spartan: block {} message {} does not fit four 64-bit limbs",
            field.name(),
            j + 1
          ),
        });
      }
    }
    if &digests[f] >= p {
      return Err(SpartanError::InvalidInputLength {
        reason: format!(
          "poseidon2_spartan: block {} claimed digest is not a canonical residue (>= p)",
          field.name()
        ),
      });
    }
  }
  Ok(())
}

/// Test-only core constructor: independent per-block message vectors and
/// explicit claimed digests (which need not be the true chain outputs — a
/// wrong-but-canonical digest yields an unsatisfiable circuit, which the
/// negative tests rely on). Applies the same semantic validation as the
/// public builder.
pub(crate) fn build_circuit_core<E: Engine>(
  set: &Poseidon2ParamsSet,
  hashes_per_field: usize,
  messages: [Vec<BigUint>; NUM_FIELDS],
  digests: [BigUint; NUM_FIELDS],
) -> Result<Poseidon2SpartanCircuit<E>, SpartanError> {
  validate_workload(set, hashes_per_field, &messages, &digests)?;
  Ok(Poseidon2SpartanCircuit {
    params: Arc::new(set.clone()),
    hashes_per_field,
    messages: Arc::new(messages),
    digests: Arc::new(digests),
    _marker: PhantomData,
  })
}

/// Build the benchmark circuit: the pinned `build_inputs(H)` messages are
/// copied into all three blocks and each block's claimed digest is the host
/// reference chain output, so the circuit's public values and the KAT
/// expectations come from one computation.
pub fn build_circuit<E: Engine>(
  set: &Poseidon2ParamsSet,
  hashes_per_field: usize,
) -> Result<Poseidon2SpartanCircuit<E>, SpartanError> {
  let msgs = build_inputs(hashes_per_field)?;
  let digests: [BigUint; NUM_FIELDS] = {
    let mut out = Vec::with_capacity(NUM_FIELDS);
    for field in FIELD_ORDER {
      let chain = expected_chain(set.get(field), &msgs)?;
      out.push(chain.last().expect("H >= 1 chain is nonempty").clone());
    }
    match out.try_into() {
      Ok(d) => d,
      Err(_) => unreachable!("exactly three digests"),
    }
  };
  build_circuit_core(
    set,
    hashes_per_field,
    [msgs.clone(), msgs.clone(), msgs],
    digests,
  )
}

/// Synthesize one field block: allocate `H` private messages, run the
/// chain from the fixed IV feeding each unreduced terminal lane 0 onward,
/// then link the final state to four public digest limbs.
fn synthesize_block<F, P, CS>(
  cs: &mut CS,
  params: &Poseidon2Params,
  messages: &[BigUint],
  digest: &BigUint,
) -> Result<(), SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  let zero = EmulatedFieldElement::<F, P>::from(&BigInt::from(0u8));
  let mut h = EmulatedFieldElement::<F, P>::from(&biguint_to_bigint(&chain_iv()));
  for (i, m) in messages.iter().enumerate() {
    let em =
      alloc_private_checked::<F, P, _>(&mut cs.namespace(|| format!("message {i}")), Some(m))?;
    let [h_next, _, _] = synthesize_permutation(
      &mut cs.namespace(|| format!("perm {i}")),
      params,
      [h, em, zero.clone()],
    )?;
    h = h_next;
  }
  let (pub_elt, _limbs) =
    alloc_public_digest_checked::<F, P, _>(&mut cs.namespace(|| "digest"), Some(digest))?;
  EmulatedFieldElement::assert_is_equal(&mut cs.namespace(|| "digest linkage"), &h, &pub_elt)
}

impl<E: Engine> SpartanCircuit<E> for Poseidon2SpartanCircuit<E> {
  fn public_values(&self) -> Result<Vec<E::Scalar>, SynthesisError> {
    let mut out = Vec::with_capacity(NUM_PUBLIC_LIMBS);
    for digest in self.digests.iter() {
      for limb in le_u64_limbs(digest)? {
        out.push(E::Scalar::from(limb));
      }
    }
    Ok(out)
  }

  fn shared<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<E::Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<E::Scalar>],
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<E::Scalar>],
    precommitted: &[AllocatedNum<E::Scalar>],
    challenges: Option<&[E::Scalar]>,
  ) -> Result<(), SynthesisError> {
    if !shared.is_empty() || !precommitted.is_empty() || challenges.is_some_and(|c| !c.is_empty()) {
      return Err(SynthesisError::Unsatisfiable);
    }
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let params = self.params.get(*field);
      let messages = &self.messages[f];
      let digest = &self.digests[f];
      let mut ns = cs.namespace(|| format!("block {}", field.name()));
      match field {
        Field::Bn254Fr => {
          synthesize_block::<E::Scalar, Bn254FrParams, _>(&mut ns, params, messages, digest)?
        }
        Field::Bls12381Fr => {
          synthesize_block::<E::Scalar, Bls12381FrParams, _>(&mut ns, params, messages, digest)?
        }
        Field::Secp256k1Fr => {
          synthesize_block::<E::Scalar, Secp256k1FrParams, _>(&mut ns, params, messages, digest)?
        }
      }
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::poseidon2::build_all_params;
  use crate::provider::T256HyraxEngine;
  use bellpepper_core::test_cs::TestConstraintSystem;

  type E = T256HyraxEngine;
  type S = <E as Engine>::Scalar;

  /// `TestConstraintSystem` stores a formatted path string and materialized
  /// LCs per constraint (~1 KB+ each), so a combined synthesis costs
  /// gigabytes of test-harness memory. Serialize the heavyweight tests so
  /// only one such system is alive at a time regardless of cargo's test
  /// parallelism; the real prover path does not have this profile.
  static SYNTH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

  fn synthesize_combined(circuit: &Poseidon2SpartanCircuit<E>) -> TestConstraintSystem<S> {
    let mut cs = TestConstraintSystem::<S>::new();
    circuit.synthesize(&mut cs, &[], &[], None).unwrap();
    cs
  }

  #[test]
  fn builder_semantic_bounds() {
    let set = build_all_params().unwrap();
    assert!(matches!(
      build_circuit::<E>(&set, 0),
      Err(SpartanError::InvalidInputLength { .. })
    ));
    // A message too wide for four limbs is rejected; a noncanonical residue
    // REPRESENTATIVE (`p <= m < 2^256`) is admitted (S3) - satisfiability is
    // covered by `noncanonical_message_representative_is_admitted`. A
    // noncanonical claimed digest stays rejected.
    let msgs = build_inputs(1).unwrap();
    let p_bn = set.get(Field::Bn254Fr).modulus().clone();
    let digests = [BigUint::from(1u8), BigUint::from(1u8), BigUint::from(1u8)];
    let too_wide = BigUint::from(1u8) << 256u32;
    assert!(matches!(
      build_circuit_core::<E>(
        &set,
        1,
        [vec![too_wide], vec![msgs[0].clone()], vec![msgs[0].clone()]],
        digests.clone(),
      ),
      Err(SpartanError::InvalidInputLength { .. })
    ));
    assert!(
      build_circuit_core::<E>(
        &set,
        1,
        [
          vec![p_bn.clone()],
          vec![msgs[0].clone()],
          vec![msgs[0].clone()]
        ],
        digests.clone(),
      )
      .is_ok()
    );
    assert!(matches!(
      build_circuit_core::<E>(
        &set,
        1,
        [
          vec![msgs[0].clone()],
          vec![msgs[0].clone()],
          vec![msgs[0].clone()]
        ],
        [p_bn, BigUint::from(1u8), BigUint::from(1u8)],
      ),
      Err(SpartanError::InvalidInputLength { .. })
    ));
    // Wrong message-vector length.
    assert!(matches!(
      build_circuit_core::<E>(
        &set,
        2,
        [
          vec![msgs[0].clone()],
          vec![msgs[0].clone()],
          vec![msgs[0].clone()]
        ],
        digests,
      ),
      Err(SpartanError::InvalidInputLength { .. })
    ));
  }

  #[test]
  fn combined_h1_matches_reference_and_public_order() {
    let _guard = SYNTH_LOCK.lock().unwrap();
    let set = build_all_params().unwrap();
    let circuit = build_circuit::<E>(&set, 1).unwrap();
    let cs = synthesize_combined(&circuit);
    assert!(cs.is_satisfied());
    // One constant input plus 12 public limb scalars, field-major,
    // limb-minor little-endian, matching public_values() exactly.
    assert_eq!(cs.num_inputs(), 1 + NUM_PUBLIC_LIMBS);
    let expected = SpartanCircuit::<E>::public_values(&circuit).unwrap();
    assert_eq!(cs.scalar_inputs()[1..], expected[..]);
    // Independent recomposition: digests come from expected_chain.
    let msgs = build_inputs(1).unwrap();
    for (f, field) in FIELD_ORDER.iter().enumerate() {
      let chain = expected_chain(set.get(*field), &msgs).unwrap();
      let digest = chain.last().unwrap();
      for (l, limb) in le_u64_limbs(digest).unwrap().iter().enumerate() {
        assert_eq!(
          expected[4 * f + l],
          S::from(*limb),
          "{} limb {l}",
          field.name()
        );
      }
    }
  }

  #[test]
  fn three_blocks_are_independent() {
    let _guard = SYNTH_LOCK.lock().unwrap();
    // Unequal message vectors across blocks; each block's digest is its own
    // reference chain output; changing one block's message set leaves the
    // other digests' satisfiability intact.
    let set = build_all_params().unwrap();
    let msgs = build_inputs(6).unwrap();
    let per_block: [Vec<BigUint>; 3] = [
      msgs[0..2].to_vec(),
      msgs[2..4].to_vec(),
      msgs[4..6].to_vec(),
    ];
    let digests: [BigUint; 3] = {
      let mut out = Vec::new();
      for (f, field) in FIELD_ORDER.iter().enumerate() {
        out.push(
          expected_chain(set.get(*field), &per_block[f])
            .unwrap()
            .last()
            .unwrap()
            .clone(),
        );
      }
      out.try_into().unwrap()
    };
    let circuit = build_circuit_core::<E>(&set, 2, per_block.clone(), digests.clone()).unwrap();
    let cs = synthesize_combined(&circuit);
    assert!(cs.is_satisfied());

    // Swap one block's messages without updating its digest: only that
    // block's linkage breaks (the whole system is unsatisfied), while the
    // digests of the untouched blocks remain the correct claims.
    let mut tampered = per_block;
    tampered[1] = vec![msgs[0].clone(), msgs[5].clone()];
    let bad = build_circuit_core::<E>(&set, 2, tampered, digests).unwrap();
    let cs = synthesize_combined(&bad);
    assert!(!cs.is_satisfied());
    let unsat = cs.which_is_unsatisfied().unwrap();
    assert!(
      unsat.contains("block bls12_381"),
      "unsatisfied constraint should sit in the tampered block, got: {unsat}"
    );
  }

  /// Exact combined shape dimensions, pinned from real `ShapeCS` synthesis
  /// (`plan/poseidon_spartan_bench.md` §4). A dependency bump or schedule
  /// edit that changes any dimension fails loudly here.
  #[test]
  fn combined_shape_dimensions_pinned() {
    use crate::bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS};
    let set = build_all_params().unwrap();
    // (H, real constraints, real rest vars, padded constraints, padded vars)
    for (h, cons_u, rest_u, cons, rest) in [
      (1usize, 935_187, 930_420, 1 << 20, 1 << 20),
      (2, 1_881_735, 1_872_162, 1 << 21, 1 << 21),
      (10, 9_454_119, 9_406_098, 1 << 24, 1 << 24),
    ] {
      let circuit = build_circuit::<E>(&set, h).unwrap();
      let s = ShapeCS::r1cs_shape(&circuit).unwrap();
      assert_eq!(s.num_cons_unpadded, cons_u, "H={h} real constraints");
      assert_eq!(s.num_rest_unpadded, rest_u, "H={h} real rest vars");
      assert_eq!(s.num_cons, cons, "H={h} padded constraints");
      assert_eq!(s.num_rest, rest, "H={h} padded rest vars");
      assert_eq!(s.num_shared, 0, "H={h}");
      assert_eq!(s.num_precommitted, 0, "H={h}");
      assert_eq!(s.num_public, NUM_PUBLIC_LIMBS, "H={h}");
      assert_eq!(s.num_challenges, 0, "H={h}");
    }
  }

  /// A private message may be any four-limb residue representative: adding
  /// `p_f` to a message leaves the circuit satisfiable against the digest
  /// of the canonical representative (residue semantics, §3), and the
  /// validator admits it.
  #[test]
  fn noncanonical_message_representative_is_admitted() {
    let _guard = SYNTH_LOCK.lock().unwrap();
    let set = build_all_params().unwrap();
    let msgs = build_inputs(1).unwrap();
    let digests: [BigUint; 3] = {
      let mut out = Vec::new();
      for field in FIELD_ORDER {
        out.push(
          expected_chain(set.get(field), &msgs)
            .unwrap()
            .last()
            .unwrap()
            .clone(),
        );
      }
      out.try_into().unwrap()
    };
    let p_bn = set.get(Field::Bn254Fr).modulus().clone();
    let shifted = &msgs[0] + &p_bn;
    assert!(shifted.bits() <= 256, "representative must fit four limbs");
    let circuit = build_circuit_core::<E>(
      &set,
      1,
      [vec![shifted], vec![msgs[0].clone()], vec![msgs[0].clone()]],
      digests,
    )
    .unwrap();
    let cs = synthesize_combined(&circuit);
    assert!(cs.is_satisfied());
  }

  #[test]
  fn wrong_digest_is_unsatisfied_per_block() {
    let _guard = SYNTH_LOCK.lock().unwrap();
    let set = build_all_params().unwrap();
    let msgs = build_inputs(1).unwrap();
    let honest: [BigUint; 3] = {
      let mut out = Vec::new();
      for field in FIELD_ORDER {
        out.push(
          expected_chain(set.get(field), &msgs)
            .unwrap()
            .last()
            .unwrap()
            .clone(),
        );
      }
      out.try_into().unwrap()
    };
    for f in 0..3 {
      let mut digests = honest.clone();
      let p = set.get(FIELD_ORDER[f]).modulus();
      digests[f] = (&digests[f] + 1u8) % p; // canonical but noncongruent
      let circuit =
        build_circuit_core::<E>(&set, 1, [msgs.clone(), msgs.clone(), msgs.clone()], digests)
          .unwrap();
      let cs = synthesize_combined(&circuit);
      assert!(!cs.is_satisfied(), "block {f} accepted a wrong digest");
    }
  }
}
