//! Emulated-field gadget layer over the revision-pinned `bellpepper-emulated`
//! crate: per-field [`EmulatedFieldParams`] impls and the checked-boundary
//! adapters that are the only permitted variable-allocation entry points.
//!
//! Design constraints inherited from the pinned upstream revision (see
//! `plan/poseidon_spartan_bench.md` §3):
//!
//! - allocated `check_field_membership` supports only pseudo-Mersenne moduli
//!   and must never be called for these fields;
//! - `EmulatedFieldElement::limbs` is crate-private upstream, but
//!   `new_internal_element` is public, so the adapters below allocate limbs
//!   themselves, range-check each one to [`BITS_PER_LIMB`] bits, and wrap
//!   them with overflow 0;
//! - `mul` returns an overflow-tracked convolution; reductions happen only
//!   when a later operation's overflow precondition fails;
//! - `mul_const` panics on constant-limb elements, so constant scaling is
//!   routed through [`scale_small`], which folds constants host-side.

use crate::poseidon2::{Field, Poseidon2Params, is_full_round};
use bellpepper_core::{
  ConstraintSystem, SynthesisError,
  num::{AllocatedNum, Num},
};
use bellpepper_emulated::field_element::{
  EmulatedFieldElement, EmulatedFieldParams, EmulatedLimbs,
};
use bellpepper_emulated::util::range_check_num;
use ff::PrimeFieldBits;
use num_bigint::{BigInt, BigUint, Sign};

/// Limbs per emulated 256-bit value.
pub const NUM_LIMBS: usize = 4;

/// Bits per limb. With 64-bit limbs the base multiplication precondition is
/// `2·64 + ⌈log₂ 4⌉ = 130` bits, far below the native scalar capacity.
pub const BITS_PER_LIMB: usize = 64;

/// Number of state lanes (`t = 3`).
const T: usize = 3;

/// Convert a nonnegative [`BigUint`] into the [`BigInt`] type the emulated
/// crate's parameter trait uses.
pub fn biguint_to_bigint(v: &BigUint) -> BigInt {
  BigInt::from_biguint(Sign::Plus, v.clone())
}

/// Declare one emulated-field parameter type wired to the matching
/// [`Field`] modulus, so the two sources cannot drift.
macro_rules! emulated_params {
  ($(#[$doc:meta])* $name:ident, $field:expr) => {
    $(#[$doc])*
    #[derive(Clone, Debug)]
    pub struct $name;

    impl EmulatedFieldParams for $name {
      fn num_limbs() -> usize {
        NUM_LIMBS
      }

      fn bits_per_limb() -> usize {
        BITS_PER_LIMB
      }

      fn modulus() -> BigInt {
        biguint_to_bigint(&$field.modulus())
      }
    }
  };
}

emulated_params!(
  /// Emulation parameters for the BN254 scalar field (4 × 64-bit limbs).
  Bn254FrParams,
  Field::Bn254Fr
);
emulated_params!(
  /// Emulation parameters for the BLS12-381 scalar field (4 × 64-bit limbs).
  Bls12381FrParams,
  Field::Bls12381Fr
);
emulated_params!(
  /// Emulation parameters for the secp256k1 scalar field (4 × 64-bit limbs).
  Secp256k1FrParams,
  Field::Secp256k1Fr
);

/// Little-endian 64-bit limbs of a value `< 2^256`; errors on anything
/// wider rather than truncating.
pub(crate) fn le_u64_limbs(v: &BigUint) -> Result<[u64; NUM_LIMBS], SynthesisError> {
  let digits = v.to_u64_digits();
  if digits.len() > NUM_LIMBS {
    return Err(SynthesisError::Unsatisfiable);
  }
  let mut limbs = [0u64; NUM_LIMBS];
  limbs[..digits.len()].copy_from_slice(&digits);
  Ok(limbs)
}

/// Allocate limbs (via `alloc_fn`), range-check each to [`BITS_PER_LIMB`]
/// bits, and wrap them as an overflow-zero emulated element. The manual
/// range checks are load-bearing: `new_internal_element` marks the element
/// internal, which suppresses the library's conditional width enforcement.
fn alloc_checked_impl<F, P, CS>(
  cs: &mut CS,
  value: Option<&BigUint>,
  public: bool,
) -> Result<(EmulatedFieldElement<F, P>, Vec<AllocatedNum<F>>), SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  let limbs: Option<[u64; NUM_LIMBS]> = value.map(le_u64_limbs).transpose()?;
  let mut allocated = Vec::with_capacity(NUM_LIMBS);
  let mut nums = Vec::with_capacity(NUM_LIMBS);
  for i in 0..NUM_LIMBS {
    let value_fn = || {
      limbs
        .map(|l| F::from(l[i]))
        .ok_or(SynthesisError::AssignmentMissing)
    };
    let an = if public {
      AllocatedNum::alloc_input(cs.namespace(|| format!("limb {i}")), value_fn)?
    } else {
      AllocatedNum::alloc(cs.namespace(|| format!("limb {i}")), value_fn)?
    };
    let num = Num::<F>::from(an.clone());
    range_check_num(
      &mut cs.namespace(|| format!("range check limb {i}")),
      &num,
      BITS_PER_LIMB,
    )?;
    allocated.push(an);
    nums.push(num);
  }
  Ok((
    EmulatedFieldElement::new_internal_element(EmulatedLimbs::Allocated(nums), 0),
    allocated,
  ))
}

/// Allocate a private witness value (`< 2^256`, a residue representative —
/// not required to be canonical) as four range-checked 64-bit limbs.
pub fn alloc_private_checked<F, P, CS>(
  cs: &mut CS,
  value: Option<&BigUint>,
) -> Result<EmulatedFieldElement<F, P>, SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  alloc_checked_impl(cs, value, false).map(|(e, _)| e)
}

/// Allocate a public digest as four range-checked 64-bit input limbs
/// (little-endian). Returns both the wrapped emulated element and the
/// public variables in limb order.
pub fn alloc_public_digest_checked<F, P, CS>(
  cs: &mut CS,
  value: Option<&BigUint>,
) -> Result<(EmulatedFieldElement<F, P>, Vec<AllocatedNum<F>>), SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  alloc_checked_impl(cs, value, true)
}

/// `c · x` for a pinned matrix coefficient `c ∈ {0, 1, 2, 3}` (§3
/// contract). `0` returns constant zero, `1` returns an exact clone
/// without changing overflow metadata, and `2`/`3` host-fold constant
/// operands or call upstream `mul_const` only for allocated operands —
/// never `mul_const(0)` or `mul_const(1)` (upstream also panics on
/// constant limbs). Coefficients outside the pinned set are rejected.
pub fn scale_small<F, P, CS>(
  cs: &mut CS,
  x: &EmulatedFieldElement<F, P>,
  c: u64,
) -> Result<EmulatedFieldElement<F, P>, SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  match c {
    0 => Ok(EmulatedFieldElement::from(&BigInt::from(0u8))),
    1 => Ok(x.clone()),
    2 | 3 => {
      if x.is_constant() {
        let v = BigInt::try_from(x)?;
        let r = (v * c) % P::modulus();
        Ok(EmulatedFieldElement::from(&r))
      } else {
        x.mul_const(cs, &BigInt::from(c))
      }
    }
    _ => Err(SynthesisError::Unsatisfiable),
  }
}

/// One linear-layer output lane: `Σ_j m_row[j] · state[j]`. Source lanes
/// are visited in index order `0, 1, 2`; zero-coefficient terms are
/// elided; allocated terms left-fold as limb LCs in that order; constant
/// terms fold modulo `p_f` into one accumulator added exactly once at the
/// end. Row order, namespaces, operation order, and hence the reduction
/// trajectory are pinned (§3).
fn matrix_row<F, P, CS>(
  cs: &mut CS,
  m_row: &[u64; T],
  state: &[EmulatedFieldElement<F, P>; T],
) -> Result<EmulatedFieldElement<F, P>, SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  let mut const_acc = BigInt::from(0u8);
  let mut acc: Option<EmulatedFieldElement<F, P>> = None;
  for (j, lane) in state.iter().enumerate() {
    if m_row[j] == 0 {
      continue;
    }
    if lane.is_constant() {
      const_acc += BigInt::try_from(lane)? * m_row[j];
    } else {
      let term = scale_small(
        &mut cs.namespace(|| format!("scale lane {j}")),
        lane,
        m_row[j],
      )?;
      acc = Some(match acc {
        None => term,
        Some(a) => a.add(&mut cs.namespace(|| format!("add lane {j}")), &term)?,
      });
    }
  }
  const_acc %= P::modulus();
  match acc {
    None => Ok(EmulatedFieldElement::from(&const_acc)),
    Some(a) => {
      if const_acc == BigInt::from(0u8) {
        Ok(a)
      } else {
        a.add(
          &mut cs.namespace(|| "add constant part"),
          &EmulatedFieldElement::from(&const_acc),
        )
      }
    }
  }
}

/// Apply a 3×3 linear-layer matrix to the state.
fn apply_matrix<F, P, CS>(
  cs: &mut CS,
  m: &[[u64; T]; T],
  state: &[EmulatedFieldElement<F, P>; T],
) -> Result<[EmulatedFieldElement<F, P>; T], SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  let mut out = Vec::with_capacity(T);
  for (i, row) in m.iter().enumerate() {
    out.push(matrix_row(
      &mut cs.namespace(|| format!("row {i}")),
      row,
      state,
    )?);
  }
  match out.try_into() {
    Ok(lanes) => Ok(lanes),
    Err(_) => unreachable!("exactly T rows"),
  }
}

/// `x + rc` for a round constant. Free (host-side or limb-LC fold) unless
/// the addition's overflow precondition forces a reduction first.
fn add_round_constant<F, P, CS>(
  cs: &mut CS,
  x: &EmulatedFieldElement<F, P>,
  rc: &BigUint,
) -> Result<EmulatedFieldElement<F, P>, SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  x.add(cs, &EmulatedFieldElement::from(&biguint_to_bigint(rc)))
}

/// `x^5` as three emulated multiplications (`x²`, `x⁴ = x²·x²`,
/// `x⁵ = x⁴·x`) with an explicit two-reduction schedule.
///
/// Left to its own overflow preconditions, the upstream library reduces a
/// *clone* of an over-full operand inside every `mul` call and discards it,
/// so an unreduced S-box input is re-reduced by `x²` and again by `x⁵`, and
/// `x²` is re-reduced by `x⁴` — about five discarded reductions per S-box
/// (measured: ~8.7k constraints per steady-state S-box, 818k per
/// permutation). Reducing `x` and `x²` once each and reusing them makes
/// every `mul` precondition pass outright: `x²` and `x⁴` cost 66 bits of
/// overflow, `x⁵ = 64 + 66 + 0 + 2 = 132`, and no upstream auto-reduction
/// fires. Two reductions per S-box is the floor for this gadget's `mul`
/// (its output must be re-consumed twice).
pub fn sbox<F, P, CS>(
  cs: &mut CS,
  x: &EmulatedFieldElement<F, P>,
) -> Result<EmulatedFieldElement<F, P>, SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  let xr = x.reduce(&mut cs.namespace(|| "reduce input"))?;
  let x2 = xr.mul(&mut cs.namespace(|| "x^2"), &xr)?;
  let x2r = x2.reduce(&mut cs.namespace(|| "reduce x^2"))?;
  let x4 = x2r.mul(&mut cs.namespace(|| "x^4"), &x2r)?;
  x4.mul(&mut cs.namespace(|| "x^5"), &xr)
}

/// Reduce lanes 1–2 after every `LANE_REDUCTION_STRIDE`-th partial round.
///
/// This explicit schedule is load-bearing, not an optimization knob. The
/// pinned upstream reduces an operand only when an operation's overflow
/// precondition fails, and its `reduce` itself asserts
/// `overflow + 2 <= max_overflow`. A pure addition/scaling chain grows
/// overflow by ~4 bits per partial round; left alone, lanes 1–2 creep to
/// `max_overflow` (191 at 64-bit limbs over T256) and the next forced
/// reduction PANICS (`193 > 191`) — observed, not theoretical. S-box `mul`
/// chains are safe (their large overflow jumps force reductions early, far
/// below the cliff), so lane 0 needs no schedule. Lanes 1–2 restart near
/// ~137 overflow after every reduction (they immediately reabsorb the
/// S-box output through the matrix) and creep ~4 bits per partial round;
/// with the accepted stride of 8 the worst tracked overflow stays near
/// 170, under the 189 safety line. A stride of 16 was observed to panic
/// (the raw-cliff regression test reproduces the mechanism).
const LANE_REDUCTION_STRIDE: usize = 8;

/// Synthesize one Poseidon2 permutation, mirroring the host reference
/// [`crate::poseidon2::permute`]: initial external layer, then per round
/// ARC → S-box → linear layer (full rounds touch all lanes, partial rounds
/// lane 0 only), plus the [`LANE_REDUCTION_STRIDE`] lane-reduction
/// schedule. The output lanes are tracked residues, not canonical values.
/// Rejects a parameter set whose modulus disagrees with `P`.
pub fn synthesize_permutation<F, P, CS>(
  cs: &mut CS,
  params: &Poseidon2Params,
  state: [EmulatedFieldElement<F, P>; T],
) -> Result<[EmulatedFieldElement<F, P>; T], SynthesisError>
where
  F: PrimeFieldBits,
  P: EmulatedFieldParams,
  CS: ConstraintSystem<F>,
{
  if biguint_to_bigint(params.modulus()) != P::modulus() {
    return Err(SynthesisError::Unsatisfiable);
  }
  let mut s = apply_matrix(
    &mut cs.namespace(|| "initial linear layer"),
    params.m_e(),
    &state,
  )?;
  let rounds = params.round_constants().len();
  let mut partial_count = 0usize;
  for round in 1..=rounds {
    let rc = &params.round_constants()[round - 1];
    let full = is_full_round(round);
    if rc.len() != if full { T } else { 1 } {
      return Err(SynthesisError::Unsatisfiable);
    }
    let mut rcs = cs.namespace(|| format!("round {round}"));
    if full {
      for lane in 0..T {
        let arced = add_round_constant(
          &mut rcs.namespace(|| format!("arc lane {lane}")),
          &s[lane],
          &rc[lane],
        )?;
        s[lane] = sbox(&mut rcs.namespace(|| format!("sbox lane {lane}")), &arced)?;
      }
      s = apply_matrix(&mut rcs.namespace(|| "external layer"), params.m_e(), &s)?;
    } else {
      let arced = add_round_constant(&mut rcs.namespace(|| "arc lane 0"), &s[0], &rc[0])?;
      s[0] = sbox(&mut rcs.namespace(|| "sbox lane 0"), &arced)?;
      s = apply_matrix(&mut rcs.namespace(|| "internal layer"), params.m_i(), &s)?;
      partial_count += 1;
      if partial_count.is_multiple_of(LANE_REDUCTION_STRIDE) {
        s[1] = s[1].reduce(&mut rcs.namespace(|| "scheduled reduce lane 1"))?;
        s[2] = s[2].reduce(&mut rcs.namespace(|| "scheduled reduce lane 2"))?;
      }
    }
  }
  Ok(s)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::poseidon2::{FIELD_ORDER, build_all_params, build_inputs, chain_iv, permute};
  use crate::provider::T256HyraxEngine;
  use crate::traits::Engine;
  use bellpepper_core::test_cs::TestConstraintSystem;
  use ff::PrimeField;
  use num_traits::Zero;

  type S = <T256HyraxEngine as Engine>::Scalar;

  /// Reduce a raw circuit-side integer representative modulo `p`.
  fn canonical<P: EmulatedFieldParams>(v: &BigInt) -> BigUint {
    let (_, mag) = (v % P::modulus()).into_parts();
    mag
  }

  fn field_of<P: EmulatedFieldParams>() -> Field {
    *FIELD_ORDER
      .iter()
      .find(|f| biguint_to_bigint(&f.modulus()) == P::modulus())
      .expect("params type matches a workload field")
  }

  fn per_field(run: &mut dyn FnMut(Field)) {
    for f in FIELD_ORDER {
      run(f);
    }
  }

  #[test]
  fn params_match_poseidon2_moduli() {
    let set = build_all_params().unwrap();
    assert_eq!(
      Bn254FrParams::modulus(),
      biguint_to_bigint(set.get(Field::Bn254Fr).modulus())
    );
    assert_eq!(
      Bls12381FrParams::modulus(),
      biguint_to_bigint(set.get(Field::Bls12381Fr).modulus())
    );
    assert_eq!(
      Secp256k1FrParams::modulus(),
      biguint_to_bigint(set.get(Field::Secp256k1Fr).modulus())
    );
    // 4 × 64-bit limbs cover every modulus, and the native capacity admits
    // the base multiplication precondition with ample overflow headroom.
    per_field(&mut |f| {
      assert!(f.modulus().bits() as usize <= NUM_LIMBS * BITS_PER_LIMB);
    });
    assert!((S::CAPACITY as usize) >= 2 * BITS_PER_LIMB + 2);
  }

  #[test]
  fn range_check_boundary() {
    // 2^64 − 1 satisfies a 64-bit range check; 2^64 does not.
    let mut cs = TestConstraintSystem::<S>::new();
    let good = AllocatedNum::alloc(cs.namespace(|| "good"), || Ok(S::from(u64::MAX))).unwrap();
    range_check_num(
      &mut cs.namespace(|| "check good"),
      &Num::from(good),
      BITS_PER_LIMB,
    )
    .unwrap();
    assert!(cs.is_satisfied());

    let mut cs = TestConstraintSystem::<S>::new();
    let bad =
      AllocatedNum::alloc(cs.namespace(|| "bad"), || Ok(S::from_u128(1u128 << 64))).unwrap();
    range_check_num(
      &mut cs.namespace(|| "check bad"),
      &Num::from(bad),
      BITS_PER_LIMB,
    )
    .unwrap();
    assert!(!cs.is_satisfied());
  }

  #[test]
  fn alloc_private_checked_costs_and_bounds() {
    // Full-width value allocates 4 limbs at exactly 64 range-check
    // constraints each; a 2^256-wide value is rejected before allocation.
    let max = (BigUint::from(1u8) << 256u32) - BigUint::from(1u8);
    let mut cs = TestConstraintSystem::<S>::new();
    let _ = alloc_private_checked::<S, Bn254FrParams, _>(&mut cs.namespace(|| "max"), Some(&max))
      .unwrap();
    assert!(cs.is_satisfied());
    assert_eq!(cs.num_constraints(), NUM_LIMBS * BITS_PER_LIMB);

    let too_wide = BigUint::from(1u8) << 256u32;
    let mut cs = TestConstraintSystem::<S>::new();
    assert!(
      alloc_private_checked::<S, Bn254FrParams, _>(&mut cs.namespace(|| "wide"), Some(&too_wide))
        .is_err()
    );
  }

  /// Pinned operand pairs per field: boundaries plus noncanonical
  /// representatives (`≥ p`, still `< 2^256`).
  fn mul_cases(p: &BigUint) -> Vec<(BigUint, BigUint)> {
    let msgs = build_inputs(3).unwrap();
    vec![
      (BigUint::zero(), msgs[0].clone()),
      (BigUint::from(1u8), msgs[1].clone()),
      (p - 1u8, p - 1u8),
      (p + BigUint::from(5u8), msgs[2].clone()),
      (msgs[0].clone(), msgs[1].clone()),
    ]
  }

  fn check_mul_matches_host<P: EmulatedFieldParams>() {
    let field = field_of::<P>();
    let p = field.modulus();
    for (idx, (a, b)) in mul_cases(&p).iter().enumerate() {
      let mut cs = TestConstraintSystem::<S>::new();
      let ea = alloc_private_checked::<S, P, _>(&mut cs.namespace(|| format!("a {idx}")), Some(a))
        .unwrap();
      let eb = alloc_private_checked::<S, P, _>(&mut cs.namespace(|| format!("b {idx}")), Some(b))
        .unwrap();
      let ec = ea
        .mul(&mut cs.namespace(|| format!("mul {idx}")), &eb)
        .unwrap();
      assert!(
        cs.is_satisfied(),
        "{}: mul case {idx} unsatisfied",
        field.name()
      );
      let got = canonical::<P>(&BigInt::try_from(&ec).unwrap());
      assert_eq!(got, (a * b) % &p, "{}: mul case {idx} value", field.name());
    }
  }

  #[test]
  fn mul_matches_host() {
    check_mul_matches_host::<Bn254FrParams>();
    check_mul_matches_host::<Bls12381FrParams>();
    check_mul_matches_host::<Secp256k1FrParams>();
  }

  fn check_linear_matches_host<P: EmulatedFieldParams>() {
    // 3a + 2b + c + rc against the host, exercising scale_small on both
    // constant and allocated elements.
    let field = field_of::<P>();
    let p = field.modulus();
    let msgs = build_inputs(2).unwrap();
    let (a, b) = (&msgs[0], &msgs[1]);
    let c = chain_iv();
    let rc = BigUint::from(0xdead_beefu32);

    let mut cs = TestConstraintSystem::<S>::new();
    let ea = alloc_private_checked::<S, P, _>(&mut cs.namespace(|| "a"), Some(a)).unwrap();
    let eb = alloc_private_checked::<S, P, _>(&mut cs.namespace(|| "b"), Some(b)).unwrap();
    let ec = EmulatedFieldElement::<S, P>::from(&biguint_to_bigint(&c));
    let row = matrix_row(&mut cs.namespace(|| "row"), &[3, 2, 1], &[ea, eb, ec]).unwrap();
    let out = add_round_constant(&mut cs.namespace(|| "rc"), &row, &rc).unwrap();
    let free_constraints = cs.num_constraints();
    assert!(cs.is_satisfied());
    let got = canonical::<P>(&BigInt::try_from(&out).unwrap());
    assert_eq!(got, (a * 3u8 + b * 2u8 + &c + &rc) % &p, "{}", field.name());
    // The linear layer itself adds no constraints beyond the two operand
    // allocations (4 · 64 range-check rows each).
    assert_eq!(
      free_constraints,
      2 * NUM_LIMBS * BITS_PER_LIMB,
      "{}",
      field.name()
    );
  }

  #[test]
  fn linear_layer_matches_host_and_is_free() {
    check_linear_matches_host::<Bn254FrParams>();
    check_linear_matches_host::<Bls12381FrParams>();
    check_linear_matches_host::<Secp256k1FrParams>();
  }

  fn check_sbox_matches_host<P: EmulatedFieldParams>() -> usize {
    let field = field_of::<P>();
    let p = field.modulus();
    let x = &build_inputs(1).unwrap()[0];
    let mut cs = TestConstraintSystem::<S>::new();
    let ex = alloc_private_checked::<S, P, _>(&mut cs.namespace(|| "x"), Some(x)).unwrap();
    let e5 = sbox(&mut cs.namespace(|| "sbox"), &ex).unwrap();
    assert!(cs.is_satisfied(), "{}", field.name());
    let got = canonical::<P>(&BigInt::try_from(&e5).unwrap());
    let x2 = (x * x) % &p;
    let x4 = (&x2 * &x2) % &p;
    assert_eq!(got, (&x4 * x) % &p, "{}", field.name());
    cs.num_constraints()
  }

  #[test]
  fn sbox_matches_host() {
    check_sbox_matches_host::<Bn254FrParams>();
    check_sbox_matches_host::<Bls12381FrParams>();
    check_sbox_matches_host::<Secp256k1FrParams>();
  }

  /// Synthesize one chain step `P([iv, m, 0])` with a constant IV/zero
  /// lane and an allocated message, mirroring the first permutation of a
  /// field block. Returns the constraint system and output lanes.
  fn synthesize_pilot<P: EmulatedFieldParams>(
    message: &BigUint,
  ) -> (TestConstraintSystem<S>, [EmulatedFieldElement<S, P>; T]) {
    let field = field_of::<P>();
    let set = build_all_params().unwrap();
    let params = set.get(field);
    let mut cs = TestConstraintSystem::<S>::new();
    let m =
      alloc_private_checked::<S, P, _>(&mut cs.namespace(|| "message"), Some(message)).unwrap();
    let iv = EmulatedFieldElement::<S, P>::from(&biguint_to_bigint(&chain_iv()));
    let zero = EmulatedFieldElement::<S, P>::from(&BigInt::from(0u8));
    let out = synthesize_permutation(&mut cs.namespace(|| "perm"), params, [iv, m, zero]).unwrap();
    (cs, out)
  }

  fn check_permutation_matches_host<P: EmulatedFieldParams>() {
    let field = field_of::<P>();
    let set = build_all_params().unwrap();
    let params = set.get(field);
    let message = &build_inputs(1).unwrap()[0];
    let (cs, out) = synthesize_pilot::<P>(message);
    assert!(cs.is_satisfied(), "{}", field.name());
    let expect = permute(params, [chain_iv(), message.clone(), BigUint::zero()]).unwrap();
    for (lane, (got, want)) in out.iter().zip(expect.iter()).enumerate() {
      let got = canonical::<P>(&BigInt::try_from(got).unwrap());
      assert_eq!(&got, want, "{}: lane {lane}", field.name());
    }
  }

  #[test]
  fn permutation_matches_host() {
    check_permutation_matches_host::<Bn254FrParams>();
    check_permutation_matches_host::<Bls12381FrParams>();
    check_permutation_matches_host::<Secp256k1FrParams>();
  }

  fn check_digest_linkage<P: EmulatedFieldParams>() {
    let field = field_of::<P>();
    let p = field.modulus();
    let set = build_all_params().unwrap();
    let params = set.get(field);
    let message = &build_inputs(1).unwrap()[0];
    let digest =
      permute(params, [chain_iv(), message.clone(), BigUint::zero()]).unwrap()[0].clone();

    // Honest digest satisfies; a canonical-but-noncongruent digest does not.
    for (label, claimed, want_sat) in [
      ("honest", digest.clone(), true),
      ("tampered", (&digest + 1u8) % &p, false),
    ] {
      let (mut cs, out) = synthesize_pilot::<P>(message);
      let (pub_elt, limbs) = alloc_public_digest_checked::<S, P, _>(
        &mut cs.namespace(|| format!("digest {label}")),
        Some(&claimed),
      )
      .unwrap();
      assert_eq!(limbs.len(), NUM_LIMBS);
      EmulatedFieldElement::assert_is_equal(
        &mut cs.namespace(|| format!("link {label}")),
        &out[0],
        &pub_elt,
      )
      .unwrap();
      assert_eq!(
        cs.is_satisfied(),
        want_sat,
        "{}: {label} digest",
        field.name()
      );
      assert_eq!(
        cs.num_inputs(),
        1 + NUM_LIMBS,
        "{}: public limb count",
        field.name()
      );
    }
  }

  #[test]
  fn digest_linkage() {
    check_digest_linkage::<Bn254FrParams>();
    check_digest_linkage::<Bls12381FrParams>();
    check_digest_linkage::<Secp256k1FrParams>();
  }

  /// Audit-only raw-cliff regression (§3 fact 6, §9 item 4): the pinned
  /// upstream's lazy reduction PANICS — it does not error — when a pure
  /// addition chain creeps overflow to `max_overflow` and the next
  /// operation forces a reduction. The explicit schedule exists because of
  /// this; a dependency bump that changes the behavior must fail here.
  #[test]
  fn upstream_overflow_cliff_still_panics() {
    let result = std::panic::catch_unwind(|| {
      let mut cs = TestConstraintSystem::<S>::new();
      let x = alloc_private_checked::<S, Bn254FrParams, _>(
        &mut cs.namespace(|| "x"),
        Some(&BigUint::from(1u8)),
      )
      .unwrap();
      // Each add raises tracked overflow by one bit; the precondition
      // admits results up to max_overflow (191), and the subsequent add's
      // forced reduction of a 191-overflow operand asserts 193 <= 191.
      let mut acc = x;
      for i in 0..300 {
        acc = acc
          .add(&mut cs.namespace(|| format!("add {i}")), &acc.clone())
          .unwrap();
      }
      acc
    });
    assert!(
      result.is_err(),
      "upstream overflow cliff no longer panics; re-audit the reduction \
       schedule and §3 fact 6 before accepting the dependency change"
    );
  }

  #[test]
  fn tampered_witness_limb_is_unsatisfied() {
    let message = &build_inputs(1).unwrap()[0];
    let (mut cs, _out) = synthesize_pilot::<Bn254FrParams>(message);
    assert!(cs.is_satisfied());
    cs.set("message/limb 0/num", S::from(0x1234_5678u64));
    assert!(!cs.is_satisfied());
  }

  /// The reduction schedule and pilot sizes are deterministic functions of
  /// the circuit structure; pin them so a dependency bump or schedule edit
  /// fails loudly (`plan/poseidon_spartan_bench.md` §4).
  #[test]
  fn pilot_counts_pinned() {
    fn check<P: EmulatedFieldParams>(constraints: usize, aux: usize) {
      let field = field_of::<P>();
      let message = &build_inputs(1).unwrap()[0];
      let (cs, _out) = synthesize_pilot::<P>(message);
      assert_eq!(cs.num_constraints(), constraints, "{}", field.name());
      assert_eq!(cs.scalar_aux().len(), aux, "{}", field.name());
      let reduces = cs
        .pretty_print_list()
        .iter()
        .filter(|c| c.contains("remainder modulo field modulus") && c.contains("limb 0/last bit"))
        .count();
      // 80 S-boxes × 2 explicit reductions + 7 stride points × 2 lanes.
      assert_eq!(reduces, 174, "{}", field.name());
    }
    check::<Bn254FrParams>(309_066, 307_494);
    check::<Bls12381FrParams>(309_240, 307_668);
    check::<Secp256k1FrParams>(309_414, 307_842);
  }

  /// Feasibility-spike report (`plan/poseidon_spartan_bench.md` §3): pilot
  /// constraint/variable counts and overflow-triggered reduction sites.
  /// Run with `cargo test --release spike_report -- --ignored --nocapture`.
  #[test]
  #[ignore]
  fn spike_report() {
    fn report<P: EmulatedFieldParams>() {
      let field = field_of::<P>();
      let message = &build_inputs(1).unwrap()[0];
      let (cs, _out) = synthesize_pilot::<P>(message);
      let constraints = cs.pretty_print_list();
      // One "remainder" anchor per reduction site; one "quotient" anchor per
      // modular-equality assertion (each reduction contains one, plus one
      // per explicit digest linkage).
      let reduces = constraints
        .iter()
        .filter(|c| c.contains("remainder modulo field modulus") && c.contains("limb 0/last bit"))
        .count();
      let equalities = constraints
        .iter()
        .filter(|c| c.contains("quotient when divided by modulus") && c.contains("limb 0/last bit"))
        .count();
      let sbox_cost = check_sbox_matches_host::<P>();
      println!(
        "{}: permutation constraints = {}, aux vars = {}, single s-box = {}, \
         reductions = {}, modular equalities = {}",
        field.name(),
        cs.num_constraints(),
        cs.scalar_aux().len(),
        sbox_cost,
        reduces,
        equalities,
      );
    }
    report::<Bn254FrParams>();
    report::<Bls12381FrParams>();
    report::<Secp256k1FrParams>();
  }
}
