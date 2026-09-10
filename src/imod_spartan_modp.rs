//! Dual-field Integer Mod-R1CS SNARK driver, generic over `M: ModEngine`.
//!
//! Mirrors `crate::imod_spartan` but the shape, witness, and matrix
//! entries are integer-valued (`BigUint`), the prime `p` over which the
//! sumcheck runs is sampled from the transcript via Fiat-Shamir (the
//! bounded, audited P0-D sampler, see `M::sample_params` and
//! `crate::prime_sampler`), and the SNARK
//! verifies the IntMod-R1CS relation `Az ∘ Bz = Cz + m ∘ q` mod that
//! sampled `p`. The Mod-PCS commits integer polynomials and opens at
//! `Z_p` points returning `Z_p` evals — the `p ≠ q` reconciliation is
//! the Mod-PCS's responsibility (the IntEval protocol in
//! `provider::pcs::integer_modpcs`); this driver
//! treats it as a black-box contract.
//!
//! Flow:
//!   1. Bootstrap transcript with `M::bootstrap_params()` (placeholder).
//!   2. Byte-absorb the vk digest, two integer-poly commitments, and the
//!      public IO `x` (`BigUint` LE bytes).
//!   3. `params = M::sample_params(transcript, log)` derives `p` from
//!      squeeze bytes; `transcript.set_params(params)` switches
//!      typed-squeeze reductions into `Z_p`.
//!   4. Reduce shape/witness/IO `BigUint`s to `M::Scalar` mod `p`.
//!   5. Run outer + inner sumchecks in `Z_p`.
//!   6. Open `w` and `q` at `Z_p` points via `M::ModPCS::prove`, passing
//!      the original `BigUint` polynomials (integer view).
//!
//! Single witness segment; no shared/precommitted/rest split; no limb
//! decomposition; no range checks; no BDDT first-round optimization.
//!
//! Prime-sampler audit (P0-D): before the runtime-modulus draw, `prove`
//! and `verify` compute the exact number of sampler invocations their
//! schedule performs — one runtime draw plus the Mod-PCS accessor result
//! for the single batched open they issue — reject a total above
//! `MAX_PRIME_INVOCATIONS`, construct one `PrimeAuditLog`, thread it
//! through `sample_params` and the trait call, and `finish_exact` it on
//! success. `prove_with_prime_audit` / `verify_with_prime_audit` return
//! the records alongside the value or error; `prove` / `verify` wrap them
//! and discard only that diagnostic copy.

use crate::{
  errors::SpartanError,
  imod_r1cs_modp::{IntModR1CSInstanceModp, IntModR1CSShapeModp, IntModR1CSWitnessModp},
  math::Math,
  polys_modp::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  prime_sampler::{
    MAX_PRIME_INVOCATIONS, PrimeAuditLog, PrimeAuditedOutcome, checked_schedule_add,
  },
  provider::keccak::Keccak256Transcript,
  start_span,
  sumcheck_modp::SumcheckProof,
  traits::{
    mod_engine::{ModEngine, ModPCSEngineTrait, SmallValueBlock, SumcheckEngine, SumcheckField},
    transcript::{ByteTranscript, TranscriptEngineTrait},
  },
};
use num_bigint::BigUint;
use rand_core::{CryptoRng, CryptoRngCore, RngCore};
use rayon::prelude::*;
use tracing::info;

type MScalar<M> = <M as SumcheckEngine>::Scalar;
type MParams<M> = <MScalar<M> as SumcheckField>::Params;
type ModPCS<M> = <M as ModEngine>::ModPCS;
type ModCK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::CommitmentKey;
type ModVK<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::VerifierKey;
type ModBatchEvalArg<M> = <ModPCS<M> as ModPCSEngineTrait<M>>::BatchEvaluationArgument;

/// Canonical proof-component serialization: bincode 1.3 with
/// `DefaultOptions` pinned to little-endian, fixed-int encoding (the
/// crate's canonical bincode configuration). Every bincode error maps to
/// [`SpartanError::SerializationError`] — never to a zero size.
pub(crate) fn to_canonical_bytes<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, SpartanError> {
  use bincode::Options;
  bincode::DefaultOptions::new()
    .with_little_endian()
    .with_fixint_encoding()
    .serialize(value)
    .map_err(|e| SpartanError::SerializationError {
      reason: format!("canonical bincode serialization failed: {e}"),
    })
}

/// Convert a `BigUint` integer into an `M::Scalar` value by reducing
/// modulo the runtime modulus carried in `params`.
fn biguint_to_scalar<M: ModEngine>(v: &BigUint, params: &MParams<M>) -> MScalar<M> {
  MScalar::<M>::from_bytes_reduce(params, &v.to_bytes_le())
}

/// Width-grouped selector for the segment `[start, start+2^log_len)` at the
/// witness point `wp` (MSB-first): the product over the leading
/// `wp.len() - log_len` coords of `eq(bit_j, wp[j])`, with bits taken from
/// `start >> log_len`. The segment's local evaluation point is the last
/// `log_len` coords of `wp`, so `W(wp) = sum_seg selector(seg) * Seg(local)`.
fn segment_selector<M: ModEngine>(
  start: usize,
  log_len: usize,
  wp: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  let hi = wp.len() - log_len;
  let h = start >> log_len;
  let one = MScalar::<M>::one(params);
  let mut sel = one;
  for (j, wj) in wp.iter().take(hi).enumerate() {
    let bit = (h >> (hi - 1 - j)) & 1;
    let factor = if bit == 1 { *wj } else { one - *wj };
    sel *= factor;
  }
  sel
}

/// Distribute the shape's aligned small-value blocks across the width
/// segments, rebasing each block's `start` to be relative to the segment
/// that contains it. Every block lies within exactly one segment (the
/// layout keeps the 16-bit region inside one aligned segment).
fn segment_relative_blocks(
  segs: &[crate::imod_r1cs_modp::WidthSegment],
  blocks: &[SmallValueBlock],
) -> Vec<Vec<SmallValueBlock>> {
  segs
    .iter()
    .map(|seg| {
      blocks
        .iter()
        .filter(|b| b.start >= seg.start && b.start + b.size() <= seg.start + seg.size())
        .map(|b| SmallValueBlock {
          start: b.start - seg.start,
          log_len: b.log_len,
        })
        .collect()
    })
    .collect()
}

fn biguint_vec_to_scalars<M: ModEngine>(v: &[BigUint], params: &MParams<M>) -> Vec<MScalar<M>> {
  v.par_iter()
    .map(|b| biguint_to_scalar::<M>(b, params))
    .collect()
}

fn biguint_matrix_to_scalars<M: ModEngine>(
  entries: &[(usize, usize, BigUint)],
  params: &MParams<M>,
) -> Vec<(usize, usize, MScalar<M>)> {
  entries
    .par_iter()
    .map(|(i, j, v)| (*i, *j, biguint_to_scalar::<M>(v, params)))
    .collect()
}

/// Prover key.
#[derive(Clone)]
pub struct IntModSpartanModpProverKey<M: ModEngine> {
  pub(crate) ck: ModCK<M>,
  pub(crate) shape: IntModR1CSShapeModp<M>,
  pub(crate) vk_digest: [u8; 32],
}

impl<M: ModEngine> IntModSpartanModpProverKey<M> {
  /// Commitment key used for `w` and `q`. External callers need this to
  /// build witness/instance pairs via `IntModR1CSWitnessModp::new`.
  pub fn ck(&self) -> &ModCK<M> {
    &self.ck
  }
}

/// Verifier key.
#[derive(Clone)]
pub struct IntModSpartanModpVerifierKey<M: ModEngine> {
  pub(crate) vk_ee: ModVK<M>,
  pub(crate) shape: IntModR1CSShapeModp<M>,
  pub(crate) digest: [u8; 32],
}

impl<M: ModEngine> IntModSpartanModpVerifierKey<M> {
  /// 32-byte verifier-key digest (Keccak256 of the shape).
  pub fn digest(&self) -> [u8; 32] {
    self.digest
  }
}

/// IntMod-R1CS SNARK proof over a `ModEngine`. Serialization is deferred — the
/// dynamic-prime types (`M::Scalar`, `Params`) aren't `Serialize` yet.
#[derive(Clone, Debug)]
pub struct IntModSpartanModpSNARK<M: ModEngine> {
  // outer sumcheck
  sc_outer: SumcheckProof<M>,
  v_a: MScalar<M>,
  v_b: MScalar<M>,
  v_c: MScalar<M>,
  v_m: MScalar<M>,
  v_q: MScalar<M>,
  // inner sumcheck (for w)
  sc_inner: SumcheckProof<M>,
  eval_w: MScalar<M>,
  // Per-segment evaluations of W (width-grouped commitment): eval_w =
  // sum_i selector_i(r_y[1..]) · seg_evals[i]. Empty when the shape
  // declares no segments (W committed as one polynomial).
  seg_evals: Vec<MScalar<M>>,
  // Mod-PCS opening of W (at r_y[1..]) and Q (at r_x) merged into ONE
  // batched argument: a single shared LogUp-GKR range check and a single
  // combined inner-product opening cover both polynomials.
  eval_arg: ModBatchEvalArg<M>,
}

impl IntModSpartanModpSNARK<crate::provider::T256DynPrimeEngine> {
  /// The full batch evaluation argument, for serialization accounting
  /// (e.g. dumping the bytes to measure compressed size).
  pub fn eval_arg_ref(
    &self,
  ) -> &crate::provider::pcs::integer_modpcs::IntEvalBatchArgument<
    crate::provider::pcs::integer_modpcs::HyBackend,
  > {
    &self.eval_arg
  }

  /// Per-component serialized sizes of the batch evaluation argument:
  /// `(per_poly, range_check, combined_open)` bytes. Proof-size
  /// accounting; see `eval_arg_bytes` for the exact total bytes.
  pub fn eval_arg_component_sizes(&self) -> (usize, usize, usize) {
    self.eval_arg.component_sizes()
  }

  /// Setup with explicit IntEval params, so callers can size the
  /// committed-value norm bound (`log_t_f`) and limb bound (`log_t`) for
  /// wide operands — e.g. the ~2048-bit `mod N` values in the MultiSwap
  /// bench, which exceed the default `DEFAULT_LOG_T_F = 32`.
  pub fn setup_with_params(
    shape: IntModR1CSShapeModp<crate::provider::T256DynPrimeEngine>,
    params: crate::provider::pcs::integer_modpcs::IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<crate::provider::T256DynPrimeEngine>,
      IntModSpartanModpVerifierKey<crate::provider::T256DynPrimeEngine>,
    ),
    SpartanError,
  > {
    let (ck, vk_ee) = shape.commitment_key_with_params(params)?;
    Ok(Self::assemble_keys(shape, ck, vk_ee))
  }
}

impl IntModSpartanModpSNARK<crate::provider::T256DynPrimeBdEngine> {
  /// The full batch evaluation argument, for serialization accounting.
  pub fn eval_arg_ref(
    &self,
  ) -> &crate::provider::pcs::integer_modpcs::IntEvalBatchArgument<
    crate::provider::pcs::commit_backend::BdBackend,
  > {
    &self.eval_arg
  }

  /// The Brakedown batch-opening argument (column-opening groups plus
  /// directly-shipped small polynomials), for proof-size anatomy
  /// accounting.
  pub fn bd_open_args(
    &self,
  ) -> &crate::provider::pcs::commit_backend::BdBatchOpenArg<crate::provider::pt256::t256::Scalar>
  {
    &self.eval_arg.combined_open.backend
  }

  /// Brakedown-engine analog of `setup_with_params`.
  pub fn setup_with_params(
    shape: IntModR1CSShapeModp<crate::provider::T256DynPrimeBdEngine>,
    params: crate::provider::pcs::integer_modpcs::IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<crate::provider::T256DynPrimeBdEngine>,
      IntModSpartanModpVerifierKey<crate::provider::T256DynPrimeBdEngine>,
    ),
    SpartanError,
  > {
    let n = shape.num_vars.max(shape.num_cons);
    let num_vars = n.max(1).ilog2() as usize + if n.is_power_of_two() { 0 } else { 1 };
    params.validate(num_vars)?;
    // The fallible key constructor revalidates at the key's own capacity
    // (the original polynomial length `n`).
    let ck = crate::provider::pcs::integer_modpcs::BdModCommitmentKey::new(params.clone(), n)?;
    let vk = crate::provider::pcs::integer_modpcs::BdModVerifierKey::new(params);
    Ok(Self::assemble_keys(shape, ck, vk))
  }
}

impl IntModSpartanModpSNARK<crate::provider::M127DynPrimeBdEngine> {
  /// Per-component serialized sizes of the batch evaluation argument
  /// (proof-size accounting, mirrors the t256 accessor).
  pub fn eval_arg_component_sizes(&self) -> (usize, usize, usize) {
    self.eval_arg.component_sizes()
  }

  /// Small-field (F127/Brakedown) analog of `setup_with_params`. The
  /// params should come from `IntEvalParams::derive_for_q(127, ...)`.
  pub fn setup_with_params(
    shape: IntModR1CSShapeModp<crate::provider::M127DynPrimeBdEngine>,
    params: crate::provider::pcs::integer_modpcs::IntEvalParams,
  ) -> Result<
    (
      IntModSpartanModpProverKey<crate::provider::M127DynPrimeBdEngine>,
      IntModSpartanModpVerifierKey<crate::provider::M127DynPrimeBdEngine>,
    ),
    SpartanError,
  > {
    let n = shape.num_vars.max(shape.num_cons);
    let num_vars = n.max(1).ilog2() as usize + if n.is_power_of_two() { 0 } else { 1 };
    params.validate(num_vars)?;
    // The fallible key constructor revalidates at the key's own capacity
    // (the original polynomial length `n`).
    let ck = crate::provider::pcs::integer_modpcs::BdModCommitmentKey::new(params.clone(), n)?;
    let vk = crate::provider::pcs::integer_modpcs::BdModVerifierKey::new(params);
    Ok(Self::assemble_keys(shape, ck, vk))
  }
}

impl<M: ModEngine> IntModSpartanModpSNARK<M> {
  /// Whether every dynamic-field scalar this proof carries belongs to the
  /// modulus context `expected`: every outer/inner sumcheck round-poly
  /// coefficient, the five outer claims, `eval_w`, and — delegated to the
  /// required Mod-PCS hook — the complete evaluation argument. Called by
  /// `verify` immediately after re-sampling the runtime prime and before
  /// any proof arithmetic, so a proof from a foreign context is rejected
  /// cleanly instead of panicking inside `crypto-bigint`.
  fn is_in_context(&self, expected: &MParams<M>) -> bool {
    self.sc_outer.is_in_context(expected)
      && [self.v_a, self.v_b, self.v_c, self.v_m, self.v_q]
        .iter()
        .all(|v| v.is_in_context(expected))
      && self.sc_inner.is_in_context(expected)
      && self.eval_w.is_in_context(expected)
      && <ModPCS<M> as ModPCSEngineTrait<M>>::batch_arg_is_in_context(&self.eval_arg, expected)
  }

  /// Exact canonical bytes of the Mod-PCS batch evaluation argument (the
  /// dominant proof component), under the crate's pinned bincode
  /// configuration. Errors propagate as `SerializationError` — unlike the
  /// removed `eval_arg_size`, a failure is never reported as zero bytes.
  pub fn eval_arg_bytes(&self) -> Result<Vec<u8>, SpartanError> {
    to_canonical_bytes(&self.eval_arg)
  }

  /// `(outer sumcheck rounds, inner sumcheck rounds, claimed evaluations)`
  /// of the non-Mod-PCS proof remainder: the five outer claims
  /// (`v_a, v_b, v_c, v_m, v_q`), `eval_w`, and one evaluation per width
  /// segment. Structural counts only; see
  /// [`sumcheck_remainder_bytes`](Self::sumcheck_remainder_bytes) for the
  /// values.
  pub fn sumcheck_remainder_counts(&self) -> (usize, usize, usize) {
    (
      self.sc_outer.compressed_polys.len(),
      self.sc_inner.compressed_polys.len(),
      6 + self.seg_evals.len(),
    )
  }

  /// A canonical, length-framed little-endian serialization of every
  /// proof component outside the Mod-PCS batch argument — the outer and
  /// inner sumcheck round polynomials (compressed coefficient vectors),
  /// the five outer claims, `eval_w` and the per-segment evaluations —
  /// each scalar as its canonical LE bytes under a `u16` length prefix,
  /// each vector under a `u32` count prefix. The dynamic-prime scalars
  /// have no serde form, so this is the byte identity the benchmark's
  /// double-construction check compares (`DynPrime` values of equal
  /// modulus context serialize identically iff they are equal).
  pub fn sumcheck_remainder_bytes(&self) -> Vec<u8> {
    fn push_scalar<F: SumcheckField>(out: &mut Vec<u8>, v: &F) {
      let bytes = v.to_le_bytes();
      let len = u16::try_from(bytes.len()).expect("a field element is shorter than 64 KiB");
      out.extend_from_slice(&len.to_le_bytes());
      out.extend_from_slice(&bytes);
    }
    fn push_count(out: &mut Vec<u8>, n: usize) {
      let n = u32::try_from(n).expect("proof vectors are shorter than 2^32");
      out.extend_from_slice(&n.to_le_bytes());
    }
    fn push_sumcheck<M: ModEngine>(out: &mut Vec<u8>, sc: &SumcheckProof<M>) {
      push_count(out, sc.compressed_polys.len());
      for poly in &sc.compressed_polys {
        push_count(out, poly.coeffs_except_linear_term.len());
        for c in &poly.coeffs_except_linear_term {
          push_scalar(out, c);
        }
      }
    }
    let mut out = Vec::new();
    push_sumcheck(&mut out, &self.sc_outer);
    for v in [&self.v_a, &self.v_b, &self.v_c, &self.v_m, &self.v_q] {
      push_scalar(&mut out, v);
    }
    push_sumcheck(&mut out, &self.sc_inner);
    push_scalar(&mut out, &self.eval_w);
    push_count(&mut out, self.seg_evals.len());
    for v in &self.seg_evals {
      push_scalar(&mut out, v);
    }
    out
  }
}

impl<M> IntModSpartanModpSNARK<M>
where
  M: ModEngine<TE = Keccak256Transcript<M>>,
{
  /// Re-derive the runtime-prime context a verifier would sample for
  /// `(vk, U)`: replays the pre-`p` transcript absorption of
  /// `prove`/`verify`. Crate-internal test/bench support for the context
  /// regression fixtures.
  #[cfg(test)]
  pub(crate) fn resample_params(
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
  ) -> MParams<M> {
    let mut transcript =
      Keccak256Transcript::<M>::new_with_params(b"IntModSpartanModpSNARK", M::bootstrap_params());
    transcript.absorb_bytes(b"vk", &vk.digest);
    for cw in &U.comm_w {
      transcript.absorb(b"comm_w", cw);
    }
    transcript.absorb(b"comm_q", &U.comm_q);
    for xi in &U.x {
      transcript.absorb_bytes(b"x", &xi.to_bytes_le());
    }
    let mut log = PrimeAuditLog::with_expected(1).expect("one runtime draw is within the cap");
    M::sample_params(&mut transcript, &mut log).expect("test fixture runtime prime samples")
  }
  /// Setup: derive prover and verifier keys from the shape.
  pub fn setup(
    shape: IntModR1CSShapeModp<M>,
  ) -> Result<
    (
      IntModSpartanModpProverKey<M>,
      IntModSpartanModpVerifierKey<M>,
    ),
    SpartanError,
  > {
    let (ck, vk_ee) = shape.commitment_key();
    Ok(Self::assemble_keys(shape, ck, vk_ee))
  }

  /// Shared tail of `setup` / `setup_with_params`: precompute the
  /// commitment-key tables and assemble the prover/verifier keys from a
  /// prebuilt `(ck, vk_ee)` pair. (The Mod-PCS now owns its own size-1
  /// eval key internally, so there is no separate `ck_s` to build here.)
  pub(crate) fn assemble_keys(
    shape: IntModR1CSShapeModp<M>,
    ck: ModCK<M>,
    vk_ee: ModVK<M>,
  ) -> (
    IntModSpartanModpProverKey<M>,
    IntModSpartanModpVerifierKey<M>,
  ) {
    <ModPCS<M> as ModPCSEngineTrait<M>>::precompute_ck(&ck);

    let digest = shape.digest();
    let vk = IntModSpartanModpVerifierKey {
      vk_ee,
      shape: shape.clone(),
      digest,
    };
    let pk = IntModSpartanModpProverKey {
      ck,
      shape,
      vk_digest: digest,
    };
    (pk, vk)
  }

  /// The per-polynomial segment widths of the width-grouped open, or
  /// `None` for the plain two-polynomial (`W`, `Q`) open — the exact
  /// `log_t_fs` argument the driver passes to `*_with_params`.
  fn open_schedule(shape: &IntModR1CSShapeModp<M>, native_log_t_f: usize) -> Option<Vec<usize>> {
    let segs = shape.width_segments();
    if segs.is_empty() {
      return None;
    }
    let mut log_t_fs: Vec<usize> = segs.iter().map(|seg| seg.log_t_f).collect();
    log_t_fs.push(native_log_t_f);
    Some(log_t_fs)
  }

  /// Total prime-sampler invocations of the driver: one runtime draw plus
  /// `pcs` (the Mod-PCS accessor result), rejecting a total above the cap.
  fn total_prime_invocations(pcs: usize) -> Result<usize, SpartanError> {
    let total = checked_schedule_add(1, pcs)?;
    if total > MAX_PRIME_INVOCATIONS {
      return Err(SpartanError::PrimeAuditLog {
        reason: format!(
          "the proof schedule needs {total} prime-sampler invocations, above the cap \
           {MAX_PRIME_INVOCATIONS}"
        ),
      });
    }
    Ok(total)
  }

  /// Exact number of P0-D prime-sampler invocations `prove` performs
  /// under `pk`: one runtime-modulus draw plus the Mod-PCS count for the
  /// single batched open (two polynomials, or one per width segment plus
  /// `Q`). Computed — with every width conversion validated — before any
  /// transcript draw; a total above `MAX_PRIME_INVOCATIONS` is rejected.
  pub fn prime_sampler_invocations_for_prove(
    pk: &IntModSpartanModpProverKey<M>,
  ) -> Result<usize, SpartanError> {
    let native = <ModPCS<M> as ModPCSEngineTrait<M>>::commitment_log_t_f(&pk.ck);
    let pcs = match Self::open_schedule(&pk.shape, native) {
      None => {
        <ModPCS<M> as ModPCSEngineTrait<M>>::prime_sampler_invocations_for_prove(&pk.ck, 2, None)?
      }
      Some(log_t_fs) => <ModPCS<M> as ModPCSEngineTrait<M>>::prime_sampler_invocations_for_prove(
        &pk.ck,
        log_t_fs.len(),
        Some(&log_t_fs),
      )?,
    };
    Self::total_prime_invocations(pcs)
  }

  /// Verifier mirror of
  /// [`prime_sampler_invocations_for_prove`](Self::prime_sampler_invocations_for_prove):
  /// the count `verify` performs under `vk`. Agrees with the prover's for
  /// matching keys.
  pub fn prime_sampler_invocations_for_verify(
    vk: &IntModSpartanModpVerifierKey<M>,
  ) -> Result<usize, SpartanError> {
    let native = <ModPCS<M> as ModPCSEngineTrait<M>>::verifier_log_t_f(&vk.vk_ee);
    let pcs = match Self::open_schedule(&vk.shape, native) {
      None => <ModPCS<M> as ModPCSEngineTrait<M>>::prime_sampler_invocations_for_verify(
        &vk.vk_ee, 2, None,
      )?,
      Some(log_t_fs) => <ModPCS<M> as ModPCSEngineTrait<M>>::prime_sampler_invocations_for_verify(
        &vk.vk_ee,
        log_t_fs.len(),
        Some(&log_t_fs),
      )?,
    };
    Self::total_prime_invocations(pcs)
  }

  /// Run `body` against a fresh audit log sized by `expected`, then
  /// enforce the exact record count on success. Every failure — schedule
  /// validation, the run itself, or the exact-count check — keeps the
  /// records appended so far.
  fn run_audited<T>(
    expected: Result<usize, SpartanError>,
    body: impl FnOnce(&mut PrimeAuditLog) -> Result<T, SpartanError>,
  ) -> PrimeAuditedOutcome<T> {
    let mut log = match expected.and_then(PrimeAuditLog::with_expected) {
      Ok(log) => log,
      Err(source) => {
        return PrimeAuditedOutcome::Failure {
          source,
          records: Vec::new(),
        };
      }
    };
    match body(&mut log) {
      Ok(value) => {
        let snapshot = log.records().to_vec();
        match log.finish_exact() {
          Ok(records) => PrimeAuditedOutcome::Success { value, records },
          Err(source) => PrimeAuditedOutcome::Failure {
            source,
            records: snapshot,
          },
        }
      }
      Err(source) => PrimeAuditedOutcome::Failure {
        source,
        records: log.into_records(),
      },
    }
  }

  /// [`prove`](Self::prove) with the complete prime-sampler audit log:
  /// the schedule is validated and the one log constructed before the
  /// runtime-modulus draw; on success the log holds exactly the scheduled
  /// number of records, on failure the completed prefix.
  pub fn prove_with_prime_audit(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
  ) -> PrimeAuditedOutcome<Self> {
    Self::prove_with_prime_audit_rng(pk, U, W, &mut rand::thread_rng())
  }

  /// [`prove_with_prime_audit`](Self::prove_with_prime_audit) drawing
  /// every prover coin — the Mod-PCS internal commitment blinds and the
  /// Hyrax/IPA masking randomness — from `rng` instead of the thread RNG.
  /// The transcript challenges and the P0-D prime draws are transcript
  /// squeezes and do not consume `rng`. With a seeded generator (and a
  /// witness committed by `IntModR1CSWitnessModp::new_with_rng` from the
  /// same coins) the proof bytes are reproducible; Brakedown draws no
  /// prover randomness at all.
  pub fn prove_with_prime_audit_rng(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
    rng: &mut (impl RngCore + CryptoRng),
  ) -> PrimeAuditedOutcome<Self> {
    let rng: &mut dyn CryptoRngCore = rng;
    Self::run_audited(Self::prime_sampler_invocations_for_prove(pk), |log| {
      Self::prove_inner(pk, U, W, log, rng)
    })
  }

  /// [`prove`](Self::prove) drawing every prover coin from `rng` (see
  /// [`prove_with_prime_audit_rng`](Self::prove_with_prime_audit_rng));
  /// the diagnostic audit copy is discarded.
  pub fn prove_with_rng(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
    rng: &mut (impl RngCore + CryptoRng),
  ) -> Result<Self, SpartanError> {
    Self::prove_with_prime_audit_rng(pk, U, W, rng).into_result()
  }

  /// Prove satisfaction of the IntMod-R1CS instance. Wraps
  /// [`prove_with_prime_audit`](Self::prove_with_prime_audit) and
  /// discards only the diagnostic audit copy.
  pub fn prove(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
  ) -> Result<Self, SpartanError> {
    Self::prove_with_prime_audit(pk, U, W).into_result()
  }

  /// The proving flow against a caller-owned audit log (see the module
  /// docs for the schedule).
  fn prove_inner(
    pk: &IntModSpartanModpProverKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
    log: &mut PrimeAuditLog,
    rng: &mut dyn CryptoRngCore,
  ) -> Result<Self, SpartanError> {
    let (_prove_span, prove_t) = start_span!("imod_spartan_modp_prove");

    // 1. Bootstrap transcript with placeholder params.
    let mut transcript =
      Keccak256Transcript::<M>::new_with_params(b"IntModSpartanModpSNARK", M::bootstrap_params());

    // 2. Byte-absorb pre-`p` data. Public IO is `BigUint`, not a
    //    `TranscriptReprTrait` type, so absorb its LE bytes directly.
    transcript.absorb_bytes(b"vk", &pk.vk_digest);
    for cw in &U.comm_w {
      transcript.absorb(b"comm_w", cw);
    }
    transcript.absorb(b"comm_q", &U.comm_q);
    for xi in &U.x {
      transcript.absorb_bytes(b"x", &xi.to_bytes_le());
    }

    // 3. Sample `p` from the transcript and switch typed-squeeze context.
    let (_sp_span, sp_t) = start_span!("imod_modp_sample_p");
    let params = M::sample_params(&mut transcript, log)?;
    transcript.set_params(params.clone());
    info!(elapsed_ms = %sp_t.elapsed().as_millis(), "imod_modp_sample_p");

    let shape = &pk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = MScalar::<M>::zero(&params);
    let one = MScalar::<M>::one(&params);

    // 4. Reduce shape/witness/IO from BigUint to M::Scalar mod p.
    let (_red_span, red_t) = start_span!("imod_modp_reduce");
    let mods_p = biguint_vec_to_scalars::<M>(&shape.mods, &params);
    let w_p = biguint_vec_to_scalars::<M>(&W.w, &params);
    let q_p = biguint_vec_to_scalars::<M>(&W.q, &params);
    let x_p = biguint_vec_to_scalars::<M>(&U.x, &params);
    let a_p = biguint_matrix_to_scalars::<M>(&shape.A, &params);
    let b_p = biguint_matrix_to_scalars::<M>(&shape.B, &params);
    let c_p = biguint_matrix_to_scalars::<M>(&shape.C, &params);
    info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_modp_reduce");

    // z = (W, 1, X), padded to 2*num_vars for the MLE.
    let (_spmv_span, spmv_t) = start_span!("imod_modp_spmv");
    let mut z = Vec::with_capacity(2 * num_vars);
    z.extend_from_slice(&w_p);
    z.push(one);
    z.extend_from_slice(&x_p);
    z.resize(2 * num_vars, zero);

    let z_for_spmv = &z[..num_vars + 1 + shape.num_io];
    let (az, bz, cz) = spmv::<M>(&a_p, &b_p, &c_p, z_for_spmv, num_cons, &params);
    info!(elapsed_ms = %spmv_t.elapsed().as_millis(), "imod_modp_spmv");

    // Outer sumcheck: sum_i eq(i, tau) · (Az·Bz − Cz − M·Q) = 0.
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let mut poly_az = MultilinearPolynomial::new(az, params.clone());
    let mut poly_bz = MultilinearPolynomial::new(bz, params.clone());
    let mut poly_cz = MultilinearPolynomial::new(cz, params.clone());
    let mut poly_m = MultilinearPolynomial::new(mods_p, params.clone());
    let mut poly_q = MultilinearPolynomial::new(q_p.clone(), params.clone());

    let (_so_span, so_t) = start_span!("imod_modp_outer_sumcheck");
    let (sc_outer, r_x, outer_claims) = SumcheckProof::<M>::prove_cubic_with_five_inputs(
      &zero,
      tau,
      &mut poly_az,
      &mut poly_bz,
      &mut poly_cz,
      &mut poly_m,
      &mut poly_q,
      &mut transcript,
    )?;
    info!(elapsed_ms = %so_t.elapsed().as_millis(), "imod_modp_outer_sumcheck");

    let v_a = outer_claims[0];
    let v_b = outer_claims[1];
    let v_c = outer_claims[2];
    let v_m = outer_claims[3];
    let v_q = outer_claims[4];

    transcript.absorb(b"outer_claims", &[v_a, v_b, v_c, v_m, v_q].as_slice());

    // Inner sumcheck: sum_y (A(r_x,y) + r·B(r_x,y) + r²·C(r_x,y)) · z(y).
    let r = transcript.squeeze(b"r")?;
    let claim_inner = v_a + r * v_b + r * r * v_c;

    let (_isetup_span, isetup_t) = start_span!("imod_modp_inner_setup");
    let eq_rx = EqPolynomial::<MScalar<M>>::evals_from_points(&r_x, &params);
    let abc = bind_abc::<M>(
      &a_p,
      &b_p,
      &c_p,
      num_vars,
      shape.num_io,
      &eq_rx,
      &r,
      &params,
    );
    info!(elapsed_ms = %isetup_t.elapsed().as_millis(), "imod_modp_inner_setup");

    debug_assert_eq!(abc.len(), 2 * num_vars);
    debug_assert_eq!(z.len(), 2 * num_vars);

    let mut poly_abc = MultilinearPolynomial::new(abc, params.clone());
    let mut poly_z = MultilinearPolynomial::new(z, params.clone());

    let (_si_span, si_t) = start_span!("imod_modp_inner_sumcheck");
    let (sc_inner, r_y, _claims_inner) = SumcheckProof::<M>::prove_quad(
      &claim_inner,
      num_rounds_y,
      &mut poly_abc,
      &mut poly_z,
      &mut transcript,
    )?;
    info!(elapsed_ms = %si_t.elapsed().as_millis(), "imod_modp_inner_sumcheck");

    // Recover eval_W from eval_Z via Z = (W, 1, X, …):
    //   Z(r_y) = (1 - r_y[0]) · W(r_y[1..]) + r_y[0] · pub(r_y[1..]).
    let (_er_span, er_t) = start_span!("imod_modp_eval_recover");
    let eval_z = poly_z[0];
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &x_p, &r_y[1..], &params);
    let one_minus_r0 = one - r_y[0];
    let inv = one_minus_r0.invert().ok_or(SpartanError::DivisionByZero)?;
    let eval_w = (eval_z - r_y[0] * eval_x) * inv;
    info!(elapsed_ms = %er_t.elapsed().as_millis(), "imod_modp_eval_recover");

    // Mod-PCS open W at r_y[1..] and Q at r_x in ONE batched argument:
    // both polynomials share a single LogUp-GKR range check and a single
    // combined inner-product opening, instead of paying that fixed
    // per-open cost twice. Mod-PCS commits/opens integers — pass the
    // original BigUint witness/quotient and the Z_p evals reduced into
    // BigUints in [0, p).
    let (_open_span, open_t) = start_span!("imod_modp_wq_open");
    let eval_w_bu = BigUint::from_bytes_le(&eval_w.to_le_bytes());
    let v_q_bu = BigUint::from_bytes_le(&v_q.to_le_bytes());
    let segs = pk.shape.width_segments();
    let mut seg_evals: Vec<MScalar<M>> = Vec::new();
    let eval_arg = if segs.is_empty() {
      <ModPCS<M> as ModPCSEngineTrait<M>>::prove_batch_with_blocks_rng(
        &pk.ck,
        &mut transcript,
        log,
        &[&U.comm_w[0], &U.comm_q],
        &[W.w.as_slice(), W.q.as_slice()],
        &[&W.r_w[0], &W.r_q],
        &[&r_y[1..], &r_x[..]],
        &[&eval_w_bu, &v_q_bu],
        &[pk.shape.small_blocks.as_slice(), &[]],
        rng,
      )?
    } else {
      // Width-grouped open: one poly per segment (at its own bound) plus Q.
      // eval_w is recovered on the verifier from the per-segment evals via
      // the selector sum, so it isn't opened directly. Small-value blocks
      // are not yet threaded through the segmented open (is_sat still
      // range-checks them); see docs/imod_followups.md.
      let wp = &r_y[1..];
      let q_ltf = <ModPCS<M> as ModPCSEngineTrait<M>>::commitment_log_t_f(&pk.ck);
      let mut locals: Vec<Vec<MScalar<M>>> = Vec::with_capacity(segs.len());
      let mut ev_bus: Vec<BigUint> = Vec::with_capacity(segs.len() + 1);
      let mut log_t_fs: Vec<usize> = Vec::with_capacity(segs.len() + 1);
      let mut slices: Vec<&[BigUint]> = Vec::with_capacity(segs.len() + 1);
      for seg in segs {
        let slice = &W.w[seg.start..seg.start + seg.size()];
        let local: Vec<MScalar<M>> = wp[wp.len() - seg.log_len..].to_vec();
        let seg_fq = biguint_vec_to_scalars::<M>(slice, &params);
        let seg_eval = MultilinearPolynomial::new(seg_fq, params.clone()).evaluate(&local);
        seg_evals.push(seg_eval);
        ev_bus.push(BigUint::from_bytes_le(&seg_eval.to_le_bytes()));
        log_t_fs.push(seg.log_t_f);
        slices.push(slice);
        locals.push(local);
      }
      ev_bus.push(v_q_bu.clone());
      log_t_fs.push(q_ltf);
      slices.push(W.q.as_slice());
      let comms: Vec<_> = U.comm_w.iter().chain(std::iter::once(&U.comm_q)).collect();
      let blinds: Vec<_> = W.r_w.iter().chain(std::iter::once(&W.r_q)).collect();
      let mut points: Vec<&[MScalar<M>]> = locals.iter().map(|v| v.as_slice()).collect();
      points.push(&r_x[..]);
      let ev_refs: Vec<&BigUint> = ev_bus.iter().collect();
      // 16-bit range-check blocks, offsets made relative to their segment;
      // Q carries none. Each block lies within exactly one segment.
      let seg_blocks = segment_relative_blocks(segs, &pk.shape.small_blocks);
      let mut blocks_ref: Vec<&[SmallValueBlock]> =
        seg_blocks.iter().map(|v| v.as_slice()).collect();
      blocks_ref.push(&[]);
      <ModPCS<M> as ModPCSEngineTrait<M>>::prove_batch_with_params_rng(
        &pk.ck,
        &mut transcript,
        log,
        &comms,
        &slices,
        &blinds,
        &points,
        &ev_refs,
        &blocks_ref,
        &log_t_fs,
        rng,
      )?
    };
    info!(elapsed_ms = %open_t.elapsed().as_millis(), "imod_modp_wq_open");

    info!(elapsed_ms = %prove_t.elapsed().as_millis(), "imod_spartan_modp_prove");
    Ok(Self {
      sc_outer,
      v_a,
      v_b,
      v_c,
      v_m,
      v_q,
      sc_inner,
      eval_w,
      seg_evals,
      eval_arg,
    })
  }

  /// [`verify`](Self::verify) with the complete prime-sampler audit log
  /// (the verifier-side counterpart of
  /// [`prove_with_prime_audit`](Self::prove_with_prime_audit)); for an
  /// honest proof the records equal the prover's.
  pub fn verify_with_prime_audit(
    &self,
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
  ) -> PrimeAuditedOutcome<()> {
    Self::run_audited(Self::prime_sampler_invocations_for_verify(vk), |log| {
      self.verify_inner(vk, U, log)
    })
  }

  /// Verify the SNARK against an instance. Wraps
  /// [`verify_with_prime_audit`](Self::verify_with_prime_audit) and
  /// discards only the diagnostic audit copy.
  pub fn verify(
    &self,
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
  ) -> Result<(), SpartanError> {
    self.verify_with_prime_audit(vk, U).into_result()
  }

  /// The verification flow against a caller-owned audit log.
  fn verify_inner(
    &self,
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    log: &mut PrimeAuditLog,
  ) -> Result<(), SpartanError> {
    let (_verify_span, verify_t) = start_span!("imod_spartan_modp_verify");

    // 1. Bootstrap transcript identically to prove().
    let mut transcript =
      Keccak256Transcript::<M>::new_with_params(b"IntModSpartanModpSNARK", M::bootstrap_params());

    // 2. Byte-absorb pre-`p` data identically to prove().
    transcript.absorb_bytes(b"vk", &vk.digest);
    for cw in &U.comm_w {
      transcript.absorb(b"comm_w", cw);
    }
    transcript.absorb(b"comm_q", &U.comm_q);
    for xi in &U.x {
      transcript.absorb_bytes(b"x", &xi.to_bytes_le());
    }

    // 3. Re-sample `p` from the same byte stream → identical params.
    let (_sp_span, sp_t) = start_span!("imod_modp_sample_p");
    let params = M::sample_params(&mut transcript, log)?;
    transcript.set_params(params.clone());
    info!(elapsed_ms = %sp_t.elapsed().as_millis(), "imod_modp_sample_p");

    // Context validation contract: every proof-carried dynamic-field
    // scalar must belong to the freshly sampled context. Anything that
    // changes the pre-`p` transcript (public IO, commitments, the vk
    // digest) can change the sampled prime, and mixing contexts panics
    // inside `crypto-bigint` — so reject BEFORE constructing `zero`/`one`
    // or performing any proof arithmetic.
    if !self.is_in_context(&params) {
      return Err(SpartanError::InvalidFieldContext);
    }

    let shape = &vk.shape;
    let num_vars = shape.num_vars;
    let num_cons = shape.num_cons;
    let num_rounds_x = num_cons.log_2();
    let num_rounds_y = num_vars.log_2() + 1;

    let zero = MScalar::<M>::zero(&params);
    let one = MScalar::<M>::one(&params);

    // 4. Reduce shape/IO from BigUint to M::Scalar mod p.
    let (_red_span, red_t) = start_span!("imod_modp_reduce");
    let mods_p = biguint_vec_to_scalars::<M>(&shape.mods, &params);
    let x_p = biguint_vec_to_scalars::<M>(&U.x, &params);
    let a_p = biguint_matrix_to_scalars::<M>(&shape.A, &params);
    let b_p = biguint_matrix_to_scalars::<M>(&shape.B, &params);
    let c_p = biguint_matrix_to_scalars::<M>(&shape.C, &params);
    info!(elapsed_ms = %red_t.elapsed().as_millis(), "imod_modp_reduce");

    // Outer SC verification.
    let (_so_span, so_t) = start_span!("imod_modp_outer_sumcheck");
    let tau: Vec<MScalar<M>> = (0..num_rounds_x)
      .map(|_| transcript.squeeze(b"tau"))
      .collect::<Result<Vec<_>, SpartanError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_outer
        .verify(zero, num_rounds_x, 3, &params, &mut transcript)?;

    // v_m matches the public mods MLE at r_x.
    let v_m_expected = dense_evaluate::<M>(&mods_p, &r_x, &params);
    if v_m_expected != self.v_m {
      return Err(SpartanError::InvalidSumcheckProof);
    }

    // Reconstruct the outer-SC final claim.
    let eq_tau_rx = EqPolynomial::<MScalar<M>>::new(tau, params.clone()).evaluate(&r_x);
    let outer_final_expected = eq_tau_rx * (self.v_a * self.v_b - self.v_c - self.v_m * self.v_q);
    if claim_outer_final != outer_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %so_t.elapsed().as_millis(), "imod_modp_outer_sumcheck");

    transcript.absorb(
      b"outer_claims",
      &[self.v_a, self.v_b, self.v_c, self.v_m, self.v_q].as_slice(),
    );

    // Inner SC verification.
    let (_si_span, si_t) = start_span!("imod_modp_inner_sumcheck");
    let r = transcript.squeeze(b"r")?;
    let claim_inner = self.v_a + r * self.v_b + r * r * self.v_c;

    let (claim_inner_final, r_y) =
      self
        .sc_inner
        .verify(claim_inner, num_rounds_y, 2, &params, &mut transcript)?;
    info!(elapsed_ms = %si_t.elapsed().as_millis(), "imod_modp_inner_sumcheck");

    // Reconstruct eval_Z from eval_W and public IO, then evaluate the
    // A/B/C MLEs at (r_x, r_y) via full eq tables (the O(2^n) verifier work).
    let (_em_span, em_t) = start_span!("imod_modp_eval_matrices");
    let eval_x = eval_public_at::<M>(num_rounds_y - 1, &x_p, &r_y[1..], &params);
    let eval_z = (one - r_y[0]) * self.eval_w + r_y[0] * eval_x;

    let t_x = EqPolynomial::<MScalar<M>>::evals_from_points(&r_x, &params);
    let t_y = EqPolynomial::<MScalar<M>>::evals_from_points(&r_y, &params);
    let (eval_a, eval_b, eval_c) = evaluate_matrices::<M>(&a_p, &b_p, &c_p, &t_x, &t_y, &params);

    let inner_final_expected = (eval_a + r * eval_b + r * r * eval_c) * eval_z;
    if claim_inner_final != inner_final_expected {
      return Err(SpartanError::InvalidSumcheckProof);
    }
    info!(elapsed_ms = %em_t.elapsed().as_millis(), "imod_modp_eval_matrices");

    // Mod-PCS verification for W (at r_y[1..]) and Q (at r_x) in ONE
    // batched argument, mirroring the prover's merged open.
    let (_wqver_span, wqver_t) = start_span!("imod_modp_wq_verify");
    let eval_w_bu = BigUint::from_bytes_le(&self.eval_w.to_le_bytes());
    let v_q_bu = BigUint::from_bytes_le(&self.v_q.to_le_bytes());
    let segs = vk.shape.width_segments();
    if segs.is_empty() {
      <ModPCS<M> as ModPCSEngineTrait<M>>::verify_batch_with_blocks(
        &vk.vk_ee,
        &mut transcript,
        log,
        &[&U.comm_w[0], &U.comm_q],
        &[&r_y[1..], &r_x[..]],
        &[&eval_w_bu, &v_q_bu],
        &self.eval_arg,
        &[vk.shape.small_blocks.as_slice(), &[]],
      )?;
    } else {
      // Bind the per-segment evals to the R1CS: eval_w must equal the
      // selector-weighted sum of the segment evaluations.
      if self.seg_evals.len() != segs.len() {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      let wp = &r_y[1..];
      let mut acc = MScalar::<M>::zero(&params);
      for (seg, se) in segs.iter().zip(self.seg_evals.iter()) {
        acc += segment_selector::<M>(seg.start, seg.log_len, wp, &params) * *se;
      }
      if acc != self.eval_w {
        return Err(SpartanError::InvalidSumcheckProof);
      }
      // Open each segment at its local point + Q at r_x.
      let q_ltf = <ModPCS<M> as ModPCSEngineTrait<M>>::verifier_log_t_f(&vk.vk_ee);
      let mut locals: Vec<Vec<MScalar<M>>> = Vec::with_capacity(segs.len());
      let mut ev_bus: Vec<BigUint> = Vec::with_capacity(segs.len() + 1);
      let mut log_t_fs: Vec<usize> = Vec::with_capacity(segs.len() + 1);
      for (seg, se) in segs.iter().zip(self.seg_evals.iter()) {
        locals.push(wp[wp.len() - seg.log_len..].to_vec());
        ev_bus.push(BigUint::from_bytes_le(&se.to_le_bytes()));
        log_t_fs.push(seg.log_t_f);
      }
      ev_bus.push(v_q_bu.clone());
      log_t_fs.push(q_ltf);
      let comms: Vec<_> = U.comm_w.iter().chain(std::iter::once(&U.comm_q)).collect();
      let mut points: Vec<&[MScalar<M>]> = locals.iter().map(|v| v.as_slice()).collect();
      points.push(&r_x[..]);
      let ev_refs: Vec<&BigUint> = ev_bus.iter().collect();
      let seg_blocks = segment_relative_blocks(segs, &vk.shape.small_blocks);
      let mut blocks_ref: Vec<&[SmallValueBlock]> =
        seg_blocks.iter().map(|v| v.as_slice()).collect();
      blocks_ref.push(&[]);
      <ModPCS<M> as ModPCSEngineTrait<M>>::verify_batch_with_params(
        &vk.vk_ee,
        &mut transcript,
        log,
        &comms,
        &points,
        &ev_refs,
        &self.eval_arg,
        &blocks_ref,
        &log_t_fs,
      )?;
    }
    info!(elapsed_ms = %wqver_t.elapsed().as_millis(), "imod_modp_wq_verify");

    info!(elapsed_ms = %verify_t.elapsed().as_millis(), "imod_spartan_modp_verify");
    Ok(())
  }
}

// ---------------------------------------------------------------------------
// helpers (operate on pre-reduced M::Scalar matrices)

fn spmv<M: ModEngine>(
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  z: &[MScalar<M>],
  num_cons: usize,
  params: &MParams<M>,
) -> (Vec<MScalar<M>>, Vec<MScalar<M>>, Vec<MScalar<M>>) {
  let zero = MScalar::<M>::zero(params);
  let multiply = |entries: &[(usize, usize, MScalar<M>)]| -> Vec<MScalar<M>> {
    let mut out = vec![zero; num_cons];
    for (i, j, v) in entries {
      out[*i] += *v * z[*j];
    }
    out
  };
  let (az, (bz, cz)) = rayon::join(
    || multiply(a),
    || rayon::join(|| multiply(b), || multiply(c)),
  );
  (az, bz, cz)
}

/// ABC[y] = sum_i eq_rx[i] · (A[i,y] + r·B[i,y] + r²·C[i,y]),
/// right-padded to length 2·num_vars to match the inner-SC layout.
fn bind_abc<M: ModEngine>(
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  num_vars: usize,
  num_io: usize,
  eq_rx: &[MScalar<M>],
  r: &MScalar<M>,
  params: &MParams<M>,
) -> Vec<MScalar<M>> {
  let zero = MScalar::<M>::zero(params);
  let num_cols = num_vars + 1 + num_io;
  let mut abc = vec![zero; num_cols];
  let r_sq = *r * *r;

  for (i, j, val) in a {
    abc[*j] += eq_rx[*i] * *val;
  }
  for (i, j, val) in b {
    abc[*j] += *r * eq_rx[*i] * *val;
  }
  for (i, j, val) in c {
    abc[*j] += r_sq * eq_rx[*i] * *val;
  }
  abc.resize(2 * num_vars, zero);
  abc
}

/// Multilinear extension of the public side `(1, x, 0, …, 0)` evaluated at `r`.
fn eval_public_at<M: ModEngine>(
  num_vars_pub: usize,
  x_p: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  debug_assert_eq!(r.len(), num_vars_pub);
  let zero = MScalar::<M>::zero(params);
  let one = MScalar::<M>::one(params);
  let mut pub_vec = Vec::with_capacity(1 << num_vars_pub);
  pub_vec.push(one);
  pub_vec.extend_from_slice(x_p);
  pub_vec.resize(1 << num_vars_pub, zero);
  dense_evaluate::<M>(&pub_vec, r, params)
}

/// Dense MLE evaluation: sum_k chi_r[k] · z[k].
fn dense_evaluate<M: ModEngine>(
  z: &[MScalar<M>],
  r: &[MScalar<M>],
  params: &MParams<M>,
) -> MScalar<M> {
  let zero = MScalar::<M>::zero(params);
  let chis = EqPolynomial::<MScalar<M>>::evals_from_points(r, params);
  debug_assert_eq!(chis.len(), z.len());
  chis
    .par_iter()
    .zip(z.par_iter())
    .map(|(c, v)| *c * *v)
    .reduce(|| zero, |a, b| a + b)
}

/// Evaluate A, B, C MLEs at (r_x, r_y) via precomputed eq-tables.
fn evaluate_matrices<M: ModEngine>(
  a: &[(usize, usize, MScalar<M>)],
  b: &[(usize, usize, MScalar<M>)],
  c: &[(usize, usize, MScalar<M>)],
  t_x: &[MScalar<M>],
  t_y: &[MScalar<M>],
  params: &MParams<M>,
) -> (MScalar<M>, MScalar<M>, MScalar<M>) {
  let zero = MScalar::<M>::zero(params);
  let eval_one = |entries: &[(usize, usize, MScalar<M>)]| -> MScalar<M> {
    entries
      .iter()
      .map(|(i, j, v)| t_x[*i] * t_y[*j] * *v)
      .fold(zero, |a, b| a + b)
  };
  let (eval_a, (eval_b, eval_c)) = rayon::join(
    || eval_one(a),
    || rayon::join(|| eval_one(b), || eval_one(c)),
  );
  (eval_a, eval_b, eval_c)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::T256DynPrimeEngine;

  type ME = T256DynPrimeEngine;

  /// Toy: prove `a · b ≡ c (mod N)` over an arbitrary verifier-sampled
  /// prime `p`. Witness layout `w = [a, b, c, 0]`. One real row + one
  /// padding row to make `num_cons` a power of two.
  fn build_toy(
    a: u64,
    b: u64,
    c: u64,
    n: u64,
    q: u64,
  ) -> (IntModR1CSShapeModp<ME>, Vec<BigUint>, Vec<BigUint>) {
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);

    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(n), zero.clone()];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w = vec![
      BigUint::from(a),
      BigUint::from(b),
      BigUint::from(c),
      zero.clone(),
    ];
    let q_vec = vec![BigUint::from(q), zero];
    (shape, w, q_vec)
  }

  /// End-to-end: 3 · 5 ≡ 1 (mod 14) under a transcript-sampled prime `p`
  /// that is *not* the curve scalar prime. Validates the dual-field
  /// driver flow against the trivial Mod-PCS stub.
  #[test]
  fn imod_modp_toy_roundtrip() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// The same toy circuit through the Brakedown-backed engine: same
  /// protocol, hash commitments end to end.
  #[test]
  fn imod_modp_bd_toy_roundtrip() {
    type BE = crate::provider::T256DynPrimeBdEngine;
    let num_cons = 2usize;
    let num_vars = 4usize;
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u64), zero.clone()];
    let shape =
      IntModR1CSShapeModp::<BE>::new(num_cons, num_vars, 0, mat_a, mat_b, mat_c, mods).unwrap();
    let w = vec![
      BigUint::from(3u64),
      BigUint::from(5u64),
      BigUint::from(1u64),
      zero.clone(),
    ];
    let q = vec![BigUint::from(1u64), zero];
    let (pk, vk) = IntModSpartanModpSNARK::<BE>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<BE>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();
    let proof = IntModSpartanModpSNARK::<BE>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();

    // Tampering the witness commitment (a Merkle root byte) must break
    // verification via the transcript binding.
    let mut bad_u = U.clone();
    bad_u.comm_w[0].root[0] ^= 1;
    assert!(proof.verify(&vk, &bad_u).is_err());
  }

  /// THE small-field smoke test: the same toy circuit proved and
  /// verified entirely over F127 (mod 2^127 − 1) with Brakedown
  /// commitments — no 256-bit field, no curve, anywhere in the flow.
  /// Parameters derived at log_q = 127 under the accepted
  /// challenge-soundness target.
  #[test]
  fn imod_modp_m127_toy_roundtrip() {
    type SE = crate::provider::M127DynPrimeBdEngine;
    let num_cons = 2usize;
    let num_vars = 4usize;
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u64), zero.clone()];
    let shape =
      IntModR1CSShapeModp::<SE>::new(num_cons, num_vars, 0, mat_a, mat_b, mat_c, mods).unwrap();
    let w = vec![
      BigUint::from(3u64),
      BigUint::from(5u64),
      BigUint::from(1u64),
      zero.clone(),
    ];
    let q = vec![BigUint::from(1u64), zero];
    let params =
      crate::provider::pcs::integer_modpcs::IntEvalParams::derive_for_q(127, 16, 16, 2, 2).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<SE>::setup_with_params(shape.clone(), params).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<SE>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();
    let proof = IntModSpartanModpSNARK::<SE>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();

    // Same Merkle-root tamper check as the t256 Brakedown test.
    let mut bad_u = U.clone();
    bad_u.comm_w[0].root[0] ^= 1;
    assert!(proof.verify(&vk, &bad_u).is_err());
  }

  /// `is_sat` rejects an inconsistent witness (wrong quotient).
  #[test]
  fn imod_modp_bad_witness_rejected() {
    let (shape, w, _) = build_toy(3, 5, 1, 14, 1);
    let bad_q = vec![BigUint::from(0u32), BigUint::from(0u32)];
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, bad_q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &U, &W).is_err());
  }

  /// Tampering with `v_q` breaks transcript binding inside the SNARK
  /// driver (independent of Mod-PCS soundness).
  #[test]
  fn imod_modp_verify_rejects_tampering() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let mut proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    // The sampled p is engine-dependent; use M::Scalar::one() against the
    // tampered v_q by re-running the same sampling deterministically.
    let mut t = Keccak256Transcript::<ME>::new_with_params(
      b"IntModSpartanModpSNARK",
      <ME as ModEngine>::bootstrap_params(),
    );
    t.absorb_bytes(b"vk", &pk.vk_digest);
    for cw in &U.comm_w {
      t.absorb(b"comm_w", cw);
    }
    t.absorb(b"comm_q", &U.comm_q);
    let mut log = PrimeAuditLog::with_expected(1).unwrap();
    let params = <ME as ModEngine>::sample_params(&mut t, &mut log).unwrap();
    proof.v_q += MScalar::<ME>::one(&params);
    assert!(proof.verify(&vk, &U).is_err());
  }

  /// Two real constraints with different moduli, exercising the outer SC
  /// on more than one active row.
  ///   row 0: 3·5 ≡ 1 (mod 14), q₀ = 1 (since 15 = 1 + 14·1)
  ///   row 1: 7·9 ≡ 3 (mod 20), q₁ = 3 (since 63 = 3 + 20·3)
  #[test]
  fn imod_modp_two_row_roundtrip() {
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 8usize;
    let num_io = 0usize;

    // Layout w = [a1, b1, c1, a2, b2, c2, 0, 0]
    let mat_a = vec![(0, 0, one.clone()), (1, 3, one.clone())];
    let mat_b = vec![(0, 1, one.clone()), (1, 4, one.clone())];
    let mat_c = vec![(0, 2, one.clone()), (1, 5, one)];
    let mods = vec![BigUint::from(14u32), BigUint::from(20u32)];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w: Vec<BigUint> = [3u32, 5, 1, 7, 9, 3, 0, 0]
      .iter()
      .map(|x| BigUint::from(*x))
      .collect();
    let q: Vec<BigUint> = [1u32, 3].iter().map(|x| BigUint::from(*x)).collect();

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let _ = zero; // silence unused if test layout changes
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Modulus-0 rows are EXACT integer rows: the `m·q` term vanishes, so
  /// the row enforces `LC_A·LC_B = LC_C` over ℤ. Roundtrips a shape
  /// mixing an exact bit row (`b·b = b`, mod 0) with an ordinary mod-14
  /// row, exercising the verifier's mods-MLE path with zero entries.
  #[test]
  fn imod_modp_exact_row_mod_zero_roundtrip() {
    let one = BigUint::from(1u32);
    let num_cons = 4usize;
    let num_vars = 8usize;

    // Row 0: w[0]*w[1] = w[2] (mod 14); row 1: w[3]*w[3] = w[3] (mod 0),
    // the exact bit constraint.
    let mat_a = vec![(0, 0, one.clone()), (1, 3, one.clone())];
    let mat_b = vec![(0, 1, one.clone()), (1, 3, one.clone())];
    let mat_c = vec![(0, 2, one.clone()), (1, 3, one.clone())];
    let mods = vec![
      BigUint::from(14u32),
      BigUint::from(0u32),
      BigUint::from(2u32),
      BigUint::from(2u32),
    ];
    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, mat_a, mat_b, mat_c, mods).unwrap();

    let w: Vec<BigUint> = [3u32, 5, 1, 1, 0, 0, 0, 0]
      .iter()
      .map(|x| BigUint::from(*x))
      .collect();
    let q: Vec<BigUint> = [1u32, 0, 0, 0].iter().map(|x| BigUint::from(*x)).collect();

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &instance, &witness).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
  }

  /// The exact bit row rejects non-binary values that a mod-N bit row
  /// would accept: `b = N+1` satisfies `b² ≡ b (mod N)` with quotient
  /// `q = N+1` (a benign lift), but `b² ≠ b` over ℤ, so the mod-0 row
  /// is unsatisfiable.
  #[test]
  fn imod_modp_exact_bit_row_rejects_lift() {
    let one = BigUint::from(1u32);
    let n = BigUint::from(77u32); // stand-in composite "N"
    let num_cons = 2usize;
    let num_vars = 2usize;

    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 0, one.clone())];
    let mat_c = vec![(0, 0, one.clone())];
    let mods = vec![BigUint::from(0u32), BigUint::from(2u32)];
    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, mat_a, mat_b, mat_c, mods).unwrap();

    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();

    // b = N+1 with q = N+1 satisfies the mod-N version of this row; the
    // mod-0 row must reject it regardless of the quotient.
    let b = &n + &one;
    let w = vec![b.clone(), BigUint::from(0u32)];
    let q = vec![b, BigUint::from(0u32)];
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &instance, &witness).is_err());
  }

  /// Small-value blocks: an aligned witness block asserted `< 2^16` by
  /// the Mod-PCS with no rows. In-range values round-trip; a value
  /// `≥ 2^16` inside the block is rejected even though every row is
  /// satisfied over ℤ (the block variables are row-unconstrained).
  /// Width-scaling measurement (the width-grouping optimization's basis):
  /// the SAME narrow-value circuit proved at log_t_f=2048 (128 chunks/
  /// value) vs log_t_f=256 (16 chunks/value). Confirms commit + prove
  /// scale with committed chunk count, grounding the projection that
  /// committing narrow values at a narrow bound is proportionally faster.
  /// Run: `cargo test --release width_scaling -- --ignored --nocapture`.
  #[test]
  #[ignore]
  fn width_scaling_measurement() {
    use crate::provider::pcs::integer_modpcs::{DEFAULT_K, IntEvalParams};
    use std::time::Instant;
    // Narrow circuit: n rows of a*b = c mod (256-bit prime), all values < 2^256.
    let log_n = 12usize;
    let num_cons = 1usize << log_n;
    let num_vars = 1usize << log_n;
    let one = BigUint::from(1u32);
    // a 250-bit prime-ish modulus (values stay < 2^256)
    let m: BigUint = (BigUint::from(1u32) << 250) - BigUint::from(1u32);
    // One column per row: w[i]·1 = w[i] (mod m). Satisfied for any w[i] < m,
    // quotient 0 — a clean narrow-witness carrier for the timing measurement.
    let const_col = num_vars;
    let a_e: Vec<_> = (0..num_cons).map(|i| (i, i, one.clone())).collect();
    let b_e: Vec<_> = (0..num_cons).map(|i| (i, const_col, one.clone())).collect();
    let c_e: Vec<_> = (0..num_cons).map(|i| (i, i, one.clone())).collect();
    let mods = vec![m.clone(); num_cons];
    let shape = IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, a_e, b_e, c_e, mods).unwrap();
    // narrow witness: each w[i] a ~250-bit value < m.
    let mut w = vec![BigUint::from(0u32); num_vars];
    for (i, wi) in w.iter_mut().enumerate() {
      *wi = ((BigUint::from(0x9e37_79b9_7f4a_7c15u64) << 180) | BigUint::from(i as u64 + 1)) % &m;
    }
    let q = vec![BigUint::from(0u32); num_cons];

    for log_t_f in [2048usize, 256] {
      let params = IntEvalParams::derive(log_t_f, 64, DEFAULT_K, log_n).unwrap();
      let (pk, vk) =
        IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
      let t0 = Instant::now();
      let (witness, instance) =
        IntModR1CSWitnessModp::<ME>::new(&shape, pk.ck(), w.clone(), q.clone(), vec![]).unwrap();
      let commit_ms = t0.elapsed().as_secs_f64() * 1e3;
      let t1 = Instant::now();
      let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
      let prove_ms = t1.elapsed().as_secs_f64() * 1e3;
      proof.verify(&vk, &instance).unwrap();
      eprintln!(
        "log_t_f={log_t_f:<4} (numlimb={:2}): commit {commit_ms:7.1} ms, prove {prove_ms:7.1} ms",
        log_t_f.div_ceil(64)
      );
    }
  }

  #[test]
  fn imod_modp_small_value_block_roundtrip_and_rejects() {
    use crate::traits::mod_engine::SmallValueBlock;
    let one = BigUint::from(1u32);
    let num_cons = 4usize;
    let num_vars = 8usize;
    // Row 0: w[0]·w[1] = w[2] exactly; block [4, 8) is row-free.
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one.clone())];
    let mods = vec![
      BigUint::from(0u32),
      BigUint::from(2u32),
      BigUint::from(2u32),
      BigUint::from(2u32),
    ];
    let shape = IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, 0, mat_a, mat_b, mat_c, mods)
      .unwrap()
      .with_small_value_blocks(vec![SmallValueBlock {
        start: 4,
        log_len: 2,
      }])
      .unwrap();
    let mk = |v4: u64| -> (Vec<BigUint>, Vec<BigUint>) {
      let w = [3u64, 5, 15, 0, v4, 7, 65535, 0]
        .iter()
        .map(|x| BigUint::from(*x))
        .collect();
      let q = vec![BigUint::from(0u32); num_cons];
      (w, q)
    };
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();

    let (w, q) = mk(65535);
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &instance, &witness).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();

    // Out of range: rows still hold, `is_sat` and the verifier both reject.
    let (w, q) = mk(65536);
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    assert!(shape.is_sat(&pk.ck, &instance, &witness).is_err());
    // The prover does not self-check blocks; the verifier must reject.
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    assert!(proof.verify(&vk, &instance).is_err());
  }

  /// Wired circuit: the output of row 0 feeds into the input of row 1.
  /// w[2] is shared between C of row 0 and A of row 1.
  #[test]
  fn imod_modp_wired_two_row_roundtrip() {
    let one = BigUint::from(1u32);
    let num_cons = 4usize;
    let num_vars = 8usize;
    let num_io = 0usize;

    // Row 0: w[0]*w[1] = w[2] (mod 14) → 3*5 = 15 = 1 + 14*1
    // Row 1: w[2]*w[3] = w[4] (mod 14) → 1*4 =  4 = 4 + 14*0
    // w[2] is SHARED: output of row 0, input to row 1.
    let mat_a = vec![(0, 0, one.clone()), (1, 2, one.clone())];
    let mat_b = vec![(0, 1, one.clone()), (1, 3, one.clone())];
    let mat_c = vec![(0, 2, one.clone()), (1, 4, one.clone())];
    let mods = vec![
      BigUint::from(14u32),
      BigUint::from(14u32),
      BigUint::from(2u32),
      BigUint::from(2u32),
    ];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w: Vec<BigUint> = [3u32, 5, 1, 4, 4, 0, 0, 0]
      .iter()
      .map(|x| BigUint::from(*x))
      .collect();
    let q: Vec<BigUint> = [1u32, 0, 0, 0].iter().map(|x| BigUint::from(*x)).collect();

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &instance, &witness).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
  }

  /// Wired circuit at 2048-bit width with setup_with_params, matching the
  /// multiswap_modp bench configuration: RSA-2048 modulus, numlimb=64.
  #[test]
  fn imod_modp_wired_rsa2048_roundtrip() {
    use crate::provider::pcs::integer_modpcs::IntEvalParams;
    use num_integer::Integer;

    let one = BigUint::from(1u32);
    let n_hex = "c7970ceedcc3b0754490201a7aa613cd73911081c790f5f1a8726f463550bb5b\
                 7ff0db8e1ea1189ec72f93d1650011bd721aeeacc2acde32a04107f0648c2813\
                 a31f5b0b7765ff8b44b4b6ffc93384b646eb09c7cf5e8592d40ea33c80039f35\
                 b4f14a04b51f7bfd781be4d1673164ba8eb991c2c4d730bbbe35f592bdef524a\
                 f7e8daefd26c66fc02c479af89d64d373f442709439de66ceb955f3ea37d5159\
                 f6135809f85334b5cb1813addc80cd05609f10ac6a95ad65872c909525bdad32\
                 bc729592642920f24c61dc5b3c3b7923e56b16a4d9d373d8721f24a3fc0f1b31\
                 31f55615172866bccc30f95054c824e733a5eb6817f7bc16399d48c6361cc7e5";
    let n = BigUint::parse_bytes(n_hex.as_bytes(), 16).unwrap();

    let num_cons = 4usize;
    let num_vars = 8usize;
    let num_io = 0usize;

    // Row 0: w[0]*w[1] = w[2] (mod N)
    // Row 1: w[2]*w[3] = w[4] (mod N) — w[2] is SHARED (wired)
    let mat_a = vec![(0, 0, one.clone()), (1, 2, one.clone())];
    let mat_b = vec![(0, 1, one.clone()), (1, 3, one.clone())];
    let mat_c = vec![(0, 2, one.clone()), (1, 4, one.clone())];
    let mods = vec![
      n.clone(),
      n.clone(),
      BigUint::from(2u32),
      BigUint::from(2u32),
    ];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    // a0, b0 close to N; c0 = a0*b0 mod N; then c0*b1 mod N.
    let a0 = &n - BigUint::from(3u32);
    let b0 = &n - BigUint::from(7u32);
    let prod0 = &a0 * &b0;
    let (q0, c0) = prod0.div_rem(&n);
    let b1 = &n - BigUint::from(11u32);
    let prod1 = &c0 * &b1;
    let (q1, c1) = prod1.div_rem(&n);

    let mut w: Vec<BigUint> = vec![BigUint::from(0u32); num_vars];
    w[0] = a0;
    w[1] = b0;
    w[2] = c0;
    w[3] = b1;
    w[4] = c1;
    let q: Vec<BigUint> = vec![q0, q1, BigUint::from(0u32), BigUint::from(0u32)];

    let log_n = (num_vars.max(num_cons) as u64).ilog2() as usize;
    let params = IntEvalParams::derive(2048, 32, 10, log_n).unwrap();

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
    let (witness, instance) =
      IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &instance, &witness).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    proof.verify(&vk, &instance).unwrap();
  }

  /// End-to-end SNARK roundtrip that triggers the IntEval partial-eval
  /// iteration path (step C) on the W open. With `num_vars = 256`, the
  /// Mod-PCS opens W at a point of length `log_2(256) = 8 > k = 7`
  /// (pinned explicitly below), so `t = 1` partial-eval iteration runs
  /// per small prime. The Q open at length 1 still uses the step-B
  /// path (no iteration). Both must agree end-to-end through the
  /// SNARK protocol.
  ///
  /// `k = 7` is pinned via `setup_with_params` because the default
  /// setup now optimizes params per input length and may pick
  /// `k = point.len` here, which would silently skip the iteration
  /// path this test exists to cover.
  #[test]
  fn imod_modp_snark_with_inteval_iteration() {
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 256usize; // log_2(256) = 8 > pinned k = 7
    let num_io = 0usize;

    // One real row: 3·5 ≡ 1 (mod 14), q₀ = 1. Layout: w[0..3] = [3, 5, 1],
    // rest zero (253 trailing zeros).
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u32), zero.clone()];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let mut w: Vec<BigUint> = vec![zero.clone(); num_vars];
    w[0] = BigUint::from(3u32);
    w[1] = BigUint::from(5u32);
    w[2] = BigUint::from(1u32);
    let q: Vec<BigUint> = vec![BigUint::from(1u32), zero];

    // Pin k = 7 < log_2(num_vars) = 8 so the iteration path runs.
    let params = crate::provider::pcs::integer_modpcs::IntEvalParams::derive_no_limb_split(
      crate::provider::pcs::integer_modpcs::DEFAULT_LOG_T_F,
      7,
      8,
    )
    .unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Public IO: a tiny circuit `w₀ · w₁ ≡ x₀ (mod 14)` with `x₀ = 1` as
  /// public input. Exercises the `eval_public_at` path that the
  /// zero-IO toy doesn't touch.
  #[test]
  fn imod_modp_with_public_io() {
    let one = BigUint::from(1u32);
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 1usize;
    // num_cols = num_vars + 1 + num_io = 4 + 1 + 1 = 6
    // columns 0..4 = w, column 4 = 1 (constant), column 5 = x[0]
    let mat_a = vec![(0, 0, one.clone())]; // selects w[0] = 3
    let mat_b = vec![(0, 1, one.clone())]; // selects w[1] = 5
    let mat_c = vec![(0, 5, one)]; // selects x[0] = 1
    let mods = vec![BigUint::from(14u32), BigUint::from(0u32)];

    let shape =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a, mat_b, mat_c, mods)
        .unwrap();

    let w: Vec<BigUint> = [3u32, 5, 0, 0].iter().map(|x| BigUint::from(*x)).collect();
    let q: Vec<BigUint> = [1u32, 0].iter().map(|x| BigUint::from(*x)).collect();
    let x: Vec<BigUint> = vec![BigUint::from(1u32)];

    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, x).unwrap();
    shape.is_sat(&pk.ck, &U, &W).unwrap();

    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();
    proof.verify(&vk, &U).unwrap();
  }

  /// Shapes with the same dimensions/mods but different `A` entries must
  /// produce distinct verifier-key digests. Distinct digests are what
  /// makes vk-cross-binding work: the transcript binds `vk_digest` first,
  /// so swapping vks deterministically derives different `p` and the
  /// rest of the proof becomes incoherent under the wrong vk.
  ///
  /// (Known gap: today, verifying a proof under the wrong vk
  /// *panics* inside `crypto-bigint`'s `FixedMontyForm` op when the
  /// proof's `DynPrime` values carry params from the original `p` while
  /// the verifier reduces shape2 data with the freshly sampled `p`.
  /// The panic IS a form of rejection, but it's ungraceful — the param
  /// mismatch should eventually surface as a clean `SpartanError`.
  /// For now this test asserts only the digest distinction.)
  #[test]
  fn imod_modp_digest_binds_matrices() {
    let one = BigUint::from(1u32);
    let two = BigUint::from(2u32);
    let zero = BigUint::from(0u32);
    let num_cons = 2usize;
    let num_vars = 4usize;
    let num_io = 0usize;

    let mat_a1 = vec![(0, 0, one.clone())];
    let mat_a2 = vec![(0, 0, two)];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u32), zero];

    let shape1 = IntModR1CSShapeModp::<ME>::new(
      num_cons,
      num_vars,
      num_io,
      mat_a1,
      mat_b.clone(),
      mat_c.clone(),
      mods.clone(),
    )
    .unwrap();
    let shape2 =
      IntModR1CSShapeModp::<ME>::new(num_cons, num_vars, num_io, mat_a2, mat_b, mat_c, mods)
        .unwrap();

    let (_, vk1) = IntModSpartanModpSNARK::<ME>::setup(shape1).unwrap();
    let (_, vk2) = IntModSpartanModpSNARK::<ME>::setup(shape2).unwrap();
    assert_ne!(vk1.digest(), vk2.digest());
  }

  /// The toy multi-IO circuit `w₀ · w₁ ≡ x₀ (mod 14)` with three public
  /// values (x₁ and x₂ are transcript-bound but unconstrained), mirroring
  /// the combined Poseidon circuit's three-digest statement. Returns
  /// `(pk, vk, witness, instance, proof)` with `x = [x0, 2, 3]`.
  fn public_io_fixture(
    x0: u64,
  ) -> (
    IntModSpartanModpProverKey<ME>,
    IntModSpartanModpVerifierKey<ME>,
    IntModR1CSWitnessModp<ME>,
    IntModR1CSInstanceModp<ME>,
    IntModSpartanModpSNARK<ME>,
  ) {
    let one = BigUint::from(1u32);
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 5, one)];
    let mods = vec![BigUint::from(14u32), BigUint::from(0u32)];
    let shape = IntModR1CSShapeModp::<ME>::new(2, 4, 3, mat_a, mat_b, mat_c, mods).unwrap();
    // 3 · 5 = 15 ≡ 1 (mod 14): x0 = 1, q0 = 1.
    let w: Vec<BigUint> = [3u32, 5, 0, 0].iter().map(|v| BigUint::from(*v)).collect();
    let q: Vec<BigUint> = [1u32, 0].iter().map(|v| BigUint::from(*v)).collect();
    let x = vec![BigUint::from(x0), BigUint::from(2u32), BigUint::from(3u32)];
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (witness, instance) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, x).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &instance, &witness).unwrap();
    (pk, vk, witness, instance, proof)
  }

  /// In-range wrong-digest context regression: changing only ONE public
  /// digest to another canonical value — holding the other two fixed —
  /// changes the pre-`p` transcript and (for the searched fixture) the
  /// sampled runtime prime; verification must return
  /// `Err(InvalidFieldContext)` — never panic — in BOTH debug and
  /// release. The search avoids assuming that every changed transcript
  /// must sample a different prime.
  #[test]
  fn imod_modp_rejects_wrong_digest_context() {
    let (_pk, vk, _witness, instance, proof) = public_io_fixture(1);
    proof.verify(&vk, &instance).unwrap();
    let proof_ctx = IntModSpartanModpSNARK::<ME>::resample_params(&vk, &instance);

    let modulus = 14u64;
    let mut tampered = None;
    for delta in 1..modulus {
      let mut candidate = instance.clone();
      // Only x[0] changes; x[1] and x[2] stay fixed.
      candidate.x[0] = BigUint::from((1 + delta) % modulus);
      assert_eq!(candidate.x[1..], instance.x[1..]);
      let ctx = IntModSpartanModpSNARK::<ME>::resample_params(&vk, &candidate);
      if ctx.modulus() != proof_ctx.modulus() {
        tampered = Some(candidate);
        break;
      }
    }
    let tampered = tampered.expect("some canonical x[0] tweak must resample a different prime");
    assert_eq!(
      proof.verify(&vk, &tampered),
      Err(SpartanError::InvalidFieldContext)
    );
  }

  /// Wrong-verifier-key context regression: a valid proof cross-bound to
  /// a different shape's key re-samples a different prime (asserted
  /// explicitly) and gets the same clean rejection instead of the old
  /// `crypto-bigint` panic.
  #[test]
  fn imod_modp_rejects_wrong_vk_context() {
    let (_pk, vk, _witness, instance, proof) = public_io_fixture(1);
    proof.verify(&vk, &instance).unwrap();
    let proof_ctx = IntModSpartanModpSNARK::<ME>::resample_params(&vk, &instance);

    // Same dimensions, different A coefficient → different digest.
    let one = BigUint::from(1u32);
    let mat_a = vec![(0, 0, BigUint::from(2u32))];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 5, one)];
    let mods = vec![BigUint::from(14u32), BigUint::from(0u32)];
    let shape2 = IntModR1CSShapeModp::<ME>::new(2, 4, 3, mat_a, mat_b, mat_c, mods).unwrap();
    let (_pk2, vk2) = IntModSpartanModpSNARK::<ME>::setup(shape2).unwrap();
    assert_ne!(vk.digest(), vk2.digest());
    let cross_ctx = IntModSpartanModpSNARK::<ME>::resample_params(&vk2, &instance);
    assert_ne!(
      cross_ctx.modulus(),
      proof_ctx.modulus(),
      "fixture requires the cross-bound key to re-sample a different prime"
    );
    assert_eq!(
      proof.verify(&vk2, &instance),
      Err(SpartanError::InvalidFieldContext)
    );
  }

  /// Mixed-context traversal coverage: flipping ANY dynamic-field scalar
  /// of the proof into a foreign modulus context must be caught by
  /// `is_in_context`, so a future proof field cannot be omitted from the
  /// traversal silently.
  #[test]
  fn imod_modp_context_traversal_covers_every_field() {
    use crypto_bigint::{Odd, U128, modular::FixedMontyParams};
    let (_pk, vk, _witness, instance, proof) = public_io_fixture(1);
    let ctx = IntModSpartanModpSNARK::<ME>::resample_params(&vk, &instance);
    assert!(proof.is_in_context(&ctx));

    // A foreign context: 2^127 − 1 never equals the sampled 128-bit prime
    // (whose top bit is forced set).
    let foreign = FixedMontyParams::new(
      Odd::new(U128::from_be_hex("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")).unwrap(),
    );
    let bad = MScalar::<ME>::one(&foreign);
    assert!(!bad.is_in_context(&ctx));

    let mutations: Vec<fn(&mut IntModSpartanModpSNARK<ME>, MScalar<ME>)> = vec![
      |p, v| p.sc_outer.compressed_polys[0].coeffs_except_linear_term[0] = v,
      |p, v| p.v_a = v,
      |p, v| p.v_b = v,
      |p, v| p.v_c = v,
      |p, v| p.v_m = v,
      |p, v| p.v_q = v,
      |p, v| p.sc_inner.compressed_polys[0].coeffs_except_linear_term[0] = v,
      |p, v| p.eval_w = v,
    ];
    for (i, mutate) in mutations.iter().enumerate() {
      let mut tampered = proof.clone();
      mutate(&mut tampered, bad);
      assert!(
        !tampered.is_in_context(&ctx),
        "traversal branch {i} missed a foreign-context scalar"
      );
    }
    // The Mod-PCS hook branch: the IntEval batch argument carries no
    // dynamic-field scalars, so the delegated check accepts it.
    assert!(<ModPCS<ME> as ModPCSEngineTrait<ME>>::batch_arg_is_in_context(&proof.eval_arg, &ctx));
  }

  /// The canonical-serialization helper surfaces serializer failures as
  /// `SerializationError` (never as a zero size): exercised with a
  /// deliberately failing `Serialize` implementation.
  #[test]
  fn imod_modp_canonical_bytes_error_path() {
    struct FailingSerialize;
    impl serde::Serialize for FailingSerialize {
      fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom("deliberate test failure"))
      }
    }
    match to_canonical_bytes(&FailingSerialize) {
      Err(SpartanError::SerializationError { reason }) => {
        assert!(reason.contains("deliberate test failure"));
      }
      other => panic!("expected SerializationError, got {other:?}"),
    }
    // And the success path produces the pinned little-endian fixint
    // encoding for a simple value.
    assert_eq!(
      to_canonical_bytes(&1u64).unwrap(),
      vec![1, 0, 0, 0, 0, 0, 0, 0]
    );
  }

  /// The proof-size byte serializers produce nonempty canonical bytes for
  /// a real proof and instance.
  #[test]
  fn imod_modp_proof_size_bytes_roundtrip() {
    let (_pk, vk, _witness, instance, proof) = public_io_fixture(1);
    proof.verify(&vk, &instance).unwrap();
    let arg = proof.eval_arg_bytes().unwrap();
    assert!(!arg.is_empty());
    let comms = instance.commitment_bytes().unwrap();
    assert!(!comms.is_empty());
    // The tuple encoding is exactly the two commitments' canonical bytes,
    // consecutively, with no outer tag or length.
    let cw = to_canonical_bytes(&instance.comm_w).unwrap();
    let cq = to_canonical_bytes(&instance.comm_q).unwrap();
    assert_eq!(comms, [cw, cq].concat());
  }

  /// Seeded prover coins reproduce the proof: two witness commitments and
  /// proofs from identical `ChaCha20Rng` seeds yield identical
  /// commitment bytes, evaluation-argument bytes, remainder bytes and
  /// audit logs (the benchmark's double-construction invariant), while
  /// a different seed changes the Hyrax commitments.
  #[test]
  fn imod_modp_prove_with_rng_is_reproducible() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    let one = BigUint::from(1u32);
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 5, one)];
    let mods = vec![BigUint::from(14u32), BigUint::from(0u32)];
    let shape = IntModR1CSShapeModp::<ME>::new(2, 4, 3, mat_a, mat_b, mat_c, mods).unwrap();
    let w: Vec<BigUint> = [3u32, 5, 0, 0].iter().map(|v| BigUint::from(*v)).collect();
    let q: Vec<BigUint> = [1u32, 0].iter().map(|v| BigUint::from(*v)).collect();
    let x = vec![
      BigUint::from(1u32),
      BigUint::from(2u32),
      BigUint::from(3u32),
    ];
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let construct = |seed: [u8; 32]| {
      let mut rng = ChaCha20Rng::from_seed(seed);
      let (witness, instance) = IntModR1CSWitnessModp::<ME>::new_with_rng(
        &shape,
        &pk.ck,
        w.clone(),
        q.clone(),
        x.clone(),
        &mut rng,
      )
      .unwrap();
      let (proof, records) = match IntModSpartanModpSNARK::<ME>::prove_with_prime_audit_rng(
        &pk, &instance, &witness, &mut rng,
      ) {
        PrimeAuditedOutcome::Success { value, records } => (value, records),
        PrimeAuditedOutcome::Failure { source, .. } => panic!("prove failed: {source:?}"),
      };
      proof.verify(&vk, &instance).unwrap();
      (
        instance.commitment_bytes().unwrap(),
        proof.eval_arg_bytes().unwrap(),
        proof.sumcheck_remainder_bytes(),
        proof.sumcheck_remainder_counts(),
        records,
      )
    };
    let a = construct([7u8; 32]);
    let b = construct([7u8; 32]);
    assert_eq!(a, b, "identical seeds must reproduce every proof component");
    assert!(!a.1.is_empty() && !a.2.is_empty());
    let c = construct([8u8; 32]);
    assert_ne!(a.0, c.0, "a different seed changes the hiding commitments");
    assert_eq!(
      a.3, c.3,
      "the remainder structure does not depend on the coins"
    );
    // `prove_with_rng` is the unaudited wrapper of the same path.
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let (witness, instance) = IntModR1CSWitnessModp::<ME>::new_with_rng(
      &shape,
      &pk.ck,
      w.clone(),
      q.clone(),
      x.clone(),
      &mut rng,
    )
    .unwrap();
    let proof =
      IntModSpartanModpSNARK::<ME>::prove_with_rng(&pk, &instance, &witness, &mut rng).unwrap();
    assert_eq!(instance.commitment_bytes().unwrap(), a.0);
    assert_eq!(proof.eval_arg_bytes().unwrap(), a.1);
    assert_eq!(proof.sumcheck_remainder_bytes(), a.2);
  }

  /// The prover's and verifier's P0-D audit logs of an honest proof are
  /// complete (exactly the scheduled count) and identical, and every
  /// record has the scheduled purpose and width.
  fn assert_audited_roundtrip<M>(
    pk: &IntModSpartanModpProverKey<M>,
    vk: &IntModSpartanModpVerifierKey<M>,
    U: &IntModR1CSInstanceModp<M>,
    W: &IntModR1CSWitnessModp<M>,
    small_width: u16,
    openings: usize,
    s: usize,
  ) where
    M: ModEngine<TE = Keccak256Transcript<M>>,
  {
    use crate::prime_sampler::{PrimeSamplerPurpose, RUNTIME_P_WIDTH_BITS};
    let expected_p = IntModSpartanModpSNARK::<M>::prime_sampler_invocations_for_prove(pk).unwrap();
    let expected_v = IntModSpartanModpSNARK::<M>::prime_sampler_invocations_for_verify(vk).unwrap();
    assert_eq!(expected_p, expected_v);
    assert_eq!(expected_p, 1 + openings * s);
    let (proof, prover_records) =
      match IntModSpartanModpSNARK::<M>::prove_with_prime_audit(pk, U, W) {
        PrimeAuditedOutcome::Success { value, records } => (value, records),
        PrimeAuditedOutcome::Failure { source, .. } => panic!("prove failed: {source:?}"),
      };
    assert_eq!(prover_records.len(), expected_p);
    assert_eq!(prover_records[0].purpose(), PrimeSamplerPurpose::RuntimeP);
    assert_eq!(prover_records[0].width_bits(), RUNTIME_P_WIDTH_BITS);
    for rec in &prover_records[1..] {
      assert_eq!(rec.purpose(), PrimeSamplerPurpose::IntEvalSmallP);
      assert_eq!(rec.width_bits(), small_width);
      assert!(rec.is_success());
      assert_eq!(rec.mr_rounds_completed(), crate::prime_sampler::MR_ROUNDS);
    }
    let outcome = proof.verify_with_prime_audit(vk, U);
    assert!(outcome.is_success(), "{outcome:?}");
    assert_eq!(outcome.records(), prover_records.as_slice());
    // The production wrappers see the same value / result.
    proof.verify(vk, U).unwrap();
  }

  /// Audited round trip on the toy circuit: Hyrax engine, plain
  /// two-polynomial open (`W`, `Q`).
  #[test]
  fn imod_modp_audited_roundtrip_hyrax() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let s = pk.ck.params.s;
    let width = u16::try_from(pk.ck.params.log_p).unwrap();
    assert_audited_roundtrip::<ME>(&pk, &vk, &U, &W, width, 2, s);
  }

  /// Audited round trip through the Brakedown engine.
  #[test]
  fn imod_modp_audited_roundtrip_brakedown() {
    type BE = crate::provider::T256DynPrimeBdEngine;
    let one = BigUint::from(1u32);
    let zero = BigUint::from(0u32);
    let mat_a = vec![(0, 0, one.clone())];
    let mat_b = vec![(0, 1, one.clone())];
    let mat_c = vec![(0, 2, one)];
    let mods = vec![BigUint::from(14u64), zero.clone()];
    let shape = IntModR1CSShapeModp::<BE>::new(2, 4, 0, mat_a, mat_b, mat_c, mods).unwrap();
    let w = vec![
      BigUint::from(3u64),
      BigUint::from(5u64),
      BigUint::from(1u64),
      zero.clone(),
    ];
    let q = vec![BigUint::from(1u64), zero];
    let (pk, vk) = IntModSpartanModpSNARK::<BE>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<BE>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let s = pk.ck.params.s;
    let width = u16::try_from(pk.ck.params.log_p).unwrap();
    assert_audited_roundtrip::<BE>(&pk, &vk, &U, &W, width, 2, s);
  }

  /// Audited round trip on a width-segmented shape: the schedule is one
  /// opening per segment plus `Q`, each at the narrowed params' `s`.
  #[test]
  fn imod_modp_audited_roundtrip_width_segments() {
    use crate::{imod_r1cs_modp::WidthSegment, provider::pcs::integer_modpcs::IntEvalParams};
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let shape = shape
      .with_width_segments(vec![
        WidthSegment {
          start: 0,
          log_len: 1,
          log_t_f: 32,
        },
        WidthSegment {
          start: 2,
          log_len: 1,
          log_t_f: 64,
        },
      ])
      .unwrap();
    let params = IntEvalParams::derive(64, 32, 2, 2).unwrap();
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup_with_params(shape.clone(), params).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    assert_eq!(U.comm_w.len(), 2);
    let s = pk.ck.params.s;
    let width = u16::try_from(pk.ck.params.log_p).unwrap();
    assert_audited_roundtrip::<ME>(&pk, &vk, &U, &W, width, 3, s);
  }

  /// Driver-level undercount and overcount fail closed: a log one short
  /// of the schedule refuses the final invocation before that draw (and
  /// keeps every earlier record); a log one long completes the run but
  /// fails `finish_exact`.
  #[test]
  fn imod_modp_schedule_undercount_and_overcount_fail_closed() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let expected = IntModSpartanModpSNARK::<ME>::prime_sampler_invocations_for_prove(&pk).unwrap();
    assert!(expected >= 2);

    let mut short = PrimeAuditLog::with_expected(expected - 1).unwrap();
    let err =
      IntModSpartanModpSNARK::<ME>::prove_inner(&pk, &U, &W, &mut short, &mut rand::thread_rng())
        .unwrap_err();
    assert!(matches!(err, SpartanError::PrimeAuditLog { .. }), "{err:?}");
    assert_eq!(short.records().len(), expected - 1);
    assert!(short.records().iter().all(|r| r.is_success()));

    let mut long = PrimeAuditLog::with_expected(expected + 1).unwrap();
    IntModSpartanModpSNARK::<ME>::prove_inner(&pk, &U, &W, &mut long, &mut rand::thread_rng())
      .unwrap();
    assert_eq!(long.records().len(), expected);
    assert!(matches!(
      long.finish_exact(),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
  }

  /// Schedule validation failures — a `log_p` that does not fit the
  /// sampler's `u16` width, a total above the cap, and a verifier key
  /// whose count disagrees with the prover's — fail before the first
  /// transcript draw (no audit record) or, for the disagreeing verifier,
  /// reject the proof.
  #[test]
  fn imod_modp_schedule_validation_fails_before_the_first_draw() {
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let proof = IntModSpartanModpSNARK::<ME>::prove(&pk, &U, &W).unwrap();

    // Width conversion failure on either side.
    let mut bad_vk = vk.clone();
    bad_vk.vk_ee.params.log_p = 1 << 20;
    assert!(matches!(
      IntModSpartanModpSNARK::<ME>::prime_sampler_invocations_for_verify(&bad_vk),
      Err(SpartanError::InvalidInputLength { .. })
    ));
    match proof.verify_with_prime_audit(&bad_vk, &U) {
      PrimeAuditedOutcome::Failure { source, records } => {
        assert!(matches!(source, SpartanError::InvalidInputLength { .. }));
        assert!(records.is_empty(), "no draw before schedule validation");
      }
      PrimeAuditedOutcome::Success { .. } => panic!("must fail"),
    }
    let mut bad_pk = pk.clone();
    bad_pk.ck.params.log_p = 1 << 20;
    match IntModSpartanModpSNARK::<ME>::prove_with_prime_audit(&bad_pk, &U, &W) {
      PrimeAuditedOutcome::Failure { source, records } => {
        assert!(matches!(source, SpartanError::InvalidInputLength { .. }));
        assert!(records.is_empty());
      }
      PrimeAuditedOutcome::Success { .. } => panic!("must fail"),
    }

    // A schedule above the hard cap.
    let mut huge_vk = vk.clone();
    huge_vk.vk_ee.params.s = MAX_PRIME_INVOCATIONS;
    assert!(matches!(
      IntModSpartanModpSNARK::<ME>::prime_sampler_invocations_for_verify(&huge_vk),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
    match proof.verify_with_prime_audit(&huge_vk, &U) {
      PrimeAuditedOutcome::Failure { source, records } => {
        assert!(matches!(source, SpartanError::PrimeAuditLog { .. }));
        assert!(records.is_empty());
      }
      PrimeAuditedOutcome::Success { .. } => panic!("must fail"),
    }

    // Prover / verifier count disagreement: the accessors differ and the
    // verifier rejects the proof before sampling any IntEval prime.
    let mut skewed_vk = vk.clone();
    skewed_vk.vk_ee.params.s += 1;
    assert_ne!(
      IntModSpartanModpSNARK::<ME>::prime_sampler_invocations_for_prove(&pk).unwrap(),
      IntModSpartanModpSNARK::<ME>::prime_sampler_invocations_for_verify(&skewed_vk).unwrap()
    );
    let outcome = proof.verify_with_prime_audit(&skewed_vk, &U);
    assert!(!outcome.is_success());
    assert!(
      outcome.records().len() <= 1,
      "only the runtime draw may have happened"
    );
  }

  /// Sanity: the transcript-sampled `p` actually differs from the curve
  /// scalar prime `q`. Asserts the dual-field claim is real on this
  /// engine; the sampling derives `p` from the transcript bytes, so this
  /// also pins the byte-level Fiat-Shamir derivation in place.
  #[test]
  fn imod_modp_sampled_p_is_not_q() {
    use crate::provider::t256_scalar_params;
    let (shape, w, q) = build_toy(3, 5, 1, 14, 1);
    let (pk, _vk) = IntModSpartanModpSNARK::<ME>::setup(shape.clone()).unwrap();
    let (W, U) = IntModR1CSWitnessModp::<ME>::new(&shape, &pk.ck, w, q, vec![]).unwrap();
    let _ = W; // not needed for this check

    let mut t = Keccak256Transcript::<ME>::new_with_params(
      b"IntModSpartanModpSNARK",
      <ME as ModEngine>::bootstrap_params(),
    );
    t.absorb_bytes(b"vk", &pk.vk_digest);
    for cw in &U.comm_w {
      t.absorb(b"comm_w", cw);
    }
    t.absorb(b"comm_q", &U.comm_q);
    let mut log = PrimeAuditLog::with_expected(1).unwrap();
    let params_p = <ME as ModEngine>::sample_params(&mut t, &mut log).unwrap();
    let params_q = t256_scalar_params();
    // `p` is a transcript-sampled 128-bit prime in a 2-limb carrier; `q`
    // is the 256-bit curve scalar prime. Compare the modulus values as
    // byte strings, zero-extending `p` to `q`'s width.
    let p_enc = params_p.modulus().as_ref().to_le_bytes();
    let q_enc = params_q.modulus().as_ref().to_le_bytes();
    let p_bytes: &[u8] = p_enc.as_ref();
    let q_bytes: &[u8] = q_enc.as_ref();
    let mut p_wide = [0u8; 32];
    p_wide[..16].copy_from_slice(p_bytes);
    assert_ne!(&p_wide[..], q_bytes);
    // And the sampled p really is 128 bits (top bit forced by sampling).
    assert_eq!(p_bytes[15] & 0x80, 0x80);
  }
}
