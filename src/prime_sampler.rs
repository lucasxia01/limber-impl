//! Bounded, audited transcript prime sampler (P0-D).
//!
//! Both runtime-modulus sampling (`ModEngine::sample_params`) and the
//! IntEval small-prime sampling inside the integer Mod-PCS derive primes
//! from the Fiat-Shamir byte transcript. This module is the single
//! implementation both call sites share: [`sample_prime_v1`] draws a
//! uniform `width_bits`-bit odd candidate, prefilters it with
//! `crypto_primes::is_prime` (base-2 Miller–Rabin plus the BPSW'21 strong
//! Lucas test) and then runs [`MR_ROUNDS`] independent Miller–Rabin rounds
//! whose bases are themselves rejection-sampled from the transcript. Every
//! candidate and every base draw is one state-advancing squeeze, so prover
//! and verifier reproduce the same prime — or the same typed failure —
//! from the same transcript prefix. There is no external RNG.
//!
//! The sampler fails closed at every cap ([`T_MAX`] candidates,
//! [`BASE_DRAW_MAX`] masked draws per base) and on transcript exhaustion,
//! and it appends exactly one terminal [`PrimeSamplerAudit`] record to the
//! caller-owned [`PrimeAuditLog`] before every return. The log is bounded
//! by the driver's precomputed invocation schedule and by the hard cap
//! [`MAX_PRIME_INVOCATIONS`]; no caller can push into it directly.
//!
//! Security (per invocation, honest transcript): the candidate map is
//! uniform over the `width_bits`-bit odd integers with the top bit set,
//! composite acceptance is `< T_MAX · 4^{-MR_ROUNDS} = 2^{-132}`, and the
//! base-draw-cap probability is `< T_MAX · MR_ROUNDS · 2^{-160}`.

use crate::{errors::SpartanError, traits::transcript::ByteTranscript};
use crypto_bigint::{Odd, U256};
use crypto_primes::{Flavor, hazmat::MillerRabin, is_prime};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum number of candidates examined per invocation. Exhaustion fails
/// closed with [`PrimeSamplerErrorKind::CandidateCapExceeded`].
pub const T_MAX: u32 = 4096;
/// Independent Miller–Rabin rounds per accepted candidate. A uniform base
/// on `[2, n − 2]` is a strong liar for an odd composite `n` with
/// probability `< 1/4`, so a survivor is composite with probability
/// `< 4^{-72}`.
pub const MR_ROUNDS: u32 = 72;
/// Maximum masked draws per Miller–Rabin base. Exhaustion fails closed
/// with [`PrimeSamplerErrorKind::BaseDrawCapExceeded`] (probability
/// `< 2^{-160}` per base for an honest transcript).
pub const BASE_DRAW_MAX: u16 = 160;
/// Hard cap on sampler invocations recorded by one [`PrimeAuditLog`], and
/// therefore on the invocations one proof may contain.
pub const MAX_PRIME_INVOCATIONS: usize = 4096;
/// Smallest supported candidate width in bits (`n >= 5`, so `[2, n − 2]`
/// is non-empty).
pub const MIN_WIDTH_BITS: u16 = 3;
/// Largest supported candidate width in bits (the `U256` carrier).
pub const MAX_WIDTH_BITS: u16 = 256;
/// Width of the runtime sumcheck modulus `p` sampled by
/// `ModEngine::sample_params`.
pub const RUNTIME_P_WIDTH_BITS: u16 = 128;

/// Identifier of this sampler: forced top bit, BPSW'21 prefilter, 72
/// transcript-derived Miller–Rabin rounds, a 4096-candidate cap, a
/// 160-draw cap per base and a 4096-invocation cap per proof.
pub const LIMBER_PRIME_SAMPLER_ID: &str = "limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096";
/// Wire-format identifier of the protocol after the P0-D sampler change
/// (the squeeze labels and draw schedule changed, so the proof bytes of
/// the previous wire version are not interchangeable).
pub const LIMBER_PROTOCOL_WIRE_ID: &str = "limber/wire/v-p0d";
/// Transcript identifier: Keccak-256 byte transcript with the P0-D
/// sampler's candidate / base squeeze schedule.
pub const LIMBER_TRANSCRIPT_ID: &str = "limber/transcript/keccak256-p0d";

/// Prefix of the audit record's rolling digest.
const AUDIT_DIGEST_PREFIX: &[u8] = b"limber-prime-audit-v1\0";
/// Draw kinds in the rolling digest framing.
const DRAW_KIND_CANDIDATE: u8 = 0;
const DRAW_KIND_BASE: u8 = 1;

/// Which modulus an invocation samples. The purpose selects the squeeze
/// labels and is bound into the audit digest.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimeSamplerPurpose {
  /// The runtime sumcheck modulus `p` (width [`RUNTIME_P_WIDTH_BITS`]).
  RuntimeP = 0,
  /// An IntEval small prime `p_i` inside the integer Mod-PCS (width
  /// `IntEvalParams::log_p`).
  IntEvalSmallP = 1,
}

impl PrimeSamplerPurpose {
  /// The purpose tag bound into the rolling digest.
  pub fn tag(self) -> u8 {
    self as u8
  }

  /// Squeeze label of every candidate draw.
  pub fn candidate_label(self) -> &'static [u8] {
    match self {
      Self::RuntimeP => b"sample_p/candidate/v1",
      Self::IntEvalSmallP => b"sample_small_p/candidate/v1",
    }
  }

  /// Squeeze label of every Miller–Rabin base draw.
  pub fn mr_base_label(self) -> &'static [u8] {
    match self {
      Self::RuntimeP => b"sample_p/mr-base/v1",
      Self::IntEvalSmallP => b"sample_small_p/mr-base/v1",
    }
  }

  /// Stable lowercase name, for reports and serialized audits.
  pub fn name(self) -> &'static str {
    match self {
      Self::RuntimeP => "runtime_p",
      Self::IntEvalSmallP => "int_eval_small_p",
    }
  }
}

/// Exhaustive classification of a fail-closed sampler outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimeSamplerErrorKind {
  /// `width_bits` outside `[MIN_WIDTH_BITS, MAX_WIDTH_BITS]`. No draw
  /// happens; the record is a zero-draw record.
  InvalidWidth,
  /// No candidate survived within [`T_MAX`] trials; `trial` is the last
  /// trial examined.
  CandidateCapExceeded {
    /// The last candidate trial examined (`T_MAX`).
    trial: u32,
  },
  /// A Miller–Rabin base could not be drawn within [`BASE_DRAW_MAX`]
  /// masked draws.
  BaseDrawCapExceeded {
    /// Candidate trial (1-based) during which the cap was hit.
    trial: u32,
    /// Miller–Rabin round (1-based) during which the cap was hit.
    round: u16,
  },
  /// The transcript refused a squeeze for a reason other than round
  /// exhaustion.
  SqueezeFailed,
  /// The transcript's `u16` round counter is exhausted.
  TranscriptRoundsExhausted,
}

/// Terminal outcome of one sampler invocation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimeSamplerOutcome {
  /// The sampled prime as exactly `ceil(width_bits / 8)` canonical
  /// little-endian bytes.
  Success(Vec<u8>),
  /// The fail-closed reason.
  Failure(PrimeSamplerErrorKind),
}

/// Bounded audit record of one sampler invocation: purpose, width, the
/// terminal outcome, draw counters and a rolling SHA-256 over every
/// fixed-width draw. No per-draw values are retained, so the record size
/// is independent of the (attacker-influenced) number of draws.
///
/// Rolling digest framing: `SHA-256(b"limber-prime-audit-v1\0" ||
/// purpose_u8 || width_bits_le16 || frames...)`, one frame `kind_u8 ||
/// trial_le32 || round_le16 || draw_le16 || raw_le` per draw, where `kind`
/// is `0` for a candidate draw (`round = draw = 0`) and `1` for a base
/// draw, and `raw_le` is the complete 64-byte squeeze result before any
/// masking or bit forcing.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrimeSamplerAudit {
  purpose: PrimeSamplerPurpose,
  width_bits: u16,
  outcome: PrimeSamplerOutcome,
  candidates: u32,
  bases_accepted: u32,
  bases_rejected: u32,
  mr_rounds_completed: u32,
  rolling_digest: [u8; 32],
}

impl PrimeSamplerAudit {
  /// Which modulus the invocation sampled.
  pub fn purpose(&self) -> PrimeSamplerPurpose {
    self.purpose
  }

  /// Requested candidate width in bits.
  pub fn width_bits(&self) -> u16 {
    self.width_bits
  }

  /// The terminal outcome.
  pub fn outcome(&self) -> &PrimeSamplerOutcome {
    &self.outcome
  }

  /// Candidates drawn, including those rejected by the prefilter.
  pub fn candidates(&self) -> u32 {
    self.candidates
  }

  /// Base draws accepted by the masked rejection step (`x < m`).
  pub fn bases_accepted(&self) -> u32 {
    self.bases_accepted
  }

  /// Base draws rejected by the masked rejection step.
  pub fn bases_rejected(&self) -> u32 {
    self.bases_rejected
  }

  /// Miller–Rabin rounds that returned "probably prime".
  pub fn mr_rounds_completed(&self) -> u32 {
    self.mr_rounds_completed
  }

  /// Rolling digest of every draw (see the type docs for the framing).
  pub fn rolling_digest(&self) -> &[u8; 32] {
    &self.rolling_digest
  }

  /// Whether the invocation returned a prime.
  pub fn is_success(&self) -> bool {
    matches!(self.outcome, PrimeSamplerOutcome::Success(_))
  }
}

/// The caller-owned, bounded audit log of one prover or verifier run.
///
/// The driver computes the exact number of sampler invocations its
/// schedule performs, constructs the log with that count, threads it
/// through every sampler call, and calls [`finish_exact`](Self::finish_exact)
/// on success. The only append path is private and checks both
/// `records.len() < expected_count` and
/// `records.len() < MAX_PRIME_INVOCATIONS` before every push.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrimeAuditLog {
  expected_count: usize,
  records: Vec<PrimeSamplerAudit>,
}

impl PrimeAuditLog {
  /// A log expecting exactly `expected` sampler invocations. Rejects
  /// `expected > MAX_PRIME_INVOCATIONS` before allocating and reserves
  /// only the validated count.
  pub fn with_expected(expected: usize) -> Result<Self, SpartanError> {
    if expected > MAX_PRIME_INVOCATIONS {
      return Err(SpartanError::PrimeAuditLog {
        reason: format!(
          "expected {expected} prime-sampler invocations exceeds the cap {MAX_PRIME_INVOCATIONS}"
        ),
      });
    }
    let mut records = Vec::new();
    records.reserve_exact(expected);
    Ok(Self {
      expected_count: expected,
      records,
    })
  }

  /// The number of invocations this log was constructed for.
  pub fn expected_count(&self) -> usize {
    self.expected_count
  }

  /// The records appended so far, in invocation order.
  pub fn records(&self) -> &[PrimeSamplerAudit] {
    &self.records
  }

  /// Consume the log, requiring exactly `expected_count` records.
  pub fn finish_exact(self) -> Result<Vec<PrimeSamplerAudit>, SpartanError> {
    if self.records.len() != self.expected_count {
      return Err(SpartanError::PrimeAuditLog {
        reason: format!(
          "prime-sampler audit log holds {} records but the schedule expected {}",
          self.records.len(),
          self.expected_count
        ),
      });
    }
    Ok(self.records)
  }

  /// Consume the log without the exact-count check: the completed prefix
  /// of a failed run, for failure reporting only.
  pub fn into_records(self) -> Vec<PrimeSamplerAudit> {
    self.records
  }

  /// Whether one more record may be appended (both bounds).
  fn has_capacity(&self) -> bool {
    self.records.len() < self.expected_count && self.records.len() < MAX_PRIME_INVOCATIONS
  }

  /// Fail before any transcript draw when the schedule is exhausted: a
  /// further invocation means the driver undercounted its schedule.
  fn ensure_capacity(&self) -> Result<(), SpartanError> {
    if self.has_capacity() {
      Ok(())
    } else {
      Err(SpartanError::PrimeAuditLog {
        reason: format!(
          "prime-sampler invocation {} exceeds the scheduled count {} (cap {})",
          self.records.len().saturating_add(1),
          self.expected_count,
          MAX_PRIME_INVOCATIONS
        ),
      })
    }
  }

  /// The sole append path. Returns the index of the appended record.
  fn append(&mut self, record: PrimeSamplerAudit) -> Result<usize, SpartanError> {
    self.ensure_capacity()?;
    let index = self.records.len();
    self.records.push(record);
    Ok(index)
  }
}

/// Owned audit snapshot of a driver-level prove or verify: the value (or
/// the error) together with every record the run appended.
#[derive(Clone, Debug)]
pub enum PrimeAuditedOutcome<T> {
  /// The run succeeded and its audit log held exactly the scheduled
  /// number of records.
  Success {
    /// The proof (prove) or `()` (verify).
    value: T,
    /// The complete audit log, in invocation order.
    records: Vec<PrimeSamplerAudit>,
  },
  /// The run failed; `records` is the completed prefix, including the
  /// terminal record of a failed sampler invocation when the failure was
  /// the sampler's.
  Failure {
    /// The error the run returned.
    source: SpartanError,
    /// Every record appended before the failure.
    records: Vec<PrimeSamplerAudit>,
  },
}

impl<T> PrimeAuditedOutcome<T> {
  /// The records of either variant.
  pub fn records(&self) -> &[PrimeSamplerAudit] {
    match self {
      Self::Success { records, .. } | Self::Failure { records, .. } => records,
    }
  }

  /// Whether the run succeeded.
  pub fn is_success(&self) -> bool {
    matches!(self, Self::Success { .. })
  }

  /// Drop the audit copy and keep the production value / error.
  pub fn into_result(self) -> Result<T, SpartanError> {
    match self {
      Self::Success { value, .. } => Ok(value),
      Self::Failure { source, .. } => Err(source),
    }
  }
}

/// Rolling-digest accumulator (framing documented on [`PrimeSamplerAudit`]).
struct AuditDigest(Sha256);

impl AuditDigest {
  fn new(purpose: PrimeSamplerPurpose, width_bits: u16) -> Self {
    let mut h = Sha256::new();
    h.update(AUDIT_DIGEST_PREFIX);
    h.update([purpose.tag()]);
    h.update(width_bits.to_le_bytes());
    Self(h)
  }

  fn frame(&mut self, kind: u8, trial: u32, round: u16, draw: u16, raw: &[u8; 64]) {
    self.0.update([kind]);
    self.0.update(trial.to_le_bytes());
    self.0.update(round.to_le_bytes());
    self.0.update(draw.to_le_bytes());
    self.0.update(raw);
  }

  fn finalize(self) -> [u8; 32] {
    self.0.finalize().into()
  }
}

/// Draw counters of one invocation.
#[derive(Default)]
struct Counters {
  candidates: u32,
  bases_accepted: u32,
  bases_rejected: u32,
  mr_rounds_completed: u32,
}

/// Number of bytes a `width_bits`-bit candidate occupies.
fn width_bytes(width_bits: u16) -> usize {
  usize::from(width_bits.div_ceil(8))
}

/// The candidate map: take the low `ceil(width_bits / 8)` little-endian
/// bytes of the squeeze, clear the bits above `width_bits`, force the top
/// bit (`width_bits − 1`) and the low bit. Requires a validated width.
fn force_candidate(raw: &[u8; 64], width_bits: u16) -> U256 {
  let nbytes = width_bytes(width_bits);
  let mut buf = [0u8; 32];
  buf[..nbytes].copy_from_slice(&raw[..nbytes]);
  let top = usize::from(width_bits.saturating_sub(1));
  let top_byte = top / 8;
  let top_bit = top % 8;
  // Keep only bits 0..=top_bit of the top byte; zero everything above.
  let keep: u8 = ((1u16 << (top_bit + 1)) - 1) as u8;
  buf[top_byte] &= keep;
  for b in &mut buf[top_byte + 1..] {
    *b = 0;
  }
  buf[top_byte] |= 1u8 << top_bit;
  buf[0] |= 1;
  U256::from_le_slice(&buf)
}

/// Exactly `ceil(width_bits / 8)` canonical little-endian bytes of `p`.
fn canonical_le_bytes(p: &U256, width_bits: u16) -> Vec<u8> {
  p.to_le_bytes()[..width_bytes(width_bits)].to_vec()
}

/// Keep the `ell` low bits of `x`; `ell >= 256` is the identity.
fn keep_low_bits(x: U256, ell: u32) -> U256 {
  if ell >= U256::BITS {
    return x;
  }
  let mask = U256::ONE.shl_vartime(ell).wrapping_sub(&U256::ONE);
  x.bitand(&mask)
}

/// The low 32 bytes of a base squeeze as a `U256` (before masking).
fn base_draw_value(raw: &[u8; 64]) -> U256 {
  U256::from_le_slice(&raw[..32])
}

/// One transcript squeeze with the error mapped to a sampler kind.
fn squeeze<T: ByteTranscript>(
  transcript: &mut T,
  label: &'static [u8],
) -> Result<[u8; 64], PrimeSamplerErrorKind> {
  transcript.squeeze_bytes(label).map_err(|e| match e {
    SpartanError::InternalTranscriptError => PrimeSamplerErrorKind::TranscriptRoundsExhausted,
    _ => PrimeSamplerErrorKind::SqueezeFailed,
  })
}

/// The sampler proper. Every return is wrapped by [`sample_prime_v1`],
/// which appends the terminal record.
fn sample_inner<T: ByteTranscript>(
  transcript: &mut T,
  purpose: PrimeSamplerPurpose,
  width_bits: u16,
  counters: &mut Counters,
  digest: &mut AuditDigest,
) -> Result<U256, PrimeSamplerErrorKind> {
  if !(MIN_WIDTH_BITS..=MAX_WIDTH_BITS).contains(&width_bits) {
    return Err(PrimeSamplerErrorKind::InvalidWidth);
  }
  let one = U256::ONE;
  let two = U256::from(2u8);
  let three = U256::from(3u8);
  let candidate_label = purpose.candidate_label();
  let base_label = purpose.mr_base_label();

  'candidates: for trial in 1..=T_MAX {
    let raw = squeeze(transcript, candidate_label)?;
    counters.candidates += 1;
    digest.frame(DRAW_KIND_CANDIDATE, trial, 0, 0, &raw);
    let n = force_candidate(&raw, width_bits);
    // Prefilter: conventions, base-2 Miller–Rabin, BPSW'21 strong Lucas.
    if !is_prime(Flavor::Any, &n) {
      continue;
    }
    let Some(odd) = Odd::new(n).into_option() else {
      continue;
    };
    let mr = MillerRabin::new(odd);
    // Bases are uniform on `[2, n − 2]`: draw `x` uniform on `[0, m)`,
    // `m = n − 3`, by masked rejection, then `base = x + 2`. `n >= 5`
    // (width >= 3 with the top and low bits forced), so `m >= 2`.
    let m = n.wrapping_sub(&three);
    let ell = m.wrapping_sub(&one).bits_vartime();
    for round in 1..=(MR_ROUNDS as u16) {
      let mut base = None;
      for draw in 1..=BASE_DRAW_MAX {
        let raw = squeeze(transcript, base_label)?;
        digest.frame(DRAW_KIND_BASE, trial, round, draw, &raw);
        let x = keep_low_bits(base_draw_value(&raw), ell);
        if x < m {
          counters.bases_accepted += 1;
          base = Some(x.wrapping_add(&two));
          break;
        }
        counters.bases_rejected += 1;
      }
      let Some(base) = base else {
        return Err(PrimeSamplerErrorKind::BaseDrawCapExceeded { trial, round });
      };
      if mr.test(&base).is_composite() {
        continue 'candidates;
      }
      counters.mr_rounds_completed += 1;
    }
    return Ok(n);
  }
  Err(PrimeSamplerErrorKind::CandidateCapExceeded { trial: T_MAX })
}

/// Sample a `width_bits`-bit prime from the transcript under `purpose`
/// (see the module docs for the algorithm and its caps).
///
/// Exactly one terminal [`PrimeSamplerAudit`] is appended to `log` before
/// every return, including a width-validation failure (a zero-draw record
/// with zero counters and the initialized rolling digest). Errors carry
/// the kind and the index of that record as
/// [`SpartanError::PrimeSampler`]. The only exception is a log whose
/// schedule is already exhausted: that is a driver undercount, reported as
/// [`SpartanError::PrimeAuditLog`] before any transcript draw and without
/// a record.
pub fn sample_prime_v1<T: ByteTranscript>(
  transcript: &mut T,
  purpose: PrimeSamplerPurpose,
  width_bits: u16,
  log: &mut PrimeAuditLog,
) -> Result<U256, SpartanError> {
  log.ensure_capacity()?;
  let mut digest = AuditDigest::new(purpose, width_bits);
  let mut counters = Counters::default();
  let result = sample_inner(transcript, purpose, width_bits, &mut counters, &mut digest);
  let outcome = match &result {
    Ok(p) => PrimeSamplerOutcome::Success(canonical_le_bytes(p, width_bits)),
    Err(kind) => PrimeSamplerOutcome::Failure(*kind),
  };
  let record = PrimeSamplerAudit {
    purpose,
    width_bits,
    outcome,
    candidates: counters.candidates,
    bases_accepted: counters.bases_accepted,
    bases_rejected: counters.bases_rejected,
    mr_rounds_completed: counters.mr_rounds_completed,
    rolling_digest: digest.finalize(),
  };
  let record_index = log.append(record)?;
  result.map_err(|kind| SpartanError::PrimeSampler { kind, record_index })
}

/// Checked-sum helper for invocation schedules: `a + b` or a typed
/// schedule error on overflow.
pub fn checked_schedule_add(a: usize, b: usize) -> Result<usize, SpartanError> {
  a.checked_add(b).ok_or_else(|| SpartanError::PrimeAuditLog {
    reason: "prime-sampler invocation schedule overflows usize".to_string(),
  })
}

/// Checked conversion of an IntEval `log_p` (a `usize`) to the sampler's
/// `u16` width, performed during driver schedule validation before any
/// transcript draw. A failure is a configuration error, not a sampler
/// invocation.
pub fn schedule_width_bits(log_p: usize) -> Result<u16, SpartanError> {
  u16::try_from(log_p).map_err(|_| SpartanError::InvalidInputLength {
    reason: format!("IntEval log_p = {log_p} does not fit the sampler's u16 width"),
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{T256DynPrimeEngine, keccak::Keccak256Transcript},
    traits::{mod_engine::ModEngine, transcript::TranscriptEngineTrait},
  };

  type Tr = Keccak256Transcript<T256DynPrimeEngine>;

  fn keccak(label: &'static [u8]) -> Tr {
    let mut t = Tr::new_with_params(
      b"prime-sampler-test",
      T256DynPrimeEngine::bootstrap_params(),
    );
    t.absorb_bytes(b"seed", label);
    t
  }

  fn log(n: usize) -> PrimeAuditLog {
    PrimeAuditLog::with_expected(n).unwrap()
  }

  /// A transcript replaying scripted 64-byte frames, then a fallback
  /// frame forever; optionally failing every squeeze with a fixed error.
  struct Scripted {
    frames: Vec<[u8; 64]>,
    fallback: [u8; 64],
    cursor: usize,
    fail: Option<SpartanError>,
  }

  impl Scripted {
    fn new(frames: Vec<[u8; 64]>, fallback: [u8; 64]) -> Self {
      Self {
        frames,
        fallback,
        cursor: 0,
        fail: None,
      }
    }

    fn failing(err: SpartanError) -> Self {
      Self {
        frames: Vec::new(),
        fallback: [0; 64],
        cursor: 0,
        fail: Some(err),
      }
    }
  }

  impl ByteTranscript for Scripted {
    fn squeeze_bytes(&mut self, _label: &'static [u8]) -> Result<[u8; 64], SpartanError> {
      if let Some(e) = &self.fail {
        return Err(e.clone());
      }
      let out = self
        .frames
        .get(self.cursor)
        .copied()
        .unwrap_or(self.fallback);
      self.cursor += 1;
      Ok(out)
    }
    fn absorb_bytes(&mut self, _label: &'static [u8], _bytes: &[u8]) {}
    fn dom_sep(&mut self, _bytes: &'static [u8]) {}
  }

  fn frame(low: &[u8]) -> [u8; 64] {
    let mut f = [0u8; 64];
    f[..low.len()].copy_from_slice(low);
    f
  }

  fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
      return false;
    }
    let mut d = 2;
    while d * d <= n {
      if n.is_multiple_of(d) {
        return false;
      }
      d += 1;
    }
    true
  }

  fn u256_to_u64(x: &U256) -> u64 {
    let bytes = x.to_le_bytes();
    assert!(bytes[8..].iter().all(|b| *b == 0));
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
  }

  fn initialized_digest(purpose: PrimeSamplerPurpose, width_bits: u16) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"limber-prime-audit-v1\0");
    h.update([purpose.tag()]);
    h.update(width_bits.to_le_bytes());
    h.finalize().into()
  }

  #[test]
  fn identifiers_and_caps_match_the_contract() {
    assert_eq!(
      LIMBER_PRIME_SAMPLER_ID,
      "limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096"
    );
    assert_eq!(LIMBER_PROTOCOL_WIRE_ID, "limber/wire/v-p0d");
    assert_eq!(LIMBER_TRANSCRIPT_ID, "limber/transcript/keccak256-p0d");
    assert_eq!(T_MAX, 4096);
    assert_eq!(MR_ROUNDS, 72);
    assert_eq!(BASE_DRAW_MAX, 160);
    assert_eq!(MAX_PRIME_INVOCATIONS, 4096);
    assert_eq!(PrimeSamplerPurpose::RuntimeP.tag(), 0);
    assert_eq!(PrimeSamplerPurpose::IntEvalSmallP.tag(), 1);
    assert_eq!(
      PrimeSamplerPurpose::RuntimeP.candidate_label(),
      b"sample_p/candidate/v1"
    );
    assert_eq!(
      PrimeSamplerPurpose::RuntimeP.mr_base_label(),
      b"sample_p/mr-base/v1"
    );
    assert_eq!(
      PrimeSamplerPurpose::IntEvalSmallP.candidate_label(),
      b"sample_small_p/candidate/v1"
    );
    assert_eq!(
      PrimeSamplerPurpose::IntEvalSmallP.mr_base_label(),
      b"sample_small_p/mr-base/v1"
    );
  }

  /// The runtime width on the production transcript: deterministic,
  /// exactly 128 bits, odd, passes the prefilter, full audit record.
  #[test]
  fn runtime_width_is_deterministic_on_keccak() {
    let mut a = keccak(b"runtime");
    let mut b = keccak(b"runtime");
    let mut la = log(2);
    let mut lb = log(2);
    let pa = sample_prime_v1(&mut a, PrimeSamplerPurpose::RuntimeP, 128, &mut la).unwrap();
    let pb = sample_prime_v1(&mut b, PrimeSamplerPurpose::RuntimeP, 128, &mut lb).unwrap();
    assert_eq!(pa, pb);
    assert_eq!(la, lb);
    assert_eq!(pa.bits_vartime(), 128);
    assert!(bool::from(pa.is_odd()));
    assert!(is_prime(Flavor::Any, &pa));
    let rec = &la.records()[0];
    assert_eq!(rec.purpose(), PrimeSamplerPurpose::RuntimeP);
    assert_eq!(rec.width_bits(), 128);
    assert_eq!(rec.mr_rounds_completed(), MR_ROUNDS);
    assert!(rec.bases_accepted() >= MR_ROUNDS);
    assert!(rec.candidates() >= 1);
    assert!(rec.is_success());
    match rec.outcome() {
      PrimeSamplerOutcome::Success(bytes) => {
        assert_eq!(bytes.len(), 16);
        assert_eq!(bytes[15] & 0x80, 0x80);
        assert_eq!(bytes[0] & 1, 1);
        assert_eq!(bytes.as_slice(), &pa.to_le_bytes()[..16]);
      }
      PrimeSamplerOutcome::Failure(kind) => panic!("unexpected failure {kind:?}"),
    }
    // A second invocation on the same transcript yields a different prime
    // and a second record.
    let q = sample_prime_v1(&mut a, PrimeSamplerPurpose::RuntimeP, 128, &mut la).unwrap();
    assert_ne!(q, pa);
    assert_eq!(la.records().len(), 2);
    assert_eq!(la.finish_exact().unwrap().len(), 2);
  }

  /// `ModEngine::sample_params` narrows to a 128-bit `U128` modulus and
  /// appends exactly one `RuntimeP` record.
  #[test]
  fn sample_params_narrows_to_u128_and_records_once() {
    let mut t = keccak(b"params");
    let mut l = log(1);
    let params = <T256DynPrimeEngine as ModEngine>::sample_params(&mut t, &mut l).unwrap();
    assert_eq!(params.modulus().as_ref().bits_vartime(), 128);
    assert_eq!(l.records().len(), 1);
    assert_eq!(l.records()[0].purpose(), PrimeSamplerPurpose::RuntimeP);
    assert_eq!(l.records()[0].width_bits(), RUNTIME_P_WIDTH_BITS);
    let mut t2 = keccak(b"params");
    let mut l2 = log(1);
    let p2 = sample_prime_v1(&mut t2, PrimeSamplerPurpose::RuntimeP, 128, &mut l2).unwrap();
    assert_eq!(
      &p2.to_le_bytes()[..16],
      params.modulus().as_ref().to_le_bytes().as_ref()
    );
  }

  /// Every raw value maps onto the odd, top-bit-set values of the width
  /// uniformly: 4-to-1 at a byte-aligned width, and `2^(8·bytes − width +
  /// 2)`-to-1 at an unaligned one (the cleared high bits and the two
  /// forced bits are the only lost entropy).
  #[test]
  fn candidate_map_is_uniform_over_top_bit_set_odd_values() {
    let mut counts = [0u32; 256];
    for raw in 0..=255u8 {
      let n = u256_to_u64(&force_candidate(&frame(&[raw]), 8));
      counts[n as usize] += 1;
    }
    for (v, c) in counts.iter().enumerate() {
      let expected = if v >= 128 && v % 2 == 1 { 4 } else { 0 };
      assert_eq!(*c, expected, "value {v}");
    }
    // Width 11: raw 16-bit words, 2^(16 − 11 + 2) = 128 preimages each.
    let mut counts = vec![0u32; 1 << 11];
    for raw in 0..=u16::MAX {
      let n = u256_to_u64(&force_candidate(&frame(&raw.to_le_bytes()), 11));
      assert!(n < 1 << 11);
      counts[n as usize] += 1;
    }
    for (v, c) in counts.iter().enumerate() {
      let expected = if v >= 1 << 10 && v % 2 == 1 { 128 } else { 0 };
      assert_eq!(*c, expected, "value {v}");
    }
    // Full width: the top byte's high bit is forced, bytes above are the
    // squeeze's own low 32 bytes.
    let n = force_candidate(&frame(&[0u8; 32]), 256);
    assert!(n.bit_vartime(255));
    assert!(n.bit_vartime(0));
    assert_eq!(n.bits_vartime(), 256);
    // Success bytes are exactly ceil(width/8) long.
    assert_eq!(canonical_le_bytes(&n, 256).len(), 32);
    assert_eq!(canonical_le_bytes(&n, 11).len(), 2);
    assert_eq!(canonical_le_bytes(&n, 3).len(), 1);
  }

  /// Width validation fails before any draw with a zero-draw record.
  #[test]
  fn invalid_width_leaves_a_zero_draw_record_without_drawing() {
    for (i, width) in [0u16, 1, 2, 257, u16::MAX].into_iter().enumerate() {
      let mut t = Scripted::new(Vec::new(), [0; 64]);
      let mut l = log(MAX_PRIME_INVOCATIONS);
      // Earlier records stay in place: fill `i` failures first.
      for _ in 0..i {
        let _ = sample_prime_v1(&mut t, PrimeSamplerPurpose::IntEvalSmallP, 0, &mut l);
      }
      let err =
        sample_prime_v1(&mut t, PrimeSamplerPurpose::IntEvalSmallP, width, &mut l).unwrap_err();
      assert_eq!(
        err,
        SpartanError::PrimeSampler {
          kind: PrimeSamplerErrorKind::InvalidWidth,
          record_index: i,
        }
      );
      assert_eq!(t.cursor, 0, "no transcript draw");
      let rec = &l.records()[i];
      assert_eq!(rec.width_bits(), width);
      assert_eq!(
        rec.outcome(),
        &PrimeSamplerOutcome::Failure(PrimeSamplerErrorKind::InvalidWidth)
      );
      assert_eq!(rec.candidates(), 0);
      assert_eq!(rec.bases_accepted(), 0);
      assert_eq!(rec.bases_rejected(), 0);
      assert_eq!(rec.mr_rounds_completed(), 0);
      assert_eq!(
        rec.rolling_digest(),
        &initialized_digest(PrimeSamplerPurpose::IntEvalSmallP, width)
      );
    }
  }

  /// 2047 = 23 · 89 is a base-2 strong pseudoprime: it passes base-2
  /// Miller–Rabin but the BPSW'21 prefilter rejects it, so no base is
  /// drawn for it and the sampler moves to the next candidate.
  #[test]
  fn strong_pseudoprime_is_rejected_by_the_prefilter() {
    let n2047 = U256::from(2047u32);
    let mr = MillerRabin::new(Odd::new(n2047).unwrap());
    assert!(mr.test_base_two().is_probably_prime());
    assert!(!is_prime(Flavor::Any, &n2047));
    // And a concrete base does catch it in one round.
    assert!(mr.test(&U256::from(3u32)).is_composite());

    // Width 11: 2047 = 0x07FF has bits 10 and 0 set already, as does the
    // prime 2039 = 0x07F7. Base draws of zero give `x = 0`, base 2.
    let frames = vec![
      frame(&0x07FFu16.to_le_bytes()),
      frame(&0x07F7u16.to_le_bytes()),
    ];
    let mut t = Scripted::new(frames, [0; 64]);
    let mut l = log(1);
    let p = sample_prime_v1(&mut t, PrimeSamplerPurpose::IntEvalSmallP, 11, &mut l).unwrap();
    assert_eq!(u256_to_u64(&p), 2039);
    let rec = &l.records()[0];
    assert_eq!(rec.candidates(), 2);
    assert_eq!(rec.bases_accepted(), MR_ROUNDS);
    assert_eq!(rec.bases_rejected(), 0);
    assert_eq!(rec.mr_rounds_completed(), MR_ROUNDS);
    assert_eq!(t.cursor, 2 + MR_ROUNDS as usize);
    assert_eq!(
      rec.outcome(),
      &PrimeSamplerOutcome::Success(vec![0xF7, 0x07])
    );
  }

  /// A composite that survives the prefilter would still need to survive
  /// 72 transcript-derived bases: the per-round test is the real
  /// Miller–Rabin check, exercised here on a Carmichael number.
  #[test]
  fn mr_round_rejects_carmichael_with_a_witness_base() {
    let n = U256::from(561u32); // 3 · 11 · 17
    let mr = MillerRabin::new(Odd::new(n).unwrap());
    // 561 is not a base-2 strong pseudoprime; bases 2, 5 and 7 witness,
    // while 50 is one of its strong liars (1, 50, 101, 460, 511, 560).
    assert!(mr.test(&U256::from(2u32)).is_composite());
    assert!(mr.test(&U256::from(5u32)).is_composite());
    assert!(mr.test(&U256::from(7u32)).is_composite());
    assert!(!mr.test(&U256::from(50u32)).is_composite());
    // A genuine prime passes every base in range.
    let p = U256::from(2039u32);
    let mr = MillerRabin::new(Odd::new(p).unwrap());
    for b in 2u32..2037 {
      assert!(!mr.test(&U256::from(b)).is_composite(), "base {b}");
    }
  }

  /// Every squeeze yields the composite 129 = 3 · 43: the candidate cap
  /// fails closed with the exact counters and no base draws.
  #[test]
  fn candidate_cap_fails_closed() {
    let mut t = Scripted::new(Vec::new(), [0; 64]);
    let mut l = log(1);
    let err = sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 8, &mut l).unwrap_err();
    assert_eq!(
      err,
      SpartanError::PrimeSampler {
        kind: PrimeSamplerErrorKind::CandidateCapExceeded { trial: T_MAX },
        record_index: 0,
      }
    );
    assert_eq!(t.cursor, T_MAX as usize);
    let rec = &l.records()[0];
    assert_eq!(rec.candidates(), T_MAX);
    assert_eq!(rec.bases_accepted(), 0);
    assert_eq!(rec.bases_rejected(), 0);
    assert_eq!(rec.mr_rounds_completed(), 0);
    assert_eq!(
      rec.outcome(),
      &PrimeSamplerOutcome::Failure(PrimeSamplerErrorKind::CandidateCapExceeded { trial: T_MAX })
    );
    assert_eq!(rec.purpose(), PrimeSamplerPurpose::RuntimeP);
    // The log still finishes exactly: the failure holds its record.
    assert_eq!(l.finish_exact().unwrap().len(), 1);
  }

  /// Candidate 137 (prime, `m = 134`, `ell = bitlen(133) = 8`) with every
  /// base draw `0xFF = 255 >= m`: the base-draw cap fails closed in round
  /// 1 of trial 1 after exactly `BASE_DRAW_MAX` rejections.
  #[test]
  fn base_draw_cap_fails_closed() {
    let mut t = Scripted::new(vec![frame(&[0x89])], [0xFF; 64]);
    let mut l = log(1);
    let err = sample_prime_v1(&mut t, PrimeSamplerPurpose::IntEvalSmallP, 8, &mut l).unwrap_err();
    assert_eq!(
      err,
      SpartanError::PrimeSampler {
        kind: PrimeSamplerErrorKind::BaseDrawCapExceeded { trial: 1, round: 1 },
        record_index: 0,
      }
    );
    assert_eq!(t.cursor, 1 + usize::from(BASE_DRAW_MAX));
    let rec = &l.records()[0];
    assert_eq!(rec.candidates(), 1);
    assert_eq!(rec.bases_accepted(), 0);
    assert_eq!(rec.bases_rejected(), u32::from(BASE_DRAW_MAX));
    assert_eq!(rec.mr_rounds_completed(), 0);
  }

  /// Squeeze failures are typed: round-counter exhaustion is
  /// `TranscriptRoundsExhausted`, anything else `SqueezeFailed`; both
  /// leave a record with no framed draw.
  #[test]
  fn squeeze_failures_are_typed_and_recorded() {
    let cases = [
      (
        SpartanError::InternalTranscriptError,
        PrimeSamplerErrorKind::TranscriptRoundsExhausted,
      ),
      (
        SpartanError::InternalError {
          reason: "mock".to_string(),
        },
        PrimeSamplerErrorKind::SqueezeFailed,
      ),
    ];
    for (source, kind) in cases {
      let mut t = Scripted::failing(source);
      let mut l = log(1);
      let err = sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap_err();
      assert_eq!(
        err,
        SpartanError::PrimeSampler {
          kind,
          record_index: 0
        }
      );
      let rec = &l.records()[0];
      assert_eq!(rec.outcome(), &PrimeSamplerOutcome::Failure(kind));
      assert_eq!(rec.candidates(), 0);
      assert_eq!(
        rec.rolling_digest(),
        &initialized_digest(PrimeSamplerPurpose::RuntimeP, 128)
      );
    }
  }

  /// The production Keccak transcript's `u16` round counter: after 65535
  /// squeezes the next one fails, and the sampler reports it as the typed
  /// exhaustion kind rather than looping or panicking.
  #[test]
  fn keccak_round_exhaustion_is_fail_closed() {
    let mut t = keccak(b"exhaust");
    for _ in 0..u16::MAX {
      t.squeeze_bytes(b"burn").unwrap();
    }
    let mut l = log(1);
    let err = sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap_err();
    assert_eq!(
      err,
      SpartanError::PrimeSampler {
        kind: PrimeSamplerErrorKind::TranscriptRoundsExhausted,
        record_index: 0,
      }
    );
    assert_eq!(l.records()[0].candidates(), 0);
  }

  /// The rolling digest and every counter are reproduced by an independent
  /// replay of the squeeze sequence on a cloned transcript, using trial
  /// division as the primality oracle (exact at 16 bits, so the replay
  /// never consults the sampler's Miller–Rabin code).
  #[test]
  fn rolling_digest_matches_independent_replay() {
    let purpose = PrimeSamplerPurpose::IntEvalSmallP;
    let width = 16u16;
    let mut t = keccak(b"replay");
    let mut replay = t.clone();
    let mut l = log(1);
    let p = sample_prime_v1(&mut t, purpose, width, &mut l).unwrap();
    let p = u256_to_u64(&p);
    let rec = &l.records()[0];

    let mut h = Sha256::new();
    h.update(b"limber-prime-audit-v1\0");
    h.update([1u8]);
    h.update(16u16.to_le_bytes());
    let mut trial = 0u32;
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut rounds = 0u32;
    loop {
      trial += 1;
      let raw = replay
        .squeeze_bytes(b"sample_small_p/candidate/v1")
        .unwrap();
      h.update([0u8]);
      h.update(trial.to_le_bytes());
      h.update(0u16.to_le_bytes());
      h.update(0u16.to_le_bytes());
      h.update(raw);
      let n = u64::from(u16::from_le_bytes([raw[0], raw[1]]) | 0x8001);
      if !is_prime_u64(n) {
        continue;
      }
      let m = n - 3;
      let ell = 64 - (m - 1).leading_zeros();
      for round in 1..=(MR_ROUNDS as u16) {
        for draw in 1..=BASE_DRAW_MAX {
          let raw = replay.squeeze_bytes(b"sample_small_p/mr-base/v1").unwrap();
          h.update([1u8]);
          h.update(trial.to_le_bytes());
          h.update(round.to_le_bytes());
          h.update(draw.to_le_bytes());
          h.update(raw);
          // Low 32 bytes masked to `ell` bits; `ell <= 16` here.
          let x = u64::from(u16::from_le_bytes([raw[0], raw[1]])) & ((1u64 << ell) - 1);
          if x < m {
            accepted += 1;
            assert!((2..=n - 2).contains(&(x + 2)));
            break;
          }
          rejected += 1;
        }
        // A prime passes every Miller–Rabin base.
        rounds += 1;
      }
      assert_eq!(n, p);
      break;
    }
    let digest: [u8; 32] = h.finalize().into();
    assert_eq!(rec.rolling_digest(), &digest);
    assert_eq!(rec.candidates(), trial);
    assert_eq!(rec.bases_accepted(), accepted);
    assert_eq!(rec.bases_rejected(), rejected);
    assert_eq!(rec.mr_rounds_completed(), rounds);
    assert_eq!(accepted, MR_ROUNDS);
    assert_eq!(
      rec.outcome(),
      &PrimeSamplerOutcome::Success((p as u16).to_le_bytes().to_vec())
    );
    // The replay consumed exactly the sampler's squeezes: both transcripts
    // now produce the same next challenge.
    assert_eq!(
      t.squeeze_bytes(b"after").unwrap(),
      replay.squeeze_bytes(b"after").unwrap()
    );
  }

  /// The mask keeps `ell` bits exactly and `ell >= 256` is the identity.
  #[test]
  fn keep_low_bits_edge_cases() {
    let x = U256::MAX;
    assert_eq!(keep_low_bits(x, 256), x);
    assert_eq!(keep_low_bits(x, 300), x);
    assert_eq!(keep_low_bits(x, 255).bits_vartime(), 255);
    assert_eq!(keep_low_bits(x, 1), U256::ONE);
    assert_eq!(keep_low_bits(x, 0), U256::ZERO);
  }

  /// Log bounds: the cap, the undercount (refused before any draw), the
  /// overcount (`finish_exact` mismatch) and record retention.
  #[test]
  fn audit_log_bounds_fail_closed() {
    assert!(matches!(
      PrimeAuditLog::with_expected(MAX_PRIME_INVOCATIONS + 1),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
    assert_eq!(
      PrimeAuditLog::with_expected(MAX_PRIME_INVOCATIONS)
        .unwrap()
        .expected_count(),
      MAX_PRIME_INVOCATIONS
    );

    // Undercount: a second invocation against a one-slot log is refused
    // before the first draw; the transcript is untouched.
    let mut t = keccak(b"undercount");
    let mut l = log(1);
    sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap();
    let mut untouched = t.clone();
    let err = sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap_err();
    assert!(matches!(err, SpartanError::PrimeAuditLog { .. }), "{err:?}");
    assert_eq!(l.records().len(), 1);
    assert_eq!(
      t.squeeze_bytes(b"probe").unwrap(),
      untouched.squeeze_bytes(b"probe").unwrap()
    );
    assert_eq!(l.finish_exact().unwrap().len(), 1);

    // Overcount: fewer invocations than scheduled fail `finish_exact`.
    let mut t = keccak(b"overcount");
    let mut l = log(2);
    sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap();
    let snapshot = l.clone();
    assert!(matches!(
      l.finish_exact(),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
    assert_eq!(snapshot.into_records().len(), 1);

    // A zero-schedule log refuses the very first invocation.
    let mut t = Scripted::new(Vec::new(), [0; 64]);
    let mut l = log(0);
    assert!(matches!(
      sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
    assert_eq!(t.cursor, 0);
    assert!(l.finish_exact().unwrap().is_empty());
  }

  /// A later failure retains every earlier record; the error names the
  /// failing record's index.
  #[test]
  fn later_failure_retains_earlier_records() {
    let mut t = keccak(b"retain");
    let mut l = log(3);
    sample_prime_v1(&mut t, PrimeSamplerPurpose::RuntimeP, 128, &mut l).unwrap();
    sample_prime_v1(&mut t, PrimeSamplerPurpose::IntEvalSmallP, 40, &mut l).unwrap();
    let mut bad = Scripted::new(Vec::new(), [0; 64]);
    let err = sample_prime_v1(&mut bad, PrimeSamplerPurpose::IntEvalSmallP, 8, &mut l).unwrap_err();
    assert_eq!(
      err,
      SpartanError::PrimeSampler {
        kind: PrimeSamplerErrorKind::CandidateCapExceeded { trial: T_MAX },
        record_index: 2,
      }
    );
    let records = l.finish_exact().unwrap();
    assert_eq!(records.len(), 3);
    assert!(records[0].is_success());
    assert_eq!(records[0].purpose(), PrimeSamplerPurpose::RuntimeP);
    assert!(records[1].is_success());
    assert_eq!(records[1].width_bits(), 40);
    assert!(!records[2].is_success());
  }

  /// Schedule helpers: the `usize -> u16` width conversion and the checked
  /// sum both fail typed.
  #[test]
  fn schedule_helpers_fail_typed() {
    assert_eq!(schedule_width_bits(64).unwrap(), 64);
    assert!(matches!(
      schedule_width_bits(1 << 20),
      Err(SpartanError::InvalidInputLength { .. })
    ));
    assert_eq!(checked_schedule_add(1, 2).unwrap(), 3);
    assert!(matches!(
      checked_schedule_add(usize::MAX, 1),
      Err(SpartanError::PrimeAuditLog { .. })
    ));
  }

  /// `PrimeAuditedOutcome` accessors.
  #[test]
  fn audited_outcome_accessors() {
    let ok: PrimeAuditedOutcome<u8> = PrimeAuditedOutcome::Success {
      value: 7,
      records: Vec::new(),
    };
    assert!(ok.is_success());
    assert!(ok.records().is_empty());
    assert_eq!(ok.into_result().unwrap(), 7);
    let bad: PrimeAuditedOutcome<u8> = PrimeAuditedOutcome::Failure {
      source: SpartanError::InvalidFieldContext,
      records: Vec::new(),
    };
    assert!(!bad.is_success());
    assert_eq!(
      bad.into_result().unwrap_err(),
      SpartanError::InvalidFieldContext
    );
  }
}
