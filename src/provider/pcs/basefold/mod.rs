//! BaseFold multilinear polynomial commitment (hash-based, field-agnostic).
//!
//! Work in progress. Built to give the hash-mode backend `O(log²n)` proofs
//! (vs Brakedown's `O(√n)`, ~9 MB), while staying on our sampled ~256-bit
//! field where WHIR/FRI cannot go (they need a smooth/FFT-friendly field;
//! BaseFold's random foldable code does not). Reuses the Brakedown Merkle
//! tree and deterministic sampling.
//!
//! Status: the cryptographic CORE is complete and validated —
//! - `code`: field-agnostic foldable code (encode/fold, evaluation basis)
//! - `query`: FRI-style proximity IOP (Merkle commit + fold/consistency)
//! - `open`: full multilinear opening (eq-sumcheck interleaved with the fold)
//! All by round-trip + tamper tests + the code's defining invariants.
//!
//! Next (production wiring): a `CommitBackend` impl over the crate
//! `ByteTranscript` with multi-target batching + blinds, then Mod-PCS
//! integration (route the chunk-oracle / range-check opening through here).

pub mod code;
pub mod open;
pub mod query;
