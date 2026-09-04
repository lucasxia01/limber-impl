//! BaseFold multilinear polynomial commitment (hash-based, field-agnostic).
//!
//! Work in progress. Built to give the hash-mode backend `O(log²n)` proofs
//! (vs Brakedown's `O(√n)`, ~9 MB), while staying on our sampled ~256-bit
//! field where WHIR/FRI cannot go (they need a smooth/FFT-friendly field;
//! BaseFold's random foldable code does not). Reuses the Brakedown Merkle
//! tree and deterministic sampling.
//!
//! Status: foldable code (`code`) landed + tested. Next: Merkle codeword
//! commit, the folding open/verify, then a `CommitBackend` impl.

pub mod code;
pub mod query;
