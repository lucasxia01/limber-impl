//! The Poseidon2 workload of [`crate::poseidon2`] proven as a limb-emulated
//! circuit under classic Spartan ([`crate::spartan::SpartanSNARK`]) — the
//! emulated-field baseline of `plan/poseidon_spartan_bench.md`.
//!
//! Everything here is benchmark/test support, not part of the supported API
//! (the parent module is `#[doc(hidden)]`). The non-native arithmetic comes
//! from the revision-pinned `bellpepper-emulated` crate; [`emulated`] owns
//! the per-field parameter types and the checked-boundary adapters that are
//! the only permitted variable-allocation entry points.

pub mod circuit;
pub mod config;
pub mod emulated;
pub mod snark;

pub use circuit::{NUM_PUBLIC_LIMBS, Poseidon2SpartanCircuit, build_circuit};
pub use config::{
  CANONICAL_CEILINGS, EMULATED_DEP_REV, ExecutionRole, REDUCTION_SCHEDULE_VERSION,
  RESOURCE_POLICY_VERSION, ResourceCeilings, SpartanBenchMode, SpartanResolvedConfig,
  SpartanRunRequest, hard_safety_precheck, resolve_spartan_run,
};
pub use snark::{PoseidonSpartanVerifierKey, setup_poseidon_spartan, verify_poseidon_spartan};
