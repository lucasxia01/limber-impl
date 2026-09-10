//! benches/poseidon_spartan.rs
//! The same 30-hash Poseidon2 workload as `benches/poseidon_modp.rs` —
//! three independent `H`-compression chains over BN254-Fr, BLS12-381-Fr,
//! and secp256k1-Fr in the fixed `FIELD_ORDER` — proven as ONE limb-emulated
//! circuit under classic Spartan (`SpartanSNARK<T256HyraxEngine>`); the
//! emulated-field baseline of `plan/poseidon_spartan_bench.md`.
//!
//! Flags (strict grammar, parsed by the crate-side
//! `limber::poseidon2_spartan::SpartanRunRequest` — see plan §6):
//!   HASHES=<n>               hashes PER FIELD; default 10, 1..=10
//!   POSEIDON_ALLOW_LARGE=1   lift the runner cap (noncanonical run)
//!   PSIZE=1                  labelled proof-size block, no Criterion
//!   POSEIDON_ALLOW_DIRTY=1   permit a non-publishable exploratory run
//! `POSEIDON_RESOURCE_ACK` belongs to the runner's preflight child, never
//! to this process. The presence of `BDPCS`, `IMOD_K`, `BDK`, `KSWEEP`,
//! `POSEIDON_ALLOW_KNOBS`, or any of the seven repo knobs (`RUST_LOG`
//! included) is an error: this suite has no backends, no `k`, no sweep.
//!
//! Lifecycle (plan §4, §6): the managed runner's dedicated preflight child
//! proves and exits BEFORE this process starts; this process re-synthesizes
//! the shape exactly once, asserts resolved-config equality, and only then
//! registers groups (`timing`) or measures sizes (`proof_size`). While the
//! §4 canonical ceilings are unset, the `H = 10` roles fail closed into
//! proposal/checkpoint handling — which lives in the runner, so this
//! process refuses those roles with a pointer to the runner.
//!
//! Measured groups, in pinned order, each configured group-level with
//! `sample_size(10) / warm_up 1 s / measurement 20 s / SamplingMode::Flat`:
//!   setup/      `setup_poseidon_spartan` (generic setup + typed binding)
//!   prove_e2e/  fresh `prep_prove` + `prove` per sample — the headline
//!               pair for modp's `prove_e2e`
//!   verify/     `verify_poseidon_spartan` (binding + proof + canonicality)
//!
//! Fixture lifetime order: setup runs with only its input fixture (the
//! circuit); the proving fixture (prover key) is constructed after the
//! setup group finishes; the verification fixture (typed key + one untimed
//! proof) is constructed only for `verify`.
//!
//! No zero-knowledge claim is made for this driver (plan §0).
#[cfg(feature = "jem")]
use tikv_jemallocator::Jemalloc;
#[cfg(feature = "jem")]
#[global_allocator]
static GLOBAL: Jemalloc = tikv_jemallocator::Jemalloc;

use criterion::{BatchSize, Criterion, SamplingMode};
use limber::{
  bellpepper::{r1cs::SpartanShape, shape_cs::ShapeCS},
  poseidon_bench::{
    ENV_CONFIG_PATH, ENV_CONFIG_SHA256, ENV_RUN_DIR, protocol_bytes_from_full_config,
  },
  poseidon2::{FIELD_ORDER, Poseidon2ParamsSet, build_all_params, build_inputs, expected_chain},
  poseidon2_spartan::{
    ExecutionRole, Poseidon2SpartanCircuit, SpartanBenchMode, SpartanResolvedConfig,
    SpartanRunRequest, build_circuit, hard_safety_precheck, resolve_spartan_run,
    setup_poseidon_spartan, verify_poseidon_spartan,
  },
  provider::T256HyraxEngine,
  spartan::SpartanSNARK,
  traits::snark::R1CSSNARKTrait,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, hint::black_box, path::PathBuf, time::Duration};

type E = T256HyraxEngine;

/// `is_small` for every `prep_prove`/`prove` call: only boundary limbs are
/// proven to fit `u64`; convolution, quotient, remainder, and equality
/// witnesses do not satisfy the machine-word promise (plan §3).
const IS_SMALL: bool = false;

/// Managed-run state from the runner handshake, if any.
struct Managed {
  run_dir: PathBuf,
  config12: String,
}

/// Atomic JSON write: temp file in the target directory, then rename.
fn write_json_atomic(path: &PathBuf, value: &serde_json::Value) {
  let bytes = limber::poseidon_bench::canonical_json_bytes(value);
  let tmp = path.with_extension("json.tmp");
  std::fs::write(&tmp, &bytes).expect("sidecar write");
  std::fs::rename(&tmp, path).expect("sidecar rename");
}

/// Verify the runner handshake: recompute the config file's SHA-256,
/// compare with the passed full hash, and byte-compare the canonical
/// protocol subsection against this process's own resolution. Returns
/// `None` (unmanaged) when no run directory is set.
fn handshake(resolved: &SpartanResolvedConfig) -> Option<Managed> {
  let run_dir = std::env::var_os(ENV_RUN_DIR)?;
  let config_path = std::env::var(ENV_CONFIG_PATH).expect("run dir set but no config path");
  let passed_hash = std::env::var(ENV_CONFIG_SHA256).expect("run dir set but no config hash");
  let file = std::fs::read(&config_path).expect("run-config readable");
  let mut h = Sha256::new();
  h.update(&file);
  let actual: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
  assert_eq!(
    actual, passed_hash,
    "run-config file hash does not match the hash the runner passed"
  );
  let file_protocol =
    protocol_bytes_from_full_config(std::str::from_utf8(&file).expect("run-config is UTF-8"))
      .expect("run-config protocol subsection");
  assert_eq!(
    file_protocol,
    resolved.protocol_canonical_bytes(),
    "run-config protocol subsection does not match this process's own resolution"
  );
  Some(Managed {
    run_dir: PathBuf::from(run_dir),
    config12: passed_hash[..12].to_string(),
  })
}

/// Benchmark ID:
/// `spartan-hyrax/mixed3-emulated/Hpf{H}-total{3H}/c2^{lc}v2^{lv}/cfg-{12}`.
/// Every dimension comes from the resolved config, never the raw request.
fn bench_id(resolved: &SpartanResolvedConfig, config12: &str) -> String {
  let lc = resolved.padded_cons.ilog2();
  let lv = resolved.padded_vars.ilog2();
  format!(
    "spartan-hyrax/mixed3-emulated/Hpf{}-total{}/c2^{lc}v2^{lv}/cfg-{config12}",
    resolved.request.hashes, resolved.request.total_hashes
  )
}

/// Host-side reference agreement: the circuit's claimed digests equal the
/// reference chain outputs (which the test suite pins against the KAT
/// fixture). No proof here — the managed preflight child already proved.
fn assert_reference_agreement(set: &Poseidon2ParamsSet, circuit: &Poseidon2SpartanCircuit<E>) {
  let messages = build_inputs(circuit.hashes_per_field()).expect("inputs build");
  for (f, field) in FIELD_ORDER.iter().enumerate() {
    let chain = expected_chain(set.get(*field), &messages).expect("reference chain");
    assert_eq!(
      &circuit.digests()[f],
      chain.last().expect("nonempty chain"),
      "reference digest mismatch for block {}",
      field.name()
    );
  }
}

/// The timing groups, in the pinned literal order `[setup, prove_e2e,
/// verify]`, with the §6 fixture lifetime order.
fn run_timing(
  c: &mut Criterion,
  resolved: &SpartanResolvedConfig,
  circuit: &Poseidon2SpartanCircuit<E>,
  config12: &str,
) {
  let id = bench_id(resolved, config12);

  // setup/: only its input fixture (the circuit) exists. The key pair is
  // returned from the closure so destruction is untimed.
  {
    let mut g = c.benchmark_group("setup");
    g.sample_size(10)
      .warm_up_time(Duration::from_secs(1))
      .measurement_time(Duration::from_secs(20))
      .sampling_mode(SamplingMode::Flat);
    g.bench_function(&id, |b| {
      b.iter_batched(
        || circuit.clone(),
        |circuit| black_box(setup_poseidon_spartan::<E>(circuit).expect("setup")),
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // Proving fixture: constructed only after the setup group finished, so
  // setup timing never coexists with another shape-bearing key pair.
  let (pk, vk) = setup_poseidon_spartan::<E>(circuit.clone()).expect("typed setup");

  // prove_e2e/: a fresh prep_prove immediately followed by prove, from the
  // reusable prover key — the headline pair for modp's prove_e2e. Both the
  // proof and the returned prep state outlive the timer.
  {
    let mut g = c.benchmark_group("prove_e2e");
    g.sample_size(10)
      .warm_up_time(Duration::from_secs(1))
      .measurement_time(Duration::from_secs(20))
      .sampling_mode(SamplingMode::Flat);
    g.bench_function(&id, |b| {
      b.iter_batched(
        || (circuit.clone(), circuit.clone()),
        |(prep_circuit, prove_circuit)| {
          let prep =
            SpartanSNARK::<E>::prep_prove(&pk, prep_circuit, IS_SMALL).expect("prep_prove");
          let (proof, prep_back) =
            SpartanSNARK::<E>::prove(&pk, prove_circuit, prep, IS_SMALL).expect("prove");
          black_box((proof, prep_back))
        },
        BatchSize::PerIteration,
      );
    });
    g.finish();
  }

  // Verification fixture: one untimed proof, created only for verify/.
  let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), IS_SMALL).expect("prep_prove");
  let (proof, prep_back) =
    SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, IS_SMALL).expect("prove");
  drop(prep_back);
  drop(pk);

  {
    let mut g = c.benchmark_group("verify");
    g.sample_size(10)
      .warm_up_time(Duration::from_secs(1))
      .measurement_time(Duration::from_secs(20))
      .sampling_mode(SamplingMode::Flat);
    g.bench_function(&id, |b| {
      b.iter(|| black_box(verify_poseidon_spartan(&vk, &proof).expect("verify")));
    });
    g.finish();
  }

  // Reference agreement on the verified digests (host-side comparison).
  let digests = verify_poseidon_spartan(&vk, &proof).expect("verify");
  assert_eq!(&digests, circuit.digests(), "verified digests drifted");
}

/// PSIZE mode: one labelled component block from measured serialization,
/// plus the `proof-size.json` sidecar; exits without Criterion groups.
fn run_psize(
  resolved: &SpartanResolvedConfig,
  circuit: &Poseidon2SpartanCircuit<E>,
  managed: &Option<Managed>,
) {
  let (pk, vk) = setup_poseidon_spartan::<E>(circuit.clone()).expect("typed setup");
  let prep = SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), IS_SMALL).expect("prep_prove");
  let (proof, _prep) =
    SpartanSNARK::<E>::prove(&pk, circuit.clone(), prep, IS_SMALL).expect("prove");
  drop(pk);
  let digests = verify_poseidon_spartan(&vk, &proof).expect("verify");
  assert_eq!(&digests, circuit.digests(), "verified digests drifted");

  let sizes = proof.component_sizes().expect("proof-size serialization");
  println!(
    "spartan-hyrax/mixed3-emulated/Hpf{}-total{}: proof size\n  \
     instance commitments                       : {} B\n  \
     protocol challenges                        : {} B\n  \
     outer + inner sumchecks + outer claims     : {} B\n  \
     eval_W + blind + Hyrax eval argument       : {} B\n  \
     comparison payload, public values excluded : {} B\n  \
     public values (12-limb encoded Vec)        : {} B\n  \
     wire SpartanSNARK total                    : {} B",
    resolved.request.hashes,
    resolved.request.total_hashes,
    sizes.instance_commitments,
    sizes.protocol_challenges,
    sizes.sumchecks_and_claims,
    sizes.evaluation_opening,
    sizes.comparison_payload,
    sizes.public_values,
    sizes.wire_total,
  );
  let doc = serde_json::json!({
    "suite": "spartan",
    "circuit": "mixed3-emulated",
    "hashes_per_field": resolved.request.hashes,
    "total_hashes": resolved.request.total_hashes,
    "combined": {
      "instance_commitments": sizes.instance_commitments,
      "protocol_challenges": sizes.protocol_challenges,
      "sumchecks_and_claims": sizes.sumchecks_and_claims,
      "evaluation_opening": sizes.evaluation_opening,
      "comparison_payload": sizes.comparison_payload,
      "public_values": sizes.public_values,
      "wire_total": sizes.wire_total,
      "convention": "comparison payload excludes the serialized public values (statement data)",
    },
  });
  if let Some(m) = managed {
    write_json_atomic(&m.run_dir.join("proof-size.json"), &doc);
  }
}

fn main() {
  let env_map: BTreeMap<std::ffi::OsString, std::ffi::OsString> = std::env::vars_os().collect();
  let request = match SpartanRunRequest::parse(&env_map) {
    Ok(c) => c,
    Err(e) => {
      eprintln!("poseidon_spartan bench: {e}");
      std::process::exit(1);
    }
  };
  if let Err(e) = hard_safety_precheck(request.hashes) {
    eprintln!("poseidon_spartan bench: {e}");
    std::process::exit(1);
  }
  // This process runs only the measuring roles; the proposal/checkpoint
  // roles live in the runner's preflight child (which also fails closed
  // when the §4 ceilings are unset for canonical H).
  match request.execution_role(None) {
    Ok(ExecutionRole::Timing | ExecutionRole::ProofSize) => {}
    Ok(role) => {
      eprintln!(
        "poseidon_spartan bench: execution role {} belongs to the managed runner \
         (scripts/run_poseidon_spartan_bench.sh); this process only times or sizes",
        role.name()
      );
      std::process::exit(1);
    }
    Err(e) => {
      eprintln!("poseidon_spartan bench: {e}");
      std::process::exit(1);
    }
  }

  if request.allow_large {
    println!(
      "POSEIDON_ALLOW_LARGE: H = {} per field ({} total); this run is noncanonical, \
       and admission rests on the hard-safety precheck plus the requested-shape \
       synthesis below",
      request.hashes, request.total_hashes
    );
  }

  // One shape synthesis, frozen into the resolved config; every ID and
  // sidecar derives from it.
  let set = build_all_params().expect("params build");
  let circuit = build_circuit::<E>(&set, request.hashes).expect("circuit build");
  let shape = ShapeCS::r1cs_shape(&circuit).expect("shape synthesis");
  let resolved = resolve_spartan_run(request, &set, &shape).expect("run resolution");
  drop(shape);
  assert_reference_agreement(&set, &circuit);

  let managed = handshake(&resolved);
  if managed.is_none() {
    eprintln!(
      "poseidon_spartan bench: unmanaged run (no {ENV_RUN_DIR}); published results must go \
       through scripts/run_poseidon_spartan_bench.sh"
    );
  }

  let config12 = match &managed {
    Some(m) => m.config12.clone(),
    None => {
      let mut h = Sha256::new();
      h.update(resolved.protocol_canonical_bytes());
      h.finalize()[..6]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
    }
  };

  match resolved.request.mode {
    SpartanBenchMode::ProofSize => run_psize(&resolved, &circuit, &managed),
    SpartanBenchMode::Normal => {
      let mut c = Criterion::default().configure_from_args();
      if let Some(m) = &managed {
        c = c.output_directory(&m.run_dir.join("criterion"));
      }
      run_timing(&mut c, &resolved, &circuit, &config12);
      c.final_summary();
    }
  }
}
