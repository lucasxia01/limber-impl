//! One-shot prove/verify of a wired w-bit integer-multiplication chain
//! on the dual-field Mod-PCS driver, with every parameter set from the
//! command line. The circuit is a single multiplication chain
//!
//!     c_i = a_i · b_i mod 2^w,   a_{i+1} = c_i,
//!
//! i.e. it proves `c = a_0 · Π b_i mod 2^w` for random w-bit `a_0, b_i`
//! — per-gate wraparound multiplication with real dataflow. Wiring is
//! free in the Mod-R1CS matrices (row i+1's A-entry points at row i's
//! output variable), so each gate costs 2 fresh witness variables plus
//! its quotient, and `--log-gates L` packs 2^L − 1 gates into exactly
//! 2^L constraint rows and 2^(L+1) witness variables with no padding
//! waste.
//!
//! Usage:
//!   cargo run --release --example int_mult -- [--bits 32|64] [--log-gates L] [--k K]
//!
//!   --bits       operand width w in bits (default 32; any 1..=64 accepted)
//!   --log-gates  chain length exponent: 2^L − 1 gates (default 12, ~4k gates)
//!   --k          IntEval per-iteration variable count (default DEFAULT_K = 9)
//!
//! Prints the derived IntEval parameters `(log P, s, numlimb)`, per-phase
//! wall times (setup, witness generation, commit+prove, verify), and the
//! proof size. Witness generation is timed separately from commit+prove
//! so the commit+prove line is comparable to prover times reported by
//! systems that exclude witness generation (e.g. Zinc+).

use limber::{
  imod_r1cs_modp::{IntModR1CSShapeModp, IntModR1CSWitnessModp},
  imod_spartan_modp::IntModSpartanModpSNARK,
  provider::T256DynPrimeEngine,
  provider::pcs::integer_modpcs::{DEFAULT_K, IntEvalParams},
};
use num_bigint::{BigUint, RandBigInt};
use rand::SeedableRng;
use std::time::Instant;

type M = T256DynPrimeEngine;

fn usage() -> ! {
  eprintln!("usage: int_mult [--bits 32|64] [--log-gates L] [--k K]");
  std::process::exit(2);
}

fn parse_args() -> (usize, usize, usize) {
  let (mut bits, mut log_gates, mut k) = (32usize, 12usize, DEFAULT_K);
  let mut args = std::env::args().skip(1);
  while let Some(flag) = args.next() {
    let mut val = || args.next().unwrap_or_else(|| usage());
    match flag.as_str() {
      "--bits" => bits = val().parse().unwrap_or_else(|_| usage()),
      "--log-gates" => log_gates = val().parse().unwrap_or_else(|_| usage()),
      "--k" => k = val().parse().unwrap_or_else(|_| usage()),
      _ => usage(),
    }
  }
  if bits == 0 || bits > 64 || log_gates == 0 || log_gates > 24 {
    usage();
  }
  (bits, log_gates, k)
}

fn main() {
  let (bits, log_gates, k) = parse_args();
  let gates = (1usize << log_gates) - 1;

  // Shape: one chained gate per row, a_{i+1} = c_i. Variable layout:
  // [a_0, b_0..b_{gates-1}, c_0..c_{gates-1}], 2^(log_gates+1) slots.
  let num_cons = 1usize << log_gates;
  let num_vars = 1usize << (log_gates + 1);
  let modulus = BigUint::from(1u32) << bits;
  let one = BigUint::from(1u32);
  let mut a_entries = Vec::with_capacity(gates);
  let mut b_entries = Vec::with_capacity(gates);
  let mut c_entries = Vec::with_capacity(gates);
  for i in 0..gates {
    let a_idx = if i == 0 { 0 } else { 1 + gates + (i - 1) };
    a_entries.push((i, a_idx, one.clone()));
    b_entries.push((i, 1 + i, one.clone()));
    c_entries.push((i, 1 + gates + i, one.clone()));
  }
  let mods = vec![modulus.clone(); num_cons];
  let shape =
    IntModR1CSShapeModp::<M>::new(num_cons, num_vars, 0, a_entries, b_entries, c_entries, mods)
      .expect("shape");

  // Params: all committed values (a_0, b_i, c_i and quotients q_i) are
  // < 2^bits, so log T_f = bits, single-limb (log T = log T_f).
  // (log P, s) are derived to λ = 128 and validated.
  let log_n = (num_vars.max(num_cons) as u64).ilog2() as usize;
  let params = IntEvalParams::derive_no_limb_split(bits, k, log_n).expect("params satisfy bounds");
  println!(
    "int_mult: 2^{log_gates}−1 = {gates} chained gates of {bits}-bit mult  (cons=2^{}, vars=2^{})",
    num_cons.ilog2(),
    num_vars.ilog2()
  );
  println!(
    "params:   log_t_f={} k={} -> log_p={} s={} numlimb={}",
    params.log_t_f, params.k, params.log_p, params.s, params.numlimb
  );

  let t = Instant::now();
  let (pk, vk) =
    IntModSpartanModpSNARK::<M>::setup_with_params(shape.clone(), params).expect("setup");
  println!("setup:          {:9.1} ms", t.elapsed().as_secs_f64() * 1e3);

  // Witness: random w-bit a_0 and b_i, chained divmods.
  let t = Instant::now();
  let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234);
  let zero = BigUint::from(0u32);
  let mut w = vec![zero.clone(); num_vars];
  let mut q = vec![zero; num_cons];
  let mut chain_a = rng.gen_biguint_below(&modulus);
  w[0] = chain_a.clone();
  for i in 0..gates {
    let b = rng.gen_biguint_below(&modulus);
    let ab = &chain_a * &b;
    q[i] = &ab >> bits;
    let c = ab % &modulus;
    w[1 + i] = b;
    w[1 + gates + i] = c.clone();
    chain_a = c;
  }
  println!("witness gen:    {:9.1} ms", t.elapsed().as_secs_f64() * 1e3);

  let t = Instant::now();
  let (witness, instance) =
    IntModR1CSWitnessModp::<M>::new(&shape, pk.ck(), w, q, vec![]).expect("witness commit");
  let proof = IntModSpartanModpSNARK::<M>::prove(&pk, &instance, &witness).expect("prove");
  println!("commit+prove:   {:9.1} ms", t.elapsed().as_secs_f64() * 1e3);

  let t = Instant::now();
  proof.verify(&vk, &instance).expect("verify");
  println!("verify:         {:9.1} ms", t.elapsed().as_secs_f64() * 1e3);

  // Proof size: serialized Mod-PCS eval argument plus the analytical
  // dynamic-prime sumcheck bytes (not `Serialize`), as in the PSIZE
  // mode of benches/imod_spartan_modp.rs.
  let arg_bytes = proof.eval_arg_bytes().expect("eval_arg serializes");
  let arg = arg_bytes.len();
  // DUMP=<path>: write the serialized eval argument so its compressed
  // size can be measured externally (zstd -19, as for the Brakedown
  // proofs; see README). The sumcheck side is analytical and counted raw.
  if let Ok(path) = std::env::var("DUMP") {
    std::fs::write(&path, &arg_bytes).expect("write proof dump");
  }
  let dyn_bytes = (num_cons.ilog2() as usize * 3 + num_vars.ilog2() as usize * 2 + 6) * 16;
  println!(
    "proof size:  {:9.1} KB  (eval_arg {arg} B + sumcheck {dyn_bytes} B)",
    (arg + dyn_bytes) as f64 / 1e3
  );
}
