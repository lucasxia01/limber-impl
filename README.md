# Limber: Low Overhead SNARKs for Integers

In this repo, we build SNARKs for Integers over Mod R1CS, where constraints support arbitrary modular arithmetic.
It implements the protocol from the accompanying paper Limber: Low Overhead SNARKs for Integers from Any PCS.

This repo is forked from [Microsoft Spartan2](https://github.com/Microsoft/Spartan2), and we accordingly build Limber-Spartan with various choices of underlying PCS, including Hyrax and Brakedown.

## Integer Mod-R1CS Arithmetization

The Integer Mod-R1CS relation is different from the standard R1CS relation, adding a modulus `m_i` and a quotient `u_i` per constraint:

```text
A·z ∘ B·z = C·z + m ∘ u        (over the bounded integers)
```

so one row is one modular multiplication.
This naturally and efficiently supports large and varying modular arithmetic gates.

## Fingerprinting

Following [Zaratan](https://eprint.iacr.org/2024/1548), proving this Integer Mod-R1CS relation over the integers can be reduced to proving it over a randomly sampled prime modulus $p$ (fingerprinting).
However, we now need to be able to construct a PCS that is able to commit to **integers** in one field $\mathbb{F}_q$ and open in another $\mathbb{F}_p$.
We call this an **integer mod-PCS**.

## Limber: An Integer Mod-PCS Compiler

Limber is a compiler from **almost any** field PCS to an integer mod-PCS with $o(1)$ multiplicative commitment overhead.
The only requirement on the underlying field is $q = \Omega(\lambda^2 \mu^2)$ for a polynomial in $\mu$ variables at $\lambda$ bits of security, so it works over small fields (64-bit). The compiled scheme inherits the assumptions and performance of whatever PCS it wraps.
It has two algorithms:

- **Commit.** An integer polynomial $f$ with hypercube evaluations bounded by $T_f$ is limb-split into a polynomial $f_{\mathsf{limb}}$ whose entries are bounded by a base bound $T = 2^{64}$, cast into $\mathbb{F}_q$, and committed with the underlying PCS.
A batched range check (LogUp-GKR) proves every limb is below $T$.

- **Evaluate mod $p$.** To open $f$ at a point $\vec r \in \mathbb{Z}_p^{\mu}$ to $v$, a sumcheck reduces the claim to one about $f_{\mathsf{limb}}$, the prover sends the *integer* evaluation $y \in \mathbb{Z}$, and the verifier checks $y \equiv v \pmod p$.
What remains is proving $y = f_{\mathsf{limb}}(\vec{r'})$ over the integers, which is handled by our IntEval protocol.
  - **IntEval.** The verifier samples $s$ small primes $p_i$ and the prover opens $f_{\mathsf{limb}}$ at $\vec r \bmod p_i$ for each.
  IntEval partially evaluates $k$ variables at a time to avoid overflow: the partially evaluated polynomial is decomposed as $a + p_i \cdot b$, the prover commits $a$ and $b$ (each $2^{-k}$ the size of the previous layer), and the protocol recurses on $a$.

### Parameters

The benchmarks below use the following parameters (the IntEval parameters table in the paper):

| Parameter | Hyrax | Brakedown | Meaning |
| --- | ---: | ---: | --- |
| $q$ | $\approx 2^{256}$ | $\approx 2^{256}$ | Field characteristic of the underlying PCS (Tom-256 scalar field) |
| $T$ | $2^{64}$ | $2^{64}$ | Limb base bound; every limb of $f_{\mathsf{limb}}$ is range-checked to $[0, T)$ |
| $k$ | $9$ | $11$ | Variables partially evaluated per IntEval layer; each layer shrinks by $2^{k}$ |
| $s$ | $16$ | $31$ | Number of small CRT primes $p_i$ sampled by the IntEval verifier |
| Commitment overhead | $\approx 0.13\times$ | $\approx 0.07\times$ | Extra committed data relative to the witness |

The values shown are for the MultiSwap benchmark, with $T_f = 2^{2048}$ and $N = 2^{14}$ rows. The plain Spartan comparison, with $T_f = 2^{256}$ and $N = 2^{10}$ to $2^{18}$, and the Poseidon2 benchmark, with $T_f = 2^{256}$ and $N = 2^{14}$, use $k = 9$ for both backends and derive the same $s = 15\text{–}16$ and 20-bit CRT primes.

$s$ and the commitment overhead are derived from the other three parameters and the polynomial size (`IntEvalParams::derive`).
Larger $k$ lowers the commitment overhead at the cost of more CRT primes. See the paper for more details.

## Code layout

- `src/provider/pcs/integer_modpcs.rs` — the mod-PCS compiler: commit, evaluate mod $p$, and IntEval.
- `src/provider/pcs/commit_backend.rs` — the `CommitBackend` trait for underlying PCSs.
- `src/provider/pcs/hyrax_pc.rs`, `src/provider/pcs/brakedown/` — the two PCS instantiations: Hyrax over the Tom-256 curve and Brakedown over the Tom-256 scalar field.
- `src/logup_gkr.rs` — batched LogUp-GKR range check for the limbs and decomposed polynomials in IntEval.
- `src/dyn_prime.rs`, `src/sumcheck_modp.rs`, `src/polys_modp/` — runtime-modulus field for the Fiat–Shamir-sampled prime $p$, and the sumcheck/polynomial code running over it.
- `src/imod_spartan_modp.rs` — the SNARK driver tying the Spartan-style mod-PIOP to the mod-PCS; trait surface in `src/traits/mod_engine.rs`.
- `src/prime_sampler.rs` — the audited transcript sampler for the fingerprinting prime $p$ and the IntEval primes $p_i$.
- `src/multiswap/` — the MultiSwap workload as an Integer Mod-R1CS circuit (RSA accumulator, Poseidon and MiMC gadgets, Pocklington hash-to-prime certificate).
- `src/poseidon2.rs`, `src/poseidon2_spartan/` — the three-field Poseidon2 workload as an Integer Mod-R1CS circuit, and its limb-emulated plain-Spartan baseline.

## Building and testing

Requires a stable Rust toolchain, **1.97 or newer** (declared as
`rust-version` in `Cargo.toml`, so an older cargo reports the requirement
instead of failing with confusing compile errors).

```bash
cargo build --release
cargo test --release
```

CI additionally runs `cargo fmt --all -- --check` and `cargo clippy --all-targets -- -D warnings`.

## Benchmarks

All benchmarks use [Criterion](https://github.com/bheisler/criterion.rs) and report setup / prove / verify times plus proof sizes.
Run with native CPU codegen:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench <name>
```

| Bench | What it measures |
| --- | --- |
| `imod_spartan_modp` | Limber-Spartan on various constraint counts |
| `spartan_synthetic` | Plain-Spartan baseline to compare to Limber |
| `multiswap_modp` | MultiSwap (RSA-accumulator verification circuit, [OWWB20](https://eprint.iacr.org/2019/1494)). `BDPCS=1` runs the Brakedown instantiation instead of Hyrax |
| `poseidon_modp` | 30 Poseidon2 compressions over three non-native fields (BN254-Fr, BLS12-381-Fr, secp256k1-Fr) in one Limber circuit, Hyrax or Brakedown. Driven by a run-config file rather than env knobs: see `scripts/run_poseidon_bench.sh` |
| `poseidon_spartan` | The same Poseidon2 workload as a limb-emulated circuit under plain Spartan, the circuit-based baseline: `scripts/run_poseidon_spartan_bench.sh` |
| `logup_gkr` | LogUp-GKR range proof in isolation |
| `imod_spartan`, `sha256_spartan` | Earlier Spartan-side benches (Integer Mod-R1CS over the curve field without fingerprinting; a SHA-256 circuit); not used in the paper |
| `int_mult` (example) | A wired chain of w-bit integer multiplications; `cargo run --release --example int_mult -- --bits 64 --log-gates 16` (see below) |

### Quick start: w-bit integer multiplication

To prove a chain of 32- or 64-bit integer multiplications, use the
`int_mult` example. The circuit is one wired multiplication chain
`c_i = a_i · b_i mod 2^w` with `a_{i+1} = c_i` (i.e. it proves
`c = a_0 · Π b_i mod 2^w` for random w-bit operands). `--log-gates L` creates `2^L − 1` gates and `2^(L+1)` witness variables.
To run the benchmark:

```bash
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo run --release --example int_mult -- --bits 64 --log-gates 16
```

It prints the derived parameters, per-phase times (setup, witness
generation, commit+prove, verify), and the proof size. Prover time
includes witness generation, as in all our benchmarks; the example
reports the two parts separately and their total. `DUMP=<path>` writes
the serialized eval argument so its compressed size can be measured
(`zstd -19`).


## Results

All numbers are single-threaded (`RAYON_NUM_THREADS=1`) on a MacBook (Apple M4 Pro, 24 GB RAM, 14 cores); all baselines were re-run on the same machine.

### MultiSwap benchmark
The MultiSwap [OWWB20](https://eprint.iacr.org/2019/1494) benchmark is two RSA accumulator updates (4 Wesolowski exponentiations with 322-bit exponents modulo an RSA-2048 modulus) plus a Poseidon-based hash-to-prime evaluation.

Constraint counts for Limber are integer gates; [Zinc+](https://eprint.iacr.org/2026/855)'s is the size of its execution trace; the others are ordinary R1CS constraints over a prime field.
The two circuit rows are proven with the same Spartan prover as Limber (Hyrax over BLS12-381, the field of both circuits), so the comparison isolates the arithmetization. Proof sizes are zstd-compressed.

| System | Constraints | Prove | Verify | Proof size |
| --- | ---: | ---: | ---: | ---: |
| Arkworks circuit (emulated field arithmetic) + Spartan | 26.4 M | 63.7 s | 5.74 s | 857 KB |
| MultiSwap [OWWB20](https://eprint.iacr.org/2019/1494) circuit (xJsnark techniques) + Spartan | 10.7 M | 31.5 s | 4.30 s | 463 KB |
| [Zinc+](https://eprint.iacr.org/2026/855) (full computation ported to their framework) | 2^11 × 248 trace | 10.2 s | 363 ms | 0.94 MB |
| **Limber-Spartan (Hyrax)** | 12,796 | **2.11 s** | **64 ms** | **273 KB** |
| **Limber-Spartan (Brakedown)** | 12,796 | **2.07 s** | **59 ms** | 8.2 MB |

Limber's Hyrax prover is 15× faster than the MultiSwap circuit and 30× faster than the Arkworks circuit under the same Spartan prover.
Against Zinc+, the Hyrax prover is 4.8× faster, the verifier 5.8× faster, and the proof 3.4× smaller (273 KB vs. 0.94 MB).

### Poseidon2 benchmark
Thirty Poseidon2 compressions (t = 3, α = 5, R_F = 8, R_P = 56) over three different non-native prime fields in one proof: three independent ten-hash chains over BN254-Fr, BLS12-381-Fr, and secp256k1-Fr.
The circuit baseline uses a limb-emulated field gadget (`bellpepper-emulated`, 4 × 64-bit limbs) and is proven with the same Spartan prover as Limber (Hyrax over Tom-256).

| System | Constraints | Prove | Verify | Proof size |
| --- | ---: | ---: | ---: | ---: |
| Emulated-field circuit + Spartan | 9.45 M | 10.6 s | 1.53 s | 340 KB |
| [Zinc+](https://eprint.iacr.org/2026/855) (Poseidon2 UAIR in their framework) | 2^10 × 72 trace | 528 ms | 29.8 ms | 3.4 MB |
| **Limber-Spartan (Hyrax)** | 12,990 | **438 ms** | **24.7 ms** | **136 KB** |
| **Limber-Spartan (Brakedown)** | 12,990 | **334 ms** | 33.0 ms | 3.8 MB |

Limber's Hyrax prover is 24× faster than the circuit baseline with a 62× faster verifier and a 2.5× smaller proof, and 1.2× (1.6× with Brakedown) faster than Zinc+ with a 1.2× faster verifier and a 25× smaller proof.

### Non-native overhead relative to native constraints
This experiment compares Limber against a plain-Spartan baseline with the same constraint and variable counts.
Each Limber gate is a random multiplication modulo the Tom-256 **base-field** modulus; the baseline proves native **scalar-field** gates.
Prover time includes witness generation and commitment.

| Constraints | Prove (Limber) | Prove (Spartan) | Ratio | Verify (Limber) | Verify (Spartan) | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2^10 | 113 ms | 18.0 ms | 6× | 27.8 ms | 13.9 ms | 2.0× |
| 2^12 | 235 ms | 43.5 ms | 5× | 29.4 ms | 14.5 ms | 2.0× |
| 2^14 | 699 ms | 92.8 ms | 8× | 27.8 ms | 16.1 ms | 1.7× |
| 2^16 | 2.58 s | 291 ms | 9× | 56.7 ms | 22.6 ms | 2.5× |
| 2^18 | 10.1 s | 1.08 s | 9× | 158 ms | 48.8 ms | 3.2× |

We demonstrate a low (5–9×) prover overhead for large 256-bit non-native gates compared to proving completely native constraints. Both provers are linear in the constraint count; the ratio rises with size because plain Spartan's fixed costs amortize faster.

## Reproducing the paper's numbers

All numbers quoted in the paper are **single-threaded** (`RAYON_NUM_THREADS=1`) with native codegen (`RUSTFLAGS="-C target-cpu=native"`).

### MultiSwap table (Table 1 of the paper)

Both of the Limber rows prove the full OWWB20 computation (`MSCFG=full`, the default): the 4 fully wired Wesolowski exponentiations with 322-bit exponents mod an RSA-2048 modulus, the Poseidon hashes, and the Pocklington hash-to-prime certificate resulting in 12,796 integer constraints, padded to $2^{14}$. Set `PSDUMP=<path>` with `PSIZE=1` to write the proof bytes for compression.

**Hyrax row**:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

This instantiation defaults to `k = 9`; override with `IMOD_K=<k>`.
Set `PSIZE=1` to print the proof size instead, and `KSWEEP=1` to sweep the IntEval reduction parameter `k`.

**Brakedown row**:

```bash
BDPCS=1 RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" cargo bench --bench multiswap_modp
```

This instantiation defaults to `k = 11` (faster than `k = 9` for the hash backend); override with `BDK=<k>`.

**Circuit rows (Arkworks, MultiSwap)**: both circuits are exported as R1CS over BLS12-381-Fr and proven with this repo's Spartan prover using Hyrax over BLS12-381. The MultiSwap circuit comes from [`bellman-bignat`](https://github.com/alex-ozdemir/bellman-bignat) `SetBench` (10,658,232 constraints, 322-bit challenges); the Arkworks circuit is a re-implementation of the full computation with r1cs-std emulated field arithmetic (26,445,064 constraints, synthesized in stages and stacked). Proving each needs 13–15 GB of RAM single-threaded.

**Zinc+ row**:
We ported the circuit into Zinc+'s framework ([`NethermindEth/zinc-plus`](https://github.com/NethermindEth/zinc-plus), `main-beta`) as a 2^11-row, 248-column trace with one modular multiplication per row, and ran their folded prover at code rate 1/8 and 114-bit security:

```bash
FULL=1 FOLD=1 NVARS=11 RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" \
  cargo bench --bench limber_multiswap --features "simd unchecked iprs-rate-1-8 sec-114"
```

### Poseidon2 table (Table 4 of the paper)

The Limber rows come from the `poseidon_modp` bench and the baseline from `poseidon_spartan`; both are driven by an immutable run-config file rather than environment knobs, through the wrapper scripts:

```bash
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" scripts/run_poseidon_bench.sh          # Limber, BACKEND=hyrax|brakedown
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" scripts/run_poseidon_spartan_bench.sh  # circuit baseline; PSIZE=1 for the proof size
```

Both use k = 9. The Zinc+ row is their `poseidon` bench (`qz` variant, 256-bit sampled prime, square shape; with a 320-bit prime: 546 ms / 32.3 ms / 3.1 MB) built with `--features simd` at commit `334f09e` of our fork of `zinc-plus`.

### Native-overhead figure (Figure 3 of the paper)
We use Limber-Spartan with Hyrax in this comparison. To generate the data and plots, run:
```bash
pip install matplotlib
RAYON_NUM_THREADS=1 ./scripts/regen_msshape_plots.sh
```

This runs the pair of benchmarks (`cargo bench --bench imod_spartan_modp -- msshape` vs `cargo bench --bench spartan_synthetic -- msshape`) and renders the figures via `scripts/plot_msshape.py`.
We get 5–9× prover overhead over plain Spartan at $2^{10}\text{–}2^{18}$ constraints; verify is under 30 ms vs 14–16 ms up to $2^{14}$ and 158 ms vs 49 ms at $2^{18}$.

## References
Limber: Low Overhead SNARKs for Integers from Any PCS — the protocol this repository implements.

[Spartan: Efficient and general-purpose zkSNARKs without trusted setup](https://eprint.iacr.org/2019/550) \
Srinath Setty \
CRYPTO 2020

[Scaling Verifiable Computation Using Efficient Set Accumulators](https://eprint.iacr.org/2019/1494) \
Alex Ozdemir, Riad S. Wahby, Barry Whitehat, Dan Boneh \
USENIX Security 2020

[Fully Succinct Arguments over the Integers from First Principles](https://eprint.iacr.org/2024/1548) (Zaratan) \
Matteo Campanelli, Mathias Hall-Andersen \
PKC 2026

[Zinc+: SNARKs for Polynomial Rings](https://eprint.iacr.org/2026/855) \
Alexander Abdugafarov, Albert Garreta, Amit Kumar, Michał Osadnik, Psi Vesely, Ilia Vlasov, Kai Zhe Zheng \
Cryptology ePrint Archive 2026/855

## License

MIT, inherited from the upstream [Spartan2](https://github.com/Microsoft/Spartan2) project — see [LICENSE](LICENSE).
