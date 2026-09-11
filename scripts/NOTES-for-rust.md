# Runner <-> bench contract (`scripts/poseidon_runner.py` <-> `benches/poseidon_modp.rs`)

limber adaptation of the Zinc+ runner contract (Zinc plan v10 section 9; limber amendment
contract sections 1, 4, 5, 6). This is the file the Rust side reads; the machine-readable
parts are pinned in the shared `specs/poseidon/timing-schema-v2.json` (`argv_forms`,
`closed_environment`) and `tuning-protocol-v2.json` (`systems.limber`). The runner never
hard-codes spec digests and treats any deviation as a recorded gate/preflight failure, never
as success. **The bench takes no environment inputs** except `RAYON_NUM_THREADS`, which it
validates against the config (and asserts through `rayon::current_num_threads()`); the
legacy `BDPCS`/`KSWEEP`/`PSIZE`/`HASHES`/`IMOD_K`/`BDK`/`POSEIDON_*` knobs and the
`POSEIDON_RUN_DIR` handshake are gone, and the seven repo knobs (`BDDIRECT BDSPEC BDROWLEN
BDSPLIT CHAIN_BITS GKRSKIP RUST_LOG`) must be absent.

## 1. argv router (`harness = false, test = false`; `main` is unconditional)

Let `args` be `argv[1..]`.

- `args == []` (what `cargo test --all-targets` does): **smoke** - serialize the metadata
  to canonical JSON and parse it back, assert `PRIMARY_GROUPS == ["prove_e2e","verify_core"]`
  and the literal diagnostic order, parse an empty synthetic run/child config and assert it
  is rejected. No env reads, no proof, no Criterion. Exit 0.
- Otherwise the last token must be exactly `--bench` (Cargo-injected), exactly once; strip
  it. The remaining tokens must match exactly one form (literal order); anything else ->
  stderr `poseidon_modp: usage error: ...`, exit 64.
  1. `["--print-protocol-metadata"]` -> exactly one canonical JSON object (sorted keys,
     compact, ASCII) plus one LF on stdout, nothing else on stdout, exit 0.
  2. `["--run-config", P, "--config-sha256", H, "--attempt", "preflight", "--artifact-dir", D]`:
     `P` absolute regular file; `H` 64 lower hex == SHA-256 of the bytes of `P`; `D` absolute,
     existing, **empty** directory with no symlink component. Runs the preflight attempt for
     `config.mode` and writes only into `D`. Always writes `attempt-result.json` **last**.
     Exit 0 on success, 3 on a recorded (fail-closed) preflight failure, 64 on usage /
     validation errors before any output.
  3. `["--child-config", P, "--child-config-sha256", H, "--artifact-dir", D]`: same
     path/hash rules (`D` is empty; the runner keeps `child-config.json` in `D`'s parent).
     Registers one primary group, the diagnostic groups in literal order, or the one sweep
     group and runs Criterion with `output_directory(D/criterion)`. Exit 0 / 64.

Criterion object (form 3 only): `Criterion::default().sample_size(10).warm_up_time(1s)
.measurement_time(20s).output_directory(D.join("criterion"))`, then `final_summary()`.
Never `configure_from_args`.

## 2. Files the runner writes for the bench

### 2.1 `run-config.json` (parent, immutable canonical JSON; SHA-256 = `config_id`)

Keys the bench reads and validates (the runner records many more keys; the bench ignores
unknown keys at every level and never re-serializes the object except to re-canonicalize
`child-config.parent_config` for the digest check):

```
schema                   "limber/poseidon-run-config/v2"
mode                     "normal" | "sweep" | "psize"
system_role              "hyrax" | "brakedown"
backend                  == system_role
k                        7..=13 | null (null only in sweep)
dimensions               {log_cons, log_vars} of k for H = 10 | null (sweep)
dimensions_by_candidate  {"7": {log_cons, log_vars}, ..., "13": {...}} copied from the
                         compiled `dimensions_by_k` (string keys)
workload                 {backend, k, hashes_per_field: 10, instance: int|null, threads: int}
                         (sweep: also candidates, instances, simplicity_order,
                         dimensions_by_candidate)
sweep                    null | {candidates: [10,7,12,9,13,8,11] literal order,
                                 simplicity_order: [7..13], instances: [1..9],
                                 dimensions_by_candidate, blocks: 14,
                                 block_orders: [{block, order}], schedule: [{block, ordinal,
                                 candidate: int, instance}]}
check_modes              {shadow: "not_applicable", timed: "release"}
criterion                {sample_size: 10, warm_up_time_s: 1, measurement_time_s: 20}
criterion_policy         {..., primary_order, diagnostic_order (per backend)}
tuning                   {tuning_id: hex|null, tuning_epoch_id: hex|null}
comparison_key           hex | null                      (null only in sweep)
ids.compiled             the exact object printed by --print-protocol-metadata; the bench
                         requires equality with its own values (exit 64 otherwise)
ids.benchmark_coins_policy {id: "poseidon-bench-chacha20-v1", framing: <tuning protocol
                         systems.limber.benchmark_coins.framing>}
features                 [] (always: limber has no bench feature)
rustflags                "-C target-cpu=native"
preflight_requirements   {check_mode: "not_applicable", deterministic_coins:
                         "poseidon-bench-chacha20-v1", double_construction: true,
                         p0d_audit: true, rayon_threads_asserted: T}
```

The bench requires `workload.hashes_per_field == 10`, `workload.threads == RAYON_NUM_THREADS
== rayon::current_num_threads()`, `tuning.tuning_id`/`k` equal to its compiled tuned default
for `backend` (`TUNED_DEFAULTS` of `src/poseidon_tuned_defaults.rs`; pre-epoch: null / the
v9 default `k = 9`) and `tuning.tuning_epoch_id == TUNING_EPOCH_ID`, `features == []`,
`rustflags == "-C target-cpu=native"`, `ids.compiled` deep-equal to the metadata and
`dimensions_by_candidate` equal to its own `dimensions_by_k` for the listed candidates.
Validation errors before any output print `poseidon_modp: validation error: ...` on stderr
and exit 64. The bench refuses symlinked path components; the runner passes realpath'ed
absolute paths.

### 2.2 `child-config.json` (per child, canonical JSON; digest in the argv)

```
schema                 "limber/poseidon-child-config/v2"
session_id             hex | null (sweep)
parent_config_sha256   hex  (== SHA-256 of the canonical parent run-config bytes)
parent_config          the complete parent run-config object
system_role            "hyrax" | "brakedown"
coordinate             {kind:"primary", metric:"prove_e2e"|"verify_core", block:0..5}
                     | {kind:"diagnostic", metric:null, block:"diag"}
                     | {kind:"sweep", block:int, ordinal:int, candidate:int k, instance:1..9}
order                  ["zinc","hyrax","brakedown"] permutation (normal) | null (sweep)
ordinal                int, 1-based position in the parent's own child sequence
                       (1..13 normal, 1..882 sweep)
```

Exactly these eight keys. A sweep child measures only `prove_e2e` for `(k, instance)`; a
primary child registers exactly its one metric; the diagnostic child registers `setup`,
`advice` (hyrax only), `commit_witness`, `prove_after_input_commit` in that order (three
groups for brakedown). Benchmark coins: seed = BLAKE3(`"limber-poseidon2-v1/bench-coins/v1\0"
|| backend_u8 || candidate_index_le32 || instance_le32`) with candidate_index = the position
of k in the pinned order (normal: of the resolved k), `ChaCha20Rng::from_seed`.

Criterion IDs (contract section 5): group_id = group name, value_str empty, function_id
`{backend}/mixed3/Hpf{H}-total{3H}/c2^{log_cons}v2^{log_vars}/k{k}/inst{i}/thr{T}/primary/blk{b}`
or `.../diagnostic/blkdiag`, where `log_cons`/`log_vars` are `dimensions_by_candidate[k]`.

## 3. Files the bench writes (canonical JSON: sorted keys, compact, ASCII, one LF, no floats)

### 3.1 `attempt-result.json` (form 2, always written last, success or failure)

```
schema        "limber/poseidon-attempt-result/v2"
attempt       "preflight"
mode          as config
config_sha256 hex (== H)
status        "ok" | "failed"
outputs       [{name, size, sha256}] for every other file written into D, sorted by name
error         null | {stage, code, message}
compiled      the metadata object (exactly what form 1 prints)
```

Only the allowlisted files per mode may be written: normal/psize `preflight.json` +
`proof-size.json`; sweep `sweep-metadata.json`. Nothing else (no subdirectories, no logs).

### 3.2 `preflight.json` (`limber/poseidon-preflight/v2`) - normal and psize

```
schema, config_sha256 (== H), status: "ok"|"failed", error: null|{stage, code, message},
backend, k, hashes_per_field, instance, dimensions,
rayon_threads_asserted: int (== workload.threads),
prime_audit: {complete: true, prime_sampler_id, invocations: int (== len(records) >= 1),
              records: [{purpose, width_bits, candidates, bases_accepted, bases_rejected,
                         mr_rounds_completed, rolling_digest, outcome}]},
double_construction: {commitments_equal, components_equal, remainder_equal, audits_equal},
digests: [3 hex], canonical_io_ok: true, shape_satisfiable: true
```

The runner requires `status == "ok"`, `rayon_threads_asserted == threads`,
`prime_audit.complete`, `prime_audit.prime_sampler_id` == the compiled id, `invocations ==
len(records)` (<= 4096), all four double-construction flags, `canonical_io_ok`,
`shape_satisfiable`, three hex digests and the workload fields equal to the config. On a
failure write the file with `status: "failed"`, skip `proof-size.json`, write
`attempt-result.json` with `status: "failed"`, exit 3.

### 3.3 `proof-size.json` (`limber/poseidon-proof-size/v2`) - normal and psize

```
schema, metric_kind: "mixed_payload_estimate", config_sha256, comparison_key (copied from
the config), wire_id, backend, k, hashes_per_field, instance,
components: {commitments_bytes: int, eval_arg_bytes: int, commitments_sha256, eval_arg_sha256},
analytical_remainder: {structured sumcheck remainder counts / bit widths},
prime_sampler_id, verified: true
```

Always written on success; the runner copies the file byte for byte. A psize reproduction
must reproduce every field except `config_sha256`.

### 3.4 `sweep-metadata.json` (`limber/poseidon-sweep-metadata/v2`) - sweep

```
schema, config_sha256, candidates (== config.sweep.candidates), instances,
matrix: [{candidate: int k, instance, dimensions, status: "ok"|"failed",
          preflight: <full 3.2 object>}]
```

One bench attempt evaluates the **complete** k x instance matrix (7 x 9 entries, each pair
exactly once) even after a failure; then `attempt-result.status = "failed"` and `error`
names the first failing pair, exit 3.

## 4. Compiled metadata (`--print-protocol-metadata`)

Required keys (the runner fails hard, `MetadataMalformed`, when any is absent):
`transcript_domain_separator`, `protocol_wire_id`, `prime_sampler_id`, `prime_sampler_caps`,
`benchmark_coins_id`, `security_accounting_id`, `timing_schema_id`, `tuning_protocol_id`,
`comparison_schema_id`, `kat_fixture_sha256`, `tuning_corpus_id`, `criterion_version`,
`panic_strategy`, `debug_assertions`, `overflow_checks`, `argv_forms_version`, `variants`,
`primary_groups`, `diagnostic_groups`, `tuning_epoch_id`, `tuning_group_set`,
`tuned_defaults`, `check_mode`, `dimensions_by_k`. `common_lift_binding_id` (null) is
optional; `checked_flag`/`unchecked_flag` must be absent (recorded failure when present).

Required values for `execution_valid` (recorded failure reasons otherwise):
`panic_strategy == "unwind"`, `debug_assertions == false`, `overflow_checks == false`,
`argv_forms_version == 2`, `check_mode == "not_applicable"`, `variants ==
["hyrax","brakedown"]`, `primary_groups == ["prove_e2e","verify_core"]`, `diagnostic_groups
== ["setup","advice","commit_witness","prove_after_input_commit"]`, `prime_sampler_caps ==
{t_max: 4096, mr_rounds: 72, base_draw_max: 160, max_invocations: 4096}`,
`kat_fixture_sha256 == db2d7fd7...bed0bd`, `criterion_version == "0.7.0"`; approved
capability tuple `("limber/wire/v-p0d",
"limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096", "poseidon-bench-chacha20-v1")`.
`dimensions_by_k` = `{"7": {log_cons, log_vars}, ..., "13": {...}}` for H = 10.
`tuned_defaults` = the generated module's `TUNED_DEFAULTS` as `[{backend, k, tuning_id}]`
(the runner also accepts `[[backend, k, tuning_id]]`). The four archived spec IDs are the
exact-byte SHA-256 of `specs/poseidon/{timing-schema-v2,tuning-protocol-v2,
security-accounting-v1}.json` and `tests/data/tune-corpus-v1.json`; `comparison_schema_id`
that of `specs/poseidon/comparison-schema-v2.json`.

## 5. Canonical commands the runner issues (`cargo` = `<SYSROOT>/bin/cargo`)

```
evidence : cargo bench --locked --offline --profile bench --color never --message-format=json-render-diagnostics --bench poseidon_modp --no-run -vv
metadata : cargo bench --locked --offline --profile bench --color never --bench poseidon_modp -vv -- --print-protocol-metadata
preflight: cargo bench --locked --offline --profile bench --color never --bench poseidon_modp -vv -- --run-config <abs> --config-sha256 <hex> --attempt preflight --artifact-dir <abs>
child    : cargo bench --locked --offline --profile bench --color never --bench poseidon_modp -vv -- --child-config <abs> --child-config-sha256 <hex> --artifact-dir <abs>
kat gate : cargo test --locked --offline --color never --lib -vv -- --exact poseidon2::tests::kat_gate
corpus   : cargo test --locked --offline --color never --lib -vv -- --exact poseidon2::tests::tuning_corpus_gate
bootstrap: cargo metadata --locked --offline --format-version 1 --filter-platform <host>
```

No command ever carries `-p` or `--features`; `T > 1` changes only `RAYON_NUM_THREADS` and
makes the run exploratory (ineligible). Every subprocess runs in the closed environment of
`poseidon_build_env` (allowlist as in the shared timing schema). Cargo appends `--bench`
after the user arguments in every form.

## 6. Generated defaults module (`src/poseidon_tuned_defaults.rs`)

Rendered only by `scripts/apply_poseidon_tuning.py` (`--initial` before the first epoch);
included from the crate with `#[rustfmt::skip] pub mod poseidon_tuned_defaults;` (the file
itself carries no `#![rustfmt::skip]`). It exports `TUNING_EPOCH_ID: Option<&str>`,
`TUNING_GROUP_SET: Option<&str>` (`"limber-both"`) and `TUNED_DEFAULTS: &[(&str, usize,
&str)]` = `[("hyrax", k, "tuning_id"), ("brakedown", k, "tuning_id")]`; the bench consults it
for the default k (v9 default 9 when empty) and prints the three values in the metadata.

## 7. Still open on the Rust/integration side

- The rustc argv grammar of the `poseidon_modp` bench is not yet captured: the shared
  timing schema has `rustc_argv_grammar.roles.zinc` only, and `poseidon_build_env` fails
  closed (`RustcArgvInvalid`) until `roles.limber.features.none.tokens` is pinned from a
  real `-vv` evidence build (same normalization rules as Zinc's).
- `dimensions_by_k` in the metadata, `dimensions` in preflight/sweep-matrix entries and the
  `tuned_defaults` object form are the shapes the runner and its stub bench assume.
