//! Pins the environment-free half of the Poseidon2 benchmark <-> runner
//! contract (`scripts/NOTES-for-rust.md`; Zinc plan v10 §9; limber
//! contract §§1, 4, 5) implemented by `benches/poseidon_modp_support.rs`:
//! the argv router forms, the run/child config parsers' acceptance and
//! rejection cases, the compiled metadata key set and its canonical JSON
//! round trip, the Criterion ID grammar, the deterministic benchmark-coins
//! seed against an independently computed BLAKE3, and the path/digest
//! validators of the file-driven forms.

#[path = "../benches/poseidon_modp_support.rs"]
mod support;

use limber::{
  poseidon_bench::{
    BACKEND_NAMES, BENCHMARK_COINS_DOMAIN, BenchBackend, CANONICAL_HASHES_PER_FIELD,
    DIAGNOSTIC_GROUPS, K_ORDER, LIMBER_PRIME_SAMPLER_ID, LIMBER_PROTOCOL_WIRE_ID,
    LIMBER_TRANSCRIPT_ID, PRIMARY_GROUPS, benchmark_coins_seed, default_k, tuned_default,
  },
  poseidon_tuned_defaults as tuned_defaults,
  poseidon2::ids,
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use regex::Regex;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use support::*;

fn args(tokens: &[&str]) -> Vec<String> {
  tokens.iter().map(|t| t.to_string()).collect()
}

const HEX64: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[test]
fn router_accepts_exactly_the_four_forms() {
  assert_eq!(route(&args(&[])), Ok(Form::Smoke));
  assert_eq!(
    route(&args(&["--print-protocol-metadata", "--bench"])),
    Ok(Form::PrintProtocolMetadata)
  );
  assert_eq!(
    route(&args(&[
      "--run-config",
      "/abs/run-config.json",
      "--config-sha256",
      HEX64,
      "--attempt",
      "preflight",
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ])),
    Ok(Form::RunConfig {
      config: PathBuf::from("/abs/run-config.json"),
      config_sha256: HEX64.to_string(),
      artifact_dir: PathBuf::from("/abs/out"),
    })
  );
  assert_eq!(
    route(&args(&[
      "--child-config",
      "/abs/child-config.json",
      "--child-config-sha256",
      HEX64,
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ])),
    Ok(Form::ChildConfig {
      config: PathBuf::from("/abs/child-config.json"),
      config_sha256: HEX64.to_string(),
      artifact_dir: PathBuf::from("/abs/out"),
    })
  );
}

#[test]
fn router_rejects_everything_else() {
  let rejected: Vec<Vec<&str>> = vec![
    vec!["--bench"],
    vec!["--print-protocol-metadata"],
    vec!["--bench", "--print-protocol-metadata"],
    vec!["--print-protocol-metadata", "--bench", "--bench"],
    vec!["--print-protocol-metadata", "extra", "--bench"],
    vec!["--help", "--bench"],
    vec![
      "--run-config",
      "run-config.json",
      "--config-sha256",
      HEX64,
      "--attempt",
      "preflight",
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ],
    vec![
      "--run-config",
      "/abs/run-config.json",
      "--config-sha256",
      "ABCD",
      "--attempt",
      "preflight",
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ],
    vec![
      "--run-config",
      "/abs/run-config.json",
      "--config-sha256",
      HEX64,
      "--attempt",
      "child",
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ],
    vec![
      "--config-sha256",
      HEX64,
      "--run-config",
      "/abs/run-config.json",
      "--attempt",
      "preflight",
      "--artifact-dir",
      "/abs/out",
      "--bench",
    ],
    vec![
      "--child-config",
      "/abs/child-config.json",
      "--child-config-sha256",
      HEX64,
      "--artifact-dir",
      "--bench",
      "--bench",
    ],
    vec![
      "--child-config",
      "/abs/child-config.json",
      "--child-config-sha256",
      HEX64,
      "--artifact-dir",
      "out",
      "--bench",
    ],
  ];
  for tokens in rejected {
    let err = route(&args(&tokens)).expect_err(&format!("{tokens:?} must be rejected"));
    assert!(
      err.to_string().starts_with("poseidon_modp: usage error: "),
      "{err}"
    );
  }
}

#[test]
fn metadata_has_exactly_the_contract_keys_and_round_trips() {
  let md = protocol_metadata();
  let obj = md.as_object().expect("metadata is an object");
  let keys: Vec<&str> = obj.keys().map(String::as_str).collect();
  assert_eq!(keys, METADATA_KEYS, "sorted key set");
  let mut sorted = METADATA_KEYS.to_vec();
  sorted.sort_unstable();
  assert_eq!(sorted, METADATA_KEYS, "METADATA_KEYS is sorted");
  assert!(!obj.contains_key("checked_flag") && !obj.contains_key("unchecked_flag"));

  assert_eq!(md["transcript_domain_separator"], LIMBER_TRANSCRIPT_ID);
  assert_eq!(md["protocol_wire_id"], LIMBER_PROTOCOL_WIRE_ID);
  assert_eq!(md["prime_sampler_id"], LIMBER_PRIME_SAMPLER_ID);
  assert_eq!(
    md["prime_sampler_caps"],
    json!({ "t_max": 4096, "mr_rounds": 72, "base_draw_max": 160, "max_invocations": 4096 })
  );
  assert_eq!(md["common_lift_binding_id"], Value::Null);
  assert_eq!(md["benchmark_coins_id"], "poseidon-bench-chacha20-v1");
  assert_eq!(md["security_accounting_id"], ids::SECURITY_ACCOUNTING_ID);
  assert_eq!(md["timing_schema_id"], ids::TIMING_SCHEMA_ID);
  assert_eq!(md["tuning_protocol_id"], ids::TUNING_PROTOCOL_ID);
  assert_eq!(md["comparison_schema_id"], ids::COMPARISON_SCHEMA_ID);
  assert_eq!(md["kat_fixture_sha256"], ids::KAT_FIXTURE_SHA256);
  assert_eq!(md["tuning_corpus_id"], ids::TUNING_CORPUS_ID);
  assert_eq!(md["criterion_version"], "0.7.0");
  assert!(matches!(
    md["panic_strategy"].as_str(),
    Some("unwind" | "abort")
  ));
  assert!(md["debug_assertions"].is_boolean());
  assert!(md["overflow_checks"].is_boolean());
  assert_eq!(md["argv_forms_version"], 2);
  assert_eq!(md["variants"], json!(BACKEND_NAMES));
  assert_eq!(md["primary_groups"], json!(PRIMARY_GROUPS));
  assert_eq!(md["diagnostic_groups"], json!(DIAGNOSTIC_GROUPS));
  assert_eq!(md["primary_groups"], json!(["prove_e2e", "verify_core"]));
  assert_eq!(
    md["diagnostic_groups"],
    json!([
      "setup",
      "advice",
      "commit_witness",
      "prove_after_input_commit"
    ])
  );
  assert_eq!(
    md["tuning_epoch_id"],
    json!(tuned_defaults::TUNING_EPOCH_ID)
  );
  assert_eq!(
    md["tuning_group_set"],
    json!(tuned_defaults::TUNING_GROUP_SET)
  );
  let defaults = md["tuned_defaults"].as_array().expect("array");
  assert_eq!(defaults.len(), tuned_defaults::TUNED_DEFAULTS.len());
  for (entry, (backend, k, id)) in defaults.iter().zip(tuned_defaults::TUNED_DEFAULTS) {
    assert_eq!(
      entry,
      &json!({ "backend": backend, "k": k, "tuning_id": id })
    );
  }
  assert_eq!(md["check_mode"], "not_applicable");
  let dims = md["dimensions_by_k"].as_object().expect("object");
  assert_eq!(
    dims.keys().map(String::as_str).collect::<Vec<_>>(),
    ["10", "11", "12", "13", "7", "8", "9"],
    "decimal-string keys in BTreeMap (lexicographic) order"
  );
  for k in 7..=13 {
    assert_eq!(
      dims[&k.to_string()],
      json!({ "log_cons": 14, "log_vars": 14 })
    );
  }

  // Canonical JSON: one line, ASCII, sorted keys, byte-identical after a
  // parse/re-serialize cycle, and exactly what `--print-protocol-metadata`
  // prints.
  let text = canonical_json(&md);
  assert!(text.ends_with('\n') && text.matches('\n').count() == 1);
  assert!(text.is_ascii());
  assert!(!text.contains(": ") && !text.contains(", "));
  let parsed: Value = serde_json::from_str(&text).expect("parses");
  assert_eq!(parsed, md);
  assert_eq!(canonical_json(&parsed), text);
}

#[test]
fn canonical_json_matches_python_json_dumps() {
  let v =
    json!({ "b": [1, true, null], "a": "\u{e9}\n\"x\"\\", "c": { "z": 0, "y": "\u{1F600}" } });
  assert_eq!(
    canonical_json(&v),
    "{\"a\":\"\\u00e9\\n\\\"x\\\"\\\\\",\"b\":[1,true,null],\"c\":{\"y\":\"\\ud83d\\ude00\",\"z\":0}}\n"
  );
}

fn parse_ok(cfg: &Value, compiled: &Value) -> RunConfig {
  parse_run_config_value(cfg, compiled).unwrap_or_else(|e| panic!("must parse: {e}"))
}

fn parse_err(cfg: &Value, compiled: &Value, key_prefix: &str) {
  let err = parse_run_config_value(cfg, compiled).expect_err("must be rejected");
  assert!(
    err.key.starts_with(key_prefix),
    "expected an error under {key_prefix:?}, got {err}"
  );
}

#[test]
fn run_config_parser_accepts_the_runner_shape() {
  let compiled = protocol_metadata();
  for backend in BenchBackend::ALL {
    let cfg = synthetic_run_config(backend, &compiled);
    let parsed = parse_ok(&cfg, &compiled);
    assert_eq!(parsed.mode, Mode::Normal);
    assert_eq!(parsed.backend, backend);
    assert_eq!(parsed.k, Some(default_k(backend)));
    assert_eq!(parsed.hashes, CANONICAL_HASHES_PER_FIELD);
    assert_eq!(parsed.instance, Some(0));
    assert_eq!(parsed.threads, 1);
    assert_eq!(
      parsed.dimensions.map(|d| (d.log_cons, d.log_vars)),
      Some((14, 14))
    );
    assert_eq!(parsed.dimensions_by_candidate.len(), 7);
    assert!(parsed.sweep.is_none());
    assert_eq!(
      parsed.tuning_id.as_deref(),
      tuned_default(backend).map(|(_, id)| id)
    );
    assert_eq!(
      parsed.tuning_epoch_id.as_deref(),
      tuned_defaults::TUNING_EPOCH_ID
    );
    assert_eq!(
      parsed.comparison_key.as_deref(),
      Some("00".repeat(32).as_str())
    );
    assert!(parsed.features.is_empty());
    // psize is the same shape.
    let mut psize = cfg.clone();
    psize["mode"] = json!("psize");
    assert_eq!(parse_ok(&psize, &compiled).mode, Mode::Psize);
    // Unknown keys are ignored at every level.
    let mut extra = cfg.clone();
    extra["plan_revision"] = json!("v10");
    extra["workload"]["candidates"] = json!(K_ORDER);
    parse_ok(&extra, &compiled);
    // Sweep.
    let sweep = synthetic_sweep_config(backend, &compiled);
    let parsed = parse_ok(&sweep, &compiled);
    assert_eq!(parsed.mode, Mode::Sweep);
    assert_eq!(parsed.k, None);
    assert_eq!(parsed.instance, None);
    assert!(parsed.dimensions.is_none());
    let spec = parsed.sweep.expect("sweep spec");
    assert_eq!(spec.candidates, K_ORDER.to_vec());
    assert_eq!(spec.instances, (1..=9).collect::<Vec<u32>>());
    assert_eq!(spec.dimensions_by_candidate.len(), 7);
    assert!(parsed.tuning_id.is_none() && parsed.tuning_epoch_id.is_none());
    assert!(parsed.comparison_key.is_none());
  }
}

#[test]
fn run_config_parser_rejects_contract_violations() {
  let compiled = protocol_metadata();
  let backend = BenchBackend::Hyrax;
  let base = synthetic_run_config(backend, &compiled);
  for document in ["", "{}", "[]", "null"] {
    assert!(
      parse_run_config(document, &compiled).is_err(),
      "{document:?}"
    );
  }

  // Wrong backend / role.
  let mut c = base.clone();
  c["system_role"] = json!("zinc");
  parse_err(&c, &compiled, "system_role");
  let mut c = base.clone();
  c["backend"] = json!("brakedown");
  parse_err(&c, &compiled, "backend");
  let mut c = base.clone();
  c["workload"]["backend"] = json!("brakedown");
  parse_err(&c, &compiled, "workload.backend");

  // k out of 7..=13, k not the compiled default, workload.k disagreeing.
  for bad in [6, 14, 0] {
    let mut c = base.clone();
    c["k"] = json!(bad);
    c["workload"]["k"] = json!(bad);
    parse_err(&c, &compiled, "k");
  }
  let other = if default_k(backend) == 9 { 11 } else { 9 };
  let mut c = base.clone();
  c["k"] = json!(other);
  c["workload"]["k"] = json!(other);
  parse_err(&c, &compiled, "k");
  let mut c = base.clone();
  c["workload"]["k"] = json!(other);
  parse_err(&c, &compiled, "workload.k");
  let mut c = base.clone();
  c["dimensions"] = json!({ "log_cons": 13, "log_vars": 14 });
  parse_err(&c, &compiled, "dimensions");
  let mut c = base.clone();
  c["dimensions_by_candidate"]["9"] = json!({ "log_cons": 14, "log_vars": 15 });
  parse_err(&c, &compiled, "dimensions_by_candidate");
  let mut c = base.clone();
  c["dimensions_by_candidate"] = json!({ "6": { "log_cons": 14, "log_vars": 14 } });
  parse_err(&c, &compiled, "dimensions_by_candidate");
  let mut c = base.clone();
  c["dimensions_by_candidate"] = json!({ "7": { "log_cons": 14, "log_vars": 14 } });
  parse_err(&c, &compiled, "dimensions_by_candidate");

  // Workload pins.
  let mut c = base.clone();
  c["workload"]["hashes_per_field"] = json!(1);
  parse_err(&c, &compiled, "workload.hashes_per_field");
  let mut c = base.clone();
  c["workload"]["threads"] = json!(0);
  parse_err(&c, &compiled, "workload.threads");
  let mut c = base.clone();
  c["workload"]["instance"] = json!(null);
  parse_err(&c, &compiled, "workload.instance");

  // Tuning mismatch: an unexpected tuning id or epoch.
  let mut c = base.clone();
  c["tuning"]["tuning_id"] = json!(HEX64);
  if tuned_default(backend).map(|(_, id)| id) != Some(HEX64) {
    parse_err(&c, &compiled, "tuning.tuning_id");
  }
  let mut c = base.clone();
  c["tuning"]["tuning_epoch_id"] = json!(HEX64);
  if tuned_defaults::TUNING_EPOCH_ID != Some(HEX64) {
    parse_err(&c, &compiled, "tuning.tuning_epoch_id");
  }
  let mut c = base.clone();
  c["tuning"]["tuning_epoch_id"] = json!("not hex");
  parse_err(&c, &compiled, "tuning.tuning_epoch_id");

  // ids mismatch: any difference from the compiled metadata.
  let mut c = base.clone();
  c["ids"]["compiled"]["criterion_version"] = json!("0.5.1");
  parse_err(&c, &compiled, "ids.compiled");
  let mut c = base.clone();
  c["ids"]["compiled"]["extra"] = json!(1);
  parse_err(&c, &compiled, "ids.compiled");
  let mut c = base.clone();
  c["ids"]["benchmark_coins_policy"]["id"] = json!("poseidon-bench-chacha20-v0");
  parse_err(&c, &compiled, "ids.benchmark_coins_policy.id");
  let mut c = base.clone();
  c["ids"]["benchmark_coins_policy"]["framing"] = json!("other");
  parse_err(&c, &compiled, "ids.benchmark_coins_policy.framing");

  // Pinned literals.
  let mut c = base.clone();
  c["check_modes"]["timed"] = json!("unchecked");
  parse_err(&c, &compiled, "check_modes.timed");
  let mut c = base.clone();
  c["criterion"]["sample_size"] = json!(100);
  parse_err(&c, &compiled, "criterion.sample_size");
  let mut c = base.clone();
  c["features"] = json!(["parallel"]);
  parse_err(&c, &compiled, "features");
  let mut c = base.clone();
  c["rustflags"] = json!("");
  parse_err(&c, &compiled, "rustflags");
  let mut c = base.clone();
  c["comparison_key"] = json!(null);
  parse_err(&c, &compiled, "comparison_key");
  let mut c = base.clone();
  c["preflight_requirements"]["rayon_threads_asserted"] = json!(2);
  parse_err(
    &c,
    &compiled,
    "preflight_requirements.rayon_threads_asserted",
  );
  let mut c = base.clone();
  c["preflight_requirements"]["double_construction"] = json!(false);
  parse_err(&c, &compiled, "preflight_requirements.double_construction");
  let mut c = base.clone();
  c.as_object_mut().unwrap().remove("sweep");
  parse_err(&c, &compiled, "sweep");

  // Sweep: literal candidate order, nonempty distinct instances without 0,
  // null k/instance/tuning/comparison key.
  let sweep = synthetic_sweep_config(backend, &compiled);
  let mut c = sweep.clone();
  c["sweep"]["candidates"] = json!([7, 8, 9, 10, 11, 12, 13]);
  parse_err(&c, &compiled, "sweep.candidates");
  let mut c = sweep.clone();
  c["sweep"]["instances"] = json!([0, 1]);
  parse_err(&c, &compiled, "sweep.instances");
  let mut c = sweep.clone();
  c["sweep"]["instances"] = json!([1, 1]);
  parse_err(&c, &compiled, "sweep.instances");
  let mut c = sweep.clone();
  c["sweep"]["instances"] = json!([]);
  parse_err(&c, &compiled, "sweep.instances");
  let mut c = sweep.clone();
  c["k"] = json!(9);
  parse_err(&c, &compiled, "k");
  let mut c = sweep.clone();
  c["tuning"]["tuning_epoch_id"] = json!(HEX64);
  parse_err(&c, &compiled, "tuning.tuning_epoch_id");
  let mut c = sweep.clone();
  c["comparison_key"] = json!(HEX64);
  parse_err(&c, &compiled, "comparison_key");
}

#[test]
fn child_config_parser_binds_the_parent_and_coordinate() {
  let compiled = protocol_metadata();
  let backend = BenchBackend::Brakedown;
  let parent = synthetic_run_config(backend, &compiled);
  let primary = json!({ "kind": "primary", "metric": "verify_core", "block": 5 });
  let cc = synthetic_child_config(&parent, primary.clone());
  let parsed = parse_child_config_value(&cc, &compiled).expect("primary child parses");
  assert_eq!(
    parsed.coordinate,
    Coordinate::Primary {
      metric: "verify_core",
      block: 5
    }
  );
  assert_eq!(parsed.parent.backend, backend);
  assert_eq!(parsed.order, Some(args(&["zinc", "hyrax", "brakedown"])));
  assert_eq!(parsed.ordinal, 1);
  assert!(parsed.session_id.is_none());
  let diag = synthetic_child_config(
    &parent,
    json!({ "kind": "diagnostic", "metric": null, "block": "diag" }),
  );
  assert_eq!(
    parse_child_config_value(&diag, &compiled)
      .unwrap()
      .coordinate,
    Coordinate::Diagnostic
  );

  // Embedded parent digest mismatch: any change to the parent, or a stale
  // digest, is rejected under parent_config.
  let mut stale = cc.clone();
  stale["parent_config"]["plan_revision"] = json!("v10");
  let err = parse_child_config_value(&stale, &compiled).expect_err("digest mismatch");
  assert_eq!(err.key, "parent_config");
  let mut stale = cc.clone();
  stale["parent_config_sha256"] = json!(HEX64);
  let err = parse_child_config_value(&stale, &compiled).expect_err("digest mismatch");
  assert_eq!(err.key, "parent_config");
  // The embedded parent is validated like a run config.
  let mut bad_parent = parent.clone();
  bad_parent["rustflags"] = json!("");
  let cc_bad = synthetic_child_config(&bad_parent, primary.clone());
  let err = parse_child_config_value(&cc_bad, &compiled).expect_err("bad parent");
  assert_eq!(err.key, "parent_config.rustflags");
  // Exactly the eight keys; the role must be the parent's backend; the
  // order is a permutation of the three systems; the ordinal is 1-based.
  let mut extra = cc.clone();
  extra["extra"] = json!(1);
  assert!(parse_child_config_value(&extra, &compiled).is_err());
  let mut fewer = cc.clone();
  fewer.as_object_mut().unwrap().remove("session_id");
  assert!(parse_child_config_value(&fewer, &compiled).is_err());
  let mut role = cc.clone();
  role["system_role"] = json!("hyrax");
  assert_eq!(
    parse_child_config_value(&role, &compiled).unwrap_err().key,
    "system_role"
  );
  let mut order = cc.clone();
  order["order"] = json!(["zinc", "hyrax"]);
  assert_eq!(
    parse_child_config_value(&order, &compiled).unwrap_err().key,
    "order"
  );
  let mut ordinal = cc.clone();
  ordinal["ordinal"] = json!(0);
  assert_eq!(
    parse_child_config_value(&ordinal, &compiled)
      .unwrap_err()
      .key,
    "ordinal"
  );
  let mut metric = cc.clone();
  metric["coordinate"]["metric"] = json!("setup");
  assert_eq!(
    parse_child_config_value(&metric, &compiled)
      .unwrap_err()
      .key,
    "coordinate.metric"
  );
  // A sweep coordinate needs a sweep parent and vice versa.
  let sweep_coord =
    json!({ "kind": "sweep", "block": 1, "ordinal": 7, "candidate": 9, "instance": 4 });
  let cc_sweep_on_normal = synthetic_child_config(&parent, sweep_coord.clone());
  assert_eq!(
    parse_child_config_value(&cc_sweep_on_normal, &compiled)
      .unwrap_err()
      .key,
    "coordinate.kind"
  );
  let sweep_parent = synthetic_sweep_config(backend, &compiled);
  let cc_sweep = synthetic_child_config(&sweep_parent, sweep_coord);
  let parsed = parse_child_config_value(&cc_sweep, &compiled).expect("sweep child parses");
  assert_eq!(
    parsed.coordinate,
    Coordinate::Sweep {
      block: 1,
      ordinal: 7,
      candidate: 9,
      instance: 4
    }
  );
  assert!(parsed.order.is_none());
  let cc_primary_on_sweep = synthetic_child_config(&sweep_parent, primary);
  assert_eq!(
    parse_child_config_value(&cc_primary_on_sweep, &compiled)
      .unwrap_err()
      .key,
    "coordinate.kind"
  );
  let mut cc_bad_inst = synthetic_child_config(
    &sweep_parent,
    json!({ "kind": "sweep", "block": 0, "ordinal": 1, "candidate": 9, "instance": 0 }),
  );
  assert_eq!(
    parse_child_config_value(&cc_bad_inst, &compiled)
      .unwrap_err()
      .key,
    "coordinate.instance"
  );
  cc_bad_inst["coordinate"]["instance"] = json!(1);
  cc_bad_inst["coordinate"]["candidate"] = json!(6);
  assert_eq!(
    parse_child_config_value(&cc_bad_inst, &compiled)
      .unwrap_err()
      .key,
    "coordinate.candidate"
  );
}

#[test]
fn function_id_grammar() {
  let re = Regex::new(FUNCTION_ID_REGEX).expect("the contract regex compiles");
  let dims = compiled_dimensions()[&9];
  let primary = function_id(
    BenchBackend::Hyrax,
    10,
    &dims,
    9,
    0,
    1,
    &coordinate_token(Some(3)),
  );
  assert_eq!(
    primary,
    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/primary/blk3"
  );
  let caps = re.captures(&primary).expect("primary id matches");
  assert_eq!(&caps["backend"], "hyrax");
  assert_eq!(&caps["hashes"], "10");
  assert_eq!(&caps["total"], "30");
  assert_eq!(&caps["log_cons"], "14");
  assert_eq!(&caps["log_vars"], "14");
  assert_eq!(&caps["k"], "9");
  assert_eq!(&caps["instance"], "0");
  assert_eq!(&caps["threads"], "1");
  assert_eq!(&caps["block"], "3");
  let diag = function_id(
    BenchBackend::Brakedown,
    10,
    &dims,
    13,
    7,
    1,
    &coordinate_token(None),
  );
  assert_eq!(
    diag,
    "brakedown/mixed3/Hpf10-total30/c2^14v2^14/k13/inst7/thr1/diagnostic/blkdiag"
  );
  let caps = re.captures(&diag).expect("diagnostic id matches");
  assert_eq!(&caps["backend"], "brakedown");
  assert!(caps.name("block").is_none());
  for bad in [
    "zincplus/qz/mixed3/Hpf10-total30/nvars14/deg5/wcols3/fl4/flat/inst0/thr1/chk0/primary/blk0",
    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/primary/blkdiag",
    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/diagnostic/blk0",
    "hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/primary/blk0/",
    "Hyrax/mixed3/Hpf10-total30/c2^14v2^14/k9/inst0/thr1/primary/blk0",
  ] {
    assert!(!re.is_match(bad), "{bad}");
  }
}

#[test]
fn benchmark_coins_seed_matches_the_independent_blake3() {
  // Reference values from an independent pure-Python BLAKE3 (single-chunk
  // compression per the BLAKE3 specification, self-checked against the
  // published vectors for "" and "abc") over the 44-byte framing
  // `"limber-poseidon2-v1/bench-coins/v1\0" || backend_u8 || index_le32 || instance_le32`.
  assert_eq!(BENCHMARK_COINS_DOMAIN.len(), 35);
  assert_eq!(
    BENCHMARK_COINS_DOMAIN,
    b"limber-poseidon2-v1/bench-coins/v1\0"
  );
  assert_eq!(
    hex(&benchmark_coins_seed(BenchBackend::Hyrax, 0, 0)),
    "678d1e683bbe802b54818a1f4be281632471308b097e648acc623b67e70893de"
  );
  assert_eq!(
    hex(&benchmark_coins_seed(BenchBackend::Brakedown, 3, 9)),
    "7ac3b4a40244d2dbd12184ae5b86e407e759c47a05cf45cfb86825ac9b7fffc4"
  );
  // The bench derives the index from k's position in K_ORDER (the single
  // pinned candidate k = 9 has index 0).
  assert_eq!(K_ORDER, [9]);
  assert_eq!(
    benchmark_seed(BenchBackend::Hyrax, 9, 0),
    benchmark_coins_seed(BenchBackend::Hyrax, 0, 0)
  );
  assert_eq!(
    benchmark_seed(BenchBackend::Brakedown, 9, 9),
    benchmark_coins_seed(BenchBackend::Brakedown, 0, 9)
  );
  // ChaCha20Rng::from_seed is reproducible from the seed.
  let mut a = benchmark_rng(BenchBackend::Hyrax, 9, 0);
  let mut b = ChaCha20Rng::from_seed(benchmark_coins_seed(BenchBackend::Hyrax, 0, 0));
  let mut c = benchmark_rng(BenchBackend::Hyrax, 9, 1);
  let (x, y, z) = (a.next_u64(), b.next_u64(), c.next_u64());
  assert_eq!(x, y);
  assert_ne!(x, z);
}

#[test]
fn hashed_file_and_artifact_dir_validators() {
  let root = std::env::temp_dir().join(format!(
    "limber-poseidon-contract-{}-{}",
    std::process::id(),
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .unwrap()
      .as_nanos()
  ));
  std::fs::create_dir_all(&root).unwrap();
  let root = root.canonicalize().unwrap();
  let file = root.join("run-config.json");
  std::fs::write(&file, b"{}\n").unwrap();
  let digest = sha256_hex(b"{}\n");
  assert_eq!(read_hashed_file(&file, &digest).unwrap(), b"{}\n");
  assert!(read_hashed_file(&file, HEX64).is_err());
  assert!(read_hashed_file(Path::new("run-config.json"), &digest).is_err());
  assert!(read_hashed_file(&root, &digest).is_err());

  let empty = root.join("out");
  std::fs::create_dir(&empty).unwrap();
  assert_eq!(validate_artifact_dir(&empty), Ok(()));
  assert!(validate_artifact_dir(Path::new("out")).is_err());
  assert!(validate_artifact_dir(&root).is_err(), "not empty");
  assert!(validate_artifact_dir(&root.join("missing")).is_err());
  assert!(validate_artifact_dir(&root.join("out/../out")).is_err());
  #[cfg(unix)]
  {
    let link = root.join("link");
    std::os::unix::fs::symlink(&empty, &link).unwrap();
    assert!(validate_artifact_dir(&link).is_err(), "symlink component");
    assert!(read_hashed_file(&root.join("filelink"), &digest).is_err());
    std::os::unix::fs::symlink(&file, root.join("filelink")).unwrap();
    assert!(
      read_hashed_file(&root.join("filelink"), &digest).is_err(),
      "symlinked config"
    );
  }
  std::fs::remove_dir_all(&root).unwrap();
}
