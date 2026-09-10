//! Pins the shared cross-system Poseidon2 benchmark specification files to
//! the identifiers compiled into `limber::poseidon2::ids` (Zinc plan v10
//! §9; limber contract §1): the exact-byte SHA-256 of every spec file, the
//! KAT fixture and the tuning corpus; the canonical JSON form of the four
//! spec files (sorted keys, compact, ASCII, one trailing LF); the
//! mandatory `sampler_union` event of limber's security-accounting file
//! under both backend variants with its exact bound; and the agreement of
//! the tuning protocol's limber section with the compiled constants.

use limber::poseidon2::ids;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A crate-relative path.
fn crate_path(rel: &str) -> PathBuf {
  Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// The exact bytes of a crate-relative file.
fn read(rel: &str) -> Vec<u8> {
  std::fs::read(crate_path(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// Lowercase hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
  Sha256::digest(bytes)
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect()
}

/// Canonical serialization: sorted keys (via `BTreeMap`, independent of
/// serde_json's map type), compact separators, no trailing newline. String
/// escaping follows serde_json, which agrees with Python's
/// `json.dumps(ensure_ascii=True)` on ASCII content.
fn write_canonical(v: &Value, out: &mut String) {
  match v {
    Value::Null => out.push_str("null"),
    Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    Value::Number(n) => out.push_str(&n.to_string()),
    Value::String(s) => out.push_str(&serde_json::to_string(s).expect("string serializes")),
    Value::Array(items) => {
      out.push('[');
      for (i, item) in items.iter().enumerate() {
        if i > 0 {
          out.push(',');
        }
        write_canonical(item, out);
      }
      out.push(']');
    }
    Value::Object(map) => {
      let sorted: BTreeMap<&String, &Value> = map.iter().collect();
      out.push('{');
      for (i, (k, item)) in sorted.iter().enumerate() {
        if i > 0 {
          out.push(',');
        }
        out.push_str(&serde_json::to_string(k).expect("key serializes"));
        out.push(':');
        write_canonical(item, out);
      }
      out.push('}');
    }
  }
}

/// Parse `bytes` and require canonical form: ASCII, exactly one LF (the
/// trailing one), and byte-identical re-serialization with sorted keys.
fn parse_canonical(rel: &str, bytes: &[u8]) -> Value {
  assert!(bytes.is_ascii(), "{rel}: not ASCII");
  assert!(bytes.ends_with(b"\n"), "{rel}: no trailing LF");
  assert_eq!(
    bytes.iter().filter(|b| **b == b'\n').count(),
    1,
    "{rel}: more than one line"
  );
  let v: Value = serde_json::from_slice(bytes).unwrap_or_else(|e| panic!("{rel}: {e}"));
  let mut canonical = String::new();
  write_canonical(&v, &mut canonical);
  canonical.push('\n');
  assert_eq!(canonical.as_bytes(), bytes, "{rel}: not canonical JSON");
  v
}

/// Every JSON number token anywhere in `v` (the security file forbids them).
fn count_numbers(v: &Value) -> usize {
  match v {
    Value::Number(_) => 1,
    Value::Array(items) => items.iter().map(count_numbers).sum(),
    Value::Object(map) => map.values().map(count_numbers).sum(),
    _ => 0,
  }
}

#[test]
fn spec_file_digests_are_pinned() {
  let pins = [
    (ids::TIMING_SCHEMA_PATH, ids::TIMING_SCHEMA_ID),
    (ids::TUNING_PROTOCOL_PATH, ids::TUNING_PROTOCOL_ID),
    (ids::COMPARISON_SCHEMA_PATH, ids::COMPARISON_SCHEMA_ID),
    (ids::SECURITY_ACCOUNTING_PATH, ids::SECURITY_ACCOUNTING_ID),
    (ids::KAT_FIXTURE_PATH, ids::KAT_FIXTURE_SHA256),
    (ids::TUNE_CORPUS_PATH, ids::TUNING_CORPUS_ID),
  ];
  for (rel, id) in pins {
    assert!(rel.starts_with(ids::SPEC_DIR) || rel.starts_with("tests/data/"));
    assert_eq!(id.len(), 64, "{rel}: id is 64 hex digits");
    assert!(id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')));
    assert_eq!(sha256_hex(&read(rel)), id, "{rel}: SHA-256");
  }
}

#[test]
fn spec_files_are_canonical_json() {
  for rel in [
    ids::TIMING_SCHEMA_PATH,
    ids::TUNING_PROTOCOL_PATH,
    ids::COMPARISON_SCHEMA_PATH,
    ids::SECURITY_ACCOUNTING_PATH,
  ] {
    parse_canonical(rel, &read(rel));
  }
}

#[test]
fn security_file_has_the_exact_sampler_union_under_both_variants() {
  let rel = ids::SECURITY_ACCOUNTING_PATH;
  let doc = parse_canonical(rel, &read(rel));
  assert_eq!(doc["schema"], "limber/security-accounting/v1");
  assert_eq!(doc["quantified_bound_available"], false);
  assert_eq!(
    count_numbers(&doc),
    0,
    "the security file must not contain JSON number tokens"
  );
  let symbols = doc["symbols"].as_object().expect("symbols table");
  for (name, value) in [
    ("T_MAX", "4096"),
    ("MR_ROUNDS", "72"),
    ("BASE_DRAW_MAX", "160"),
  ] {
    assert_eq!(symbols[name]["kind"], "constant", "{name}");
    assert_eq!(symbols[name]["value"], json!({ "int": value }), "{name}");
  }
  assert_eq!(symbols["s"]["kind"], "run_parameter");
  assert_eq!(symbols["s"]["source"], "preflight.prime_audit.invocations");
  assert_eq!(symbols["Q"]["kind"], "publisher_input");
  let transcript = &doc["transcript"];
  assert_eq!(transcript["hash"], "Keccak-256");
  assert_eq!(
    transcript["domain_separator"],
    "limber/transcript/keccak256-p0d"
  );
  assert_eq!(
    transcript["prime_sampler_id"],
    "limber-prime-v1-msb1-bpsw21-mr72-c4096-d160-i4096"
  );

  let variants = doc["variants"].as_object().expect("variants");
  let names: Vec<&str> = variants.keys().map(String::as_str).collect();
  assert_eq!(names, ["brakedown", "hyrax"], "variant keys (sorted)");
  let expected_bound = json!({
    "mul": [
      { "symbol": "s" },
      { "symbol": "T_MAX" },
      { "pow": [{ "int": "4" }, { "int": "-72" }] }
    ]
  });
  for (variant, spec) in variants {
    let events = spec["events"].as_array().expect("events list");
    assert!(!events.is_empty(), "{variant}: events");
    let mut ids_seen = Vec::new();
    for event in events {
      for key in [
        "id",
        "description",
        "bound",
        "theorem",
        "challenge_count",
        "degree",
      ] {
        assert!(!event[key].is_null(), "{variant}: event lacks {key}");
      }
      let id = event["id"].as_str().expect("id");
      assert!(!ids_seen.contains(&id), "{variant}: duplicate event {id}");
      ids_seen.push(id);
      if event["bound"] != "unquantified" {
        assert_eq!(
          id, "sampler_union",
          "{variant}: only sampler_union is quantified"
        );
      }
    }
    let sampler_union = events
      .iter()
      .find(|e| e["id"] == "sampler_union")
      .unwrap_or_else(|| panic!("{variant}: sampler_union event"));
    assert_eq!(sampler_union["bound"], expected_bound, "{variant}: bound");
    assert_eq!(
      sampler_union["challenge_count"],
      json!({ "symbol": "s" }),
      "{variant}: challenge_count"
    );
    assert_eq!(sampler_union["degree"], "0", "{variant}: degree");
    for id in [
      "sumcheck_soundness",
      "inteval_fingerprint",
      "pcs_binding",
      "fiat_shamir",
    ] {
      assert!(ids_seen.contains(&id), "{variant}: missing event {id}");
    }
  }
}

#[test]
fn tuning_protocol_agrees_with_compiled_identifiers() {
  let rel = ids::TUNING_PROTOCOL_PATH;
  let doc = parse_canonical(rel, &read(rel));
  assert_eq!(doc["schema"], "poseidon-tuning-protocol/v2");
  assert_eq!(doc["kat_fixture_sha256"], ids::KAT_FIXTURE_SHA256);
  assert_eq!(doc["tuning_corpus"]["sha256"], ids::TUNING_CORPUS_ID);
  assert_eq!(doc["held_out_instance"], 0);
  assert_eq!(doc["instances"], json!([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  let limber = &doc["systems"]["limber"];
  assert_eq!(limber["benchmark_coins_id"], ids::BENCHMARK_COINS_ID);
  assert_eq!(
    limber["benchmark_coins"]["framing"],
    ids::BENCHMARK_COINS_FRAMING
  );
  let prefix_hex: String = ids::BENCHMARK_COINS_DOMAIN
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
  assert_eq!(
    limber["benchmark_coins"]["framing_bytes_hex_prefix"],
    prefix_hex
  );
  assert!(ids::BENCHMARK_COINS_FRAMING.starts_with(&format!(
    "\"{}\\0\"",
    std::str::from_utf8(&ids::BENCHMARK_COINS_DOMAIN[..ids::BENCHMARK_COINS_DOMAIN.len() - 1])
      .expect("ASCII domain")
  )));
  let k_order: Vec<usize> = limber["k_order"]
    .as_array()
    .expect("k_order")
    .iter()
    .map(|v| usize::try_from(v.as_u64().expect("k")).expect("k fits"))
    .collect();
  assert_eq!(k_order, ids::K_ORDER.to_vec());
  assert_eq!(limber["k_range"], json!([7, 13]));
  assert!(ids::K_ORDER.iter().all(|k| (7..=13).contains(k)));
  let timing = parse_canonical(ids::TIMING_SCHEMA_PATH, &read(ids::TIMING_SCHEMA_PATH));
  assert_eq!(timing["schema"], doc["timing_schema"]);
}
