#!/usr/bin/env bash
# Canonical runner for the classic-Spartan Poseidon2 baseline benchmark
# (plan/poseidon_spartan_bench.md §6).
#
# Same two-phase protocol as scripts/run_poseidon_bench.sh: stage an
# immutable run-config.json (emitted by the crate-side
# poseidon_bench_config helper with --suite spartan — this script never
# interprets benchmark flags itself), hash it, move the staging directory
# to its run id, write manifest.running.json, launch the benchmark with
# the config handshake, and on success finalize manifest.json with
# artifact hashes. On failure manifest.failed.json is retained.
#
# No cache-audit sidecar: this suite has no Brakedown retained cache.
#
# Ordinary run output stays under target/poseidon-bench/ (ignored);
# publication into bench-results/ is a separate explicit step
# (scripts/publish_poseidon_bench.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BENCH_ROOT="target/poseidon-bench"
RUN_DIR=""
STATUS_WRITTEN=0

finalize_failure() {
  local code=$?
  if [[ $code -ne 0 && -n "$RUN_DIR" && -d "$RUN_DIR" && $STATUS_WRITTEN -eq 0 ]]; then
    python3 - "$RUN_DIR" "$code" <<'EOF'
import json, sys, time
run_dir, code = sys.argv[1], int(sys.argv[2])
manifest = {
    "status": "failed",
    "exit_code": code,
    "publishable": False,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(f"{run_dir}/manifest.failed.json", "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
EOF
    echo "run_poseidon_spartan_bench: FAILED (exit $code); manifest.failed.json written in $RUN_DIR" >&2
  fi
}
trap finalize_failure EXIT

echo "== building the config helper, preflight child, and benchmark =="
cargo build --release --bin poseidon_bench_config --bin poseidon_spartan_preflight
cargo bench --bench poseidon_spartan --no-run
HELPER="target/release/poseidon_bench_config"
PREFLIGHT="target/release/poseidon_spartan_preflight"

# Phase 1: immutable config in a staging directory. The helper — not
# shell — enforces the default clean-tree gate via the shared parsed
# POSEIDON_ALLOW_DIRTY value.
mkdir -p "$BENCH_ROOT"
STAGING="$(mktemp -d "$BENCH_ROOT/staging.XXXXXX")"
"$HELPER" --suite spartan > "$STAGING/run-config.json"

# Phase 2: hash the config's exact bytes, pick the run id, move into place.
CONFIG_SHA="$(shasum -a 256 "$STAGING/run-config.json" | cut -d' ' -f1)"
echo "$CONFIG_SHA" > "$STAGING/run-config.sha256"
CONFIG12="${CONFIG_SHA:0:12}"
START_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${START_UTC}-spartan-cfg-${CONFIG12}"
RUN_DIR="$BENCH_ROOT/$RUN_ID"
mv "$STAGING" "$RUN_DIR"
echo "== run id: $RUN_ID =="

# Phase 3: running manifest, immutability re-check, then launch with the
# handshake. The benchmark re-parses the environment, recomputes the file
# hash, and byte-compares its canonical protocol subsection.
python3 - "$RUN_DIR" "$CONFIG_SHA" "$START_UTC" <<'EOF'
import json, sys
run_dir, sha, start = sys.argv[1:4]
with open(f"{run_dir}/manifest.running.json", "w") as f:
    json.dump({"status": "running", "config_sha256": sha, "started_utc": start},
              f, indent=2, sort_keys=True)
    f.write("\n")
EOF
"$HELPER" --suite spartan --check "$RUN_DIR/run-config.json"

# Stage 3.5: the dedicated preflight child (plan §4). It re-synthesizes
# the exact shape, derives the execution role, proves once for measuring
# roles, writes proposal.json / preflight.json, and exits before Criterion
# so all preflight allocations are destroyed. Exit 20/21 are the successful
# nonpublishable proposal/checkpoint terminal states.
set +e
POSEIDON_RUN_DIR="$REPO_ROOT/$RUN_DIR" \
POSEIDON_CONFIG_PATH="$REPO_ROOT/$RUN_DIR/run-config.json" \
POSEIDON_CONFIG_SHA256="$CONFIG_SHA" \
  "$PREFLIGHT" 2>&1 | tee "$RUN_DIR/preflight.log"
PREFLIGHT_CODE=${PIPESTATUS[0]}
set -e
if [[ $PREFLIGHT_CODE -eq 20 || $PREFLIGHT_CODE -eq 21 ]]; then
  ROLE=$([[ $PREFLIGHT_CODE -eq 20 ]] && echo resource_proposal || echo resource_checkpoint)
  echo "== preflight terminal role: $ROLE =="
  python3 - "$RUN_DIR" "$CONFIG_SHA" "$START_UTC" "$ROLE" <<'PYEOF'
import hashlib, json, os, sys, time
run_dir, sha, start, role = sys.argv[1:5]
if role == "resource_proposal":
    assert os.path.exists(f"{run_dir}/proposal.json"), "proposal role requires proposal.json"
    assert not os.path.exists(f"{run_dir}/preflight.json"), "proposal role forbids preflight.json"
else:
    assert os.path.exists(f"{run_dir}/proposal.json")
    assert os.path.exists(f"{run_dir}/preflight.json"), "checkpoint role requires preflight.json"
assert not os.listdir(f"{run_dir}/criterion") if os.path.isdir(f"{run_dir}/criterion") else True
artifacts = {}
for root, _dirs, files in os.walk(run_dir):
    for name in files:
        path = os.path.join(root, name)
        rel = os.path.relpath(path, run_dir)
        if rel.startswith("manifest"):
            continue
        with open(path, "rb") as f:
            artifacts[rel] = hashlib.sha256(f.read()).hexdigest()
manifest = {
    "status": "complete",
    "execution_role": role,
    "run_id": os.path.basename(run_dir),
    "config_sha256": sha,
    "publishable": False,
    "canonical_comparison": False,
    "started_utc": start,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "exit_code": 0,
    "artifacts": artifacts,
}
tmp = f"{run_dir}/manifest.json.tmp"
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, f"{run_dir}/manifest.json")
os.remove(f"{run_dir}/manifest.running.json")
print(f"terminal {role} manifest written")
PYEOF
  STATUS_WRITTEN=1
  echo "== complete (terminal $ROLE): $RUN_DIR =="
  exit 0
elif [[ $PREFLIGHT_CODE -ne 0 ]]; then
  echo "run_poseidon_spartan_bench: preflight failed with $PREFLIGHT_CODE" >&2
  exit "$PREFLIGHT_CODE"
fi

# Validate the run-bound preflight sidecar before measuring.
python3 - "$RUN_DIR" "$CONFIG_SHA" <<'PYEOF'
import json, subprocess, sys
run_dir, sha = sys.argv[1:3]
with open(f"{run_dir}/preflight.json") as f:
    pf = json.load(f)
assert pf["resolved_config_sha256"] == sha, "preflight.json is not bound to this run's config"
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                      check=True).stdout.strip()
assert pf["git_sha"] == head, "preflight.json git commit differs from the working tree"
assert pf["pass"] is True, "preflight recorded a failing decision"
with open(f"{run_dir}/run-config.json") as f:
    cfg = json.load(f)
proto = cfg["protocol"]
assert pf["workload_digest"] == proto["workload_digest"]
assert pf["encoding_digest"] == proto["encoding_digest"]
assert pf["shape"] == proto["dims"] or {
    "real_cons": pf["shape"]["real_cons"],
    "real_vars": pf["shape"]["real_vars"],
    "padded_cons": pf["shape"]["padded_cons"],
    "padded_vars": pf["shape"]["padded_vars"],
} == proto["dims"], "preflight shape differs from the resolved config"
print("preflight sidecar validation OK")
PYEOF

mkdir -p "$RUN_DIR/criterion"
set +e
POSEIDON_RUN_DIR="$REPO_ROOT/$RUN_DIR" \
POSEIDON_CONFIG_PATH="$REPO_ROOT/$RUN_DIR/run-config.json" \
POSEIDON_CONFIG_SHA256="$CONFIG_SHA" \
  cargo bench --bench poseidon_spartan 2>&1 | tee "$RUN_DIR/bench.log"
BENCH_CODE=${PIPESTATUS[0]}
set -e
if [[ $BENCH_CODE -ne 0 ]]; then
  echo "run_poseidon_spartan_bench: benchmark exited with $BENCH_CODE" >&2
  exit "$BENCH_CODE"
fi

# Phase 4: post-run validation. Re-check the immutable config to catch
# source/toolchain changes during the measurement, validate the
# mode-specific sidecars, hash every artifact, and atomically write the
# complete manifest.
"$HELPER" --suite spartan --check "$RUN_DIR/run-config.json"

python3 - "$RUN_DIR" <<'EOF'
import json, os, sys
run_dir = sys.argv[1]
with open(f"{run_dir}/run-config.json") as f:
    cfg = json.load(f)
proto = cfg["protocol"]
if proto.get("suite") != "spartan":
    raise SystemExit("run-config protocol is not the spartan suite")
mode = proto["mode"]
if mode == "proof_size":
    with open(f"{run_dir}/proof-size.json") as f:
        ps = json.load(f)
    for key in ("instance_commitments", "protocol_challenges",
                "sumchecks_and_claims", "evaluation_opening",
                "comparison_payload", "public_values", "wire_total"):
        if not isinstance(ps["combined"].get(key), int):
            raise SystemExit(f"proof-size.json combined block is missing {key}")
else:
    groups = [d for d in os.listdir(f"{run_dir}/criterion")
              if os.path.isdir(f"{run_dir}/criterion/{d}")]
    observed = {}
    for want in ("setup", "prove_e2e", "verify"):
        if want not in groups:
            raise SystemExit(f"criterion output has no {want} group")
        # Observed Criterion contract (plan §6): Flat mode, exactly ten
        # samples, equal iteration counts across samples.
        sample_paths = []
        for root, _dirs, files in os.walk(f"{run_dir}/criterion/{want}"):
            if "sample.json" in files and root.endswith("new"):
                sample_paths.append(os.path.join(root, "sample.json"))
        if len(sample_paths) != 1:
            raise SystemExit(f"{want}: expected one new/sample.json, got {len(sample_paths)}")
        with open(sample_paths[0]) as f:
            sample = json.load(f)
        mode = sample.get("sampling_mode")
        iters = sample.get("iters", [])
        if mode != "Flat":
            raise SystemExit(f"{want}: observed sampling mode {mode!r}, required Flat")
        if len(iters) != 10:
            raise SystemExit(f"{want}: observed {len(iters)} samples, required 10")
        if len(set(iters)) != 1:
            raise SystemExit(f"{want}: unequal iteration counts {iters}")
        observed[want] = {"sampling_mode": mode, "samples": len(iters),
                          "iters_per_sample": iters[0]}
    with open(f"{run_dir}/criterion-observed.json", "w") as f:
        json.dump(observed, f, indent=2, sort_keys=True)
        f.write("\n")
print("post-run sidecar validation OK")
EOF

python3 - "$RUN_DIR" "$CONFIG_SHA" "$START_UTC" <<'EOF'
import hashlib, json, os, sys, time
run_dir, sha, start = sys.argv[1:4]
with open(f"{run_dir}/run-config.json") as f:
    cfg = json.load(f)
artifacts = {}
for root, _dirs, files in os.walk(run_dir):
    for name in files:
        path = os.path.join(root, name)
        rel = os.path.relpath(path, run_dir)
        if rel.startswith("manifest"):
            continue
        with open(path, "rb") as f:
            artifacts[rel] = hashlib.sha256(f.read()).hexdigest()
manifest = {
    "status": "complete",
    "run_id": os.path.basename(run_dir),
    "config_sha256": sha,
    "publishable": cfg["environment"]["publishable"]
        and cfg["protocol"]["canonical_env_common"],
    "canonical_comparison": cfg["environment"]["publishable"]
        and cfg["protocol"]["canonical_env_common"],
    "started_utc": start,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "exit_code": 0,
    "artifacts": artifacts,
}
tmp = f"{run_dir}/manifest.json.tmp"
with open(tmp, "w") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(tmp, f"{run_dir}/manifest.json")
os.remove(f"{run_dir}/manifest.running.json")
print(f"manifest.json written (publishable={manifest['publishable']})")
EOF
STATUS_WRITTEN=1
echo "== complete: $RUN_DIR =="
