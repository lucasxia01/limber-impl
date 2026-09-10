#!/usr/bin/env bash
# Explicit publication step for completed Poseidon2 benchmark runs
# (modp plan §12; spartan plan §6): validates every input run directory,
# enforces the compatibility gates, then copies immutable
# configs/manifests/logs/raw Criterion data into
# bench-results/poseidon2/<run-id>/ (ModP suite) or
# bench-results/poseidon2-spartan/<run-id>/ (classic-Spartan suite) and
# updates the suite's README section.
#
# Usage: scripts/publish_poseidon_bench.sh <run-dir> [<run-dir> ...]
#
# Gates:
# - every run must be complete, publishable, and hash-consistent;
# - runs of the SAME suite published together must agree on every
#   comparison-controlled field (only backend/k/mode-specific fields may
#   differ within the ModP suite);
# - runs of DIFFERENT suites published together must agree on the
#   cross-system key: source/lock hashes, machine, OS, toolchain,
#   allocator, thread count, RUSTFLAGS, workload version, field order,
#   and H — only the proof-system dimension may differ. Suite-specific
#   fields (backend, k, is_small, dims, encodings) are recorded, not
#   equated.
#
# Ordinary run output must never be written to bench-results/ directly —
# that directory is a committed publication target, and untracked output
# there would dirty the tree and fail the next canonical run's clean-tree
# gate. Do not run another canonical benchmark after publication dirties
# the tree unless these artifacts are committed first.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <completed-run-dir> [<completed-run-dir> ...]" >&2
  exit 2
fi

python3 - "$@" <<'EOF'
import hashlib, json, os, shutil, sys

run_dirs = sys.argv[1:]

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Validate every input run directory; abort on the first failure.
manifests, configs, suites = {}, {}, {}
for run_dir in run_dirs:
    mpath = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"{run_dir}: no manifest.json (incomplete run)")
    with open(mpath) as f:
        manifest = json.load(f)
    if manifest.get("status") != "complete":
        raise SystemExit(f"{run_dir}: manifest status is not complete")
    if not manifest.get("publishable"):
        raise SystemExit(f"{run_dir}: manifest is marked non-publishable")
    cfg_path = os.path.join(run_dir, "run-config.json")
    cfg_sha = sha256_file(cfg_path)
    with open(os.path.join(run_dir, "run-config.sha256")) as f:
        recorded = f.read().strip()
    if cfg_sha != recorded:
        raise SystemExit(f"{run_dir}: run-config.json does not re-hash to run-config.sha256")
    if cfg_sha != manifest["config_sha256"]:
        raise SystemExit(f"{run_dir}: run-config hash differs from the manifest's")
    for rel, expected in manifest["artifacts"].items():
        path = os.path.join(run_dir, rel)
        if not os.path.exists(path):
            raise SystemExit(f"{run_dir}: artifact {rel} is missing")
        if sha256_file(path) != expected:
            raise SystemExit(f"{run_dir}: artifact {rel} does not re-hash to its manifest value")
    with open(cfg_path) as f:
        cfg = json.load(f)
    configs[run_dir] = cfg
    manifests[run_dir] = manifest
    suites[run_dir] = cfg["protocol"].get("suite", "modp")

# Environment fields every comparison controls, regardless of suite.
def env_controlled(cfg):
    env = cfg["environment"]
    return {
        "git_sha": env["git_sha"],
        "cargo_lock_sha256": env["cargo_lock_sha256"],
        "cpu_model": env["cpu_model"],
        "os": env["os"], "kernel": env["kernel"], "arch": env["arch"],
        "rustc_vV": env["rustc_vV"], "cargo_version": env["cargo_version"],
        "criterion_version": env["criterion_version"],
        "allocator": env["allocator"], "jem_feature": env["jem_feature"],
        "rayon_num_threads": env["rayon_num_threads"],
        "rustflags": env["rustflags"],
        "target_features": env["target_features"],
    }

# Same-suite gate: every comparison-controlled field must agree.
def suite_controlled(cfg):
    proto = cfg["protocol"]
    base = env_controlled(cfg)
    base.update({
        "workload": proto["workload"],
        "hashes_per_field": proto["hashes_per_field"],
        "total_hashes": proto["total_hashes"],
        "circuit": proto["circuit"],
        "criterion": proto["criterion"],
    })
    if suites_of(proto) == "modp":
        base.update({
            "dims": proto["dims"], "num_io": proto["num_io"],
            "field_blocks": proto["field_blocks"],
            "allow_knobs": proto["allow_knobs"], "knobs": proto["knobs"],
            "log_t_f": proto["log_t_f"], "log_t": proto["log_t"],
        })
    else:
        base.update({
            "field_order": proto["field_order"],
            "emulation": proto["emulation"],
            "num_public": proto["num_public"],
            "is_small": proto["is_small"],
        })
    return base

def suites_of(proto):
    return proto.get("suite", "modp")

# Cross-suite gate: the normalized compatibility key (spartan plan §6).
def cross_key(cfg):
    proto = cfg["protocol"]
    field_order = proto.get("field_order", proto.get("field_blocks"))
    base = env_controlled(cfg)
    base.update({
        "workload": proto["workload"],
        "workload_digest": proto.get("workload_digest"),
        "hashes_per_field": proto["hashes_per_field"],
        "total_hashes": proto["total_hashes"],
        "field_order": field_order,
    })
    return base

by_suite = {}
for run_dir in run_dirs:
    by_suite.setdefault(suites[run_dir], []).append(run_dir)

for suite, dirs in by_suite.items():
    if len(dirs) > 1:
        base = suite_controlled(configs[dirs[0]])
        for run_dir in dirs[1:]:
            other = suite_controlled(configs[run_dir])
            diffs = [k for k in base if base[k] != other[k]]
            if diffs:
                raise SystemExit(
                    f"compatibility gate ({suite}): {dirs[0]} and {run_dir} differ on "
                    f"comparison-controlled fields {diffs}")

if len(by_suite) > 1:
    first = run_dirs[0]
    base = cross_key(configs[first])
    for run_dir in run_dirs[1:]:
        other = cross_key(configs[run_dir])
        diffs = [k for k in base if base[k] != other[k]]
        if diffs:
            raise SystemExit(
                f"cross-system gate: {first} and {run_dir} differ on "
                f"normalized compatibility fields {diffs}")

# Copy into the publication target.
DEST_BY_SUITE = {"modp": "poseidon2", "spartan": "poseidon2-spartan"}
published = []
for run_dir in run_dirs:
    run_id = os.path.basename(os.path.normpath(run_dir))
    dest = os.path.join("bench-results", DEST_BY_SUITE[suites[run_dir]], run_id)
    if os.path.exists(dest):
        raise SystemExit(f"{dest} already exists; refusing to overwrite a publication")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(run_dir, dest)
    published.append((run_id, suites[run_dir], configs[run_dir], manifests[run_dir]))
    print(f"published {run_dir} -> {dest}")

def replace_section(readme, begin, end, section):
    if begin in readme and end in readme:
        head, rest = readme.split(begin, 1)
        _old, tail = rest.split(end, 1)
        return head + section.rstrip("\n") + tail
    return readme.rstrip("\n") + "\n\n" + section

with open("README.md", encoding="utf-8") as f:
    readme = f.read()

modp_published = [(r, c, m) for r, s, c, m in published if s == "modp"]
if modp_published:
    BEGIN = "<!-- poseidon2-bench:begin -->"
    END = "<!-- poseidon2-bench:end -->"
    lines = [BEGIN, "## Poseidon2 non-native-field benchmark", ""]
    lines.append(
        "Thirty Poseidon2 compressions (t = 3, α = 5, R_F = 8, R_P = 56) proven "
        "in ONE mixed-modulus circuit: three independent ten-compression chains — "
        "one per field block, BN254-Fr, BLS12-381-Fr, secp256k1-Fr, in that fixed "
        "order — each restarting from the same fixed IV and ending at its own "
        "ordered public digest (num_io = 3; 12,990 real rows padded once to "
        "2^14 × 2^14). **This permutation is a benchmark workload, not a "
        "security-reviewed production hash** (custom BLAKE3-derived constants). "
        "**No zero-knowledge claim is made for this driver**: Hyrax commitments "
        "are hiding, Brakedown commitments are not, and the sumcheck transcript "
        "carries unmasked witness-dependent data regardless of backend; "
        "\"messages are private\" means *not public IO*, not confidential. "
        "Verification must go through "
        "`limber::poseidon2::verify_poseidon_chain` — bypassing it forfeits the "
        "three-digest canonicality guarantee. Published Brakedown timings are "
        "layout-warm steady state with an empty retained cache per measured "
        "sample.")
    lines.append("")
    lines.append("| run id | backend | mode | H/field (total) | k | git | config |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for run_id, cfg, manifest in modp_published:
        proto = cfg["protocol"]
        lines.append(
            f"| [`{run_id}`](bench-results/poseidon2/{run_id}/) "
            f"| {proto['backend']} | {proto['mode']} "
            f"| {proto['hashes_per_field']} ({proto['total_hashes']}) "
            f"| {proto['k']} | `{cfg['environment']['git_sha'][:12]}` "
            f"| `cfg-{manifest['config_sha256'][:12]}` |")
    lines.append("")
    lines.append(
        "Raw Criterion data, immutable run configs, manifests, and proof-size / "
        "k-sweep sidecars live in each run directory. Reproduce with the exact "
        "commands in `scripts/run_poseidon_bench.sh` (see plan §12).")
    lines.append(END)
    readme = replace_section(readme, BEGIN, END, "\n".join(lines) + "\n")

spartan_published = [(r, c, m) for r, s, c, m in published if s == "spartan"]
if spartan_published:
    BEGIN = "<!-- poseidon2-spartan-bench:begin -->"
    END = "<!-- poseidon2-spartan-bench:end -->"
    lines = [BEGIN, "## Emulated-field baseline (classic Spartan)", ""]
    lines.append(
        "The same 30-hash Poseidon2 workload proven as ONE limb-emulated "
        "circuit under classic Spartan (`SpartanSNARK<T256HyraxEngine>`, "
        "4 × 64-bit limbs via the revision-pinned `bellpepper-emulated` "
        "gadget): 9,454,119 real constraints padded to 2^24, against the "
        "ModP circuit's 12,990 rows padded to 2^14. The statement is "
        "existence-only, identical to the ModP suite's; **no zero-knowledge "
        "claim is made**. The emulated circuit is a good-faith optimized "
        "baseline (free linear layers, measured lazy-reduction schedule: "
        "two explicit reductions per S-box plus lanes 1–2 every 8th partial "
        "round), not a strawman. `prep_prove` is included in the headline "
        "`prove_e2e` prover time. Classic Spartan physically serializes the "
        "12 public limb scalars; the comparison payload excludes that "
        "statement data by convention, and its component sizes are canonical "
        "bincode while the ModP sumcheck remainder is an analytical payload "
        "without framing, so proof-size comparisons are not exact wire-format "
        "ratios. Hyrax-vs-Hyrax only: ModP Brakedown rows have no "
        "counterpart here. Verification must go through "
        "`limber::poseidon2_spartan::verify_poseidon_spartan`. This is the "
        "measured baseline the ModP plan's §4 declined to estimate.")
    lines.append("")
    lines.append("| run id | mode | H/field (total) | padded | git | config |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for run_id, cfg, manifest in spartan_published:
        proto = cfg["protocol"]
        lines.append(
            f"| [`{run_id}`](bench-results/poseidon2-spartan/{run_id}/) "
            f"| {proto['mode']} "
            f"| {proto['hashes_per_field']} ({proto['total_hashes']}) "
            f"| 2^24 × 2^24 | `{cfg['environment']['git_sha'][:12]}` "
            f"| `cfg-{manifest['config_sha256'][:12]}` |")
    lines.append("")
    lines.append(
        "Raw Criterion data, immutable run configs, manifests, and the "
        "proof-size sidecar live in each run directory. Reproduce with "
        "`scripts/run_poseidon_spartan_bench.sh` "
        "(see plan/poseidon_spartan_bench.md §6, §10).")
    lines.append(END)
    readme = replace_section(readme, BEGIN, END, "\n".join(lines) + "\n")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)
print("README.md section(s) updated")
EOF
