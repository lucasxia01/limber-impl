#!/usr/bin/env python3
"""Stand-in for the limber `poseidon_modp` bench binary (limber contract section 4).

argv router (after the Cargo-injected trailing `--bench` is stripped):
  []                                                          smoke, exit 0
  ["--print-protocol-metadata"]                               one canonical JSON line
  ["--run-config", P, "--config-sha256", H,
   "--attempt", "preflight", "--artifact-dir", D]             preflight attempt (form 2)
  ["--child-config", P, "--child-config-sha256", H,
   "--artifact-dir", D]                                       Criterion child (form 3)
Anything else is a usage error (stderr `poseidon_modp: usage error: ...`, exit 64).

Form 2 writes `preflight.json` + `proof-size.json` (normal/psize) or `sweep-metadata.json`
(sweep), always followed by `attempt-result.json`; exit 0, or 3 on a recorded failure.
Form 3 writes the Criterion 0.7.0 tree under `D/criterion` with deterministic timings that
make the smallest k the unique best candidate.
Failure injection (from the closed environment the stub `be` builds):
  STUB_FAIL_PREFLIGHT=<k>:<instance>, STUB_FAIL_CHILD=<k>:<instance>:<block>.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from scripts import poseidon_common as pc  # noqa: E402

UNSAFE = '?"/\\*<>:|^'
RUN_CONFIG_SCHEMA = "limber/poseidon-run-config/v2"
CHILD_CONFIG_SCHEMA = "limber/poseidon-child-config/v2"


def usage(msg: str) -> int:
    sys.stderr.write("poseidon_modp: usage error: %s\n" % msg)
    return 64


def validation(msg: str) -> int:
    sys.stderr.write("poseidon_modp: validation error: %s\n" % msg)
    return 64


def tuned_default(md, backend):
    for entry in md.get("tuned_defaults") or []:
        if isinstance(entry, dict) and entry.get("backend") == backend:
            return entry.get("k"), entry.get("tuning_id")
        if isinstance(entry, list) and entry[:1] == [backend]:
            return entry[1], entry[2]
    return None, None


def check_config_binding(cfg):
    """The run-config checks the real bench performs before any output."""
    md = metadata()
    role = os.environ.get("STUB_SYSTEM_ROLE", "hyrax")
    if cfg.get("schema") != RUN_CONFIG_SCHEMA or cfg.get("system_role") != role or \
            cfg["ids"]["compiled"] != md:
        return "run-config is not for this bench binary"
    if cfg.get("backend") != role or cfg["workload"]["backend"] != role:
        return "run-config backend is not this role"
    if cfg["workload"]["hashes_per_field"] != 10:
        return "workload.hashes_per_field must be 10"
    if os.environ.get("RAYON_NUM_THREADS") != str(cfg["workload"]["threads"]):
        return "RAYON_NUM_THREADS does not match workload.threads"
    if cfg.get("features") != []:
        return "the limber bench has no features"
    if cfg.get("rustflags") != "-C target-cpu=native":
        return "rustflags must be -C target-cpu=native"
    if cfg.get("check_modes") != {"shadow": "not_applicable", "timed": "release"}:
        return "check_modes must be not_applicable/release"
    if cfg["ids"].get("benchmark_coins_policy", {}).get("id") != md.get("benchmark_coins_id"):
        return "benchmark coins policy differs from the compiled id"
    if cfg.get("dimensions_by_candidate") != {k: v for k, v in md["dimensions_by_k"].items()
                                              if k in cfg["dimensions_by_candidate"]}:
        return "dimensions_by_candidate differs from the compiled dimensions"
    if cfg["mode"] in ("normal", "psize"):
        k_default, tid = tuned_default(md, role)
        if cfg["tuning"]["tuning_id"] != tid or \
                cfg["tuning"]["tuning_epoch_id"] != md.get("tuning_epoch_id"):
            return "tuning ids differ from the compiled tuned defaults"
        if k_default is not None and cfg["k"] != k_default:
            return "k differs from the compiled tuned default"
        if cfg["workload"]["k"] != cfg["k"] or not 7 <= cfg["k"] <= 13:
            return "k is not a valid candidate"
    return None


def safe(s):
    s = "".join("_" if c in UNSAFE else c for c in s)
    return s[:64]


def names(group, func, value):
    full = "%s/%s/%s" % (group, func, value)
    return {"full_id": full,
            "directory_name": "%s/%s/%s" % (safe(group), safe(func), safe(value)),
            "title": full[:100] + "..." if len(full) > 100 else full}


def load_config(path, sha):
    if not os.path.isabs(path) or not os.path.isfile(path) or os.path.islink(path):
        return None, "config path must be an absolute regular file"
    with open(path, "rb") as f:
        data = f.read()
    if not pc.is_hex64(sha) or pc.sha256_hex(data) != sha:
        return None, "config digest mismatch"
    try:
        return pc.load_canonical_json(data), None
    except pc.RunnerError as exc:
        return None, str(exc)


def check_artifact_dir(d):
    if not os.path.isabs(d) or not os.path.isdir(d) or os.path.realpath(d) != d:
        return "artifact dir must be an absolute symlink-free directory"
    if os.listdir(d):
        return "artifact dir must be empty"
    return None


def metadata():
    with open(os.environ["STUB_METADATA_JSON"], "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def write(d, name, obj):
    pc.write_canonical_json(os.path.join(d, name), obj)


def hexdigest(*parts) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def audit_records(backend, k, inst):
    """A deterministic complete P0-D audit: one runtime prime plus one small prime per k."""
    records = [{"purpose": "RuntimeP", "width_bits": 128, "candidates": 40 + inst,
                "bases_accepted": 72, "bases_rejected": 1, "mr_rounds_completed": 72,
                "rolling_digest": hexdigest("runtime", backend, k, inst),
                "outcome": {"success": "01" * 16}}]
    for j in range(k - 6):
        records.append({"purpose": "IntEvalSmallP", "width_bits": 64, "candidates": 12 + j,
                        "bases_accepted": 72, "bases_rejected": 0, "mr_rounds_completed": 72,
                        "rolling_digest": hexdigest("small", backend, k, inst, j),
                        "outcome": {"success": "02" * 8}})
    return records


def preflight_object(cfg, sha, k, inst, threads, fail):
    md = cfg["ids"]["compiled"]
    backend = cfg["backend"]
    records = audit_records(backend, k, inst)
    return {"schema": "limber/poseidon-preflight/v2", "config_sha256": sha,
            "backend": backend, "k": k, "hashes_per_field": cfg["workload"]["hashes_per_field"],
            "instance": inst, "dimensions": md["dimensions_by_k"][str(k)],
            "status": "failed" if fail else "ok",
            "rayon_threads_asserted": threads,
            "prime_audit": {"complete": True, "prime_sampler_id": md["prime_sampler_id"],
                            "invocations": len(records), "records": records},
            "double_construction": {"commitments_equal": True, "components_equal": not fail,
                                    "remainder_equal": True, "audits_equal": True},
            "digests": [hexdigest("digest", backend, k, inst, i) for i in range(3)],
            "canonical_io_ok": True, "shape_satisfiable": True,
            "error": {"stage": "double_construction", "code": "ComponentsDiffer",
                      "message": "injected"} if fail else None}


def proof_size_object(cfg, sha, k, inst):
    md = cfg["ids"]["compiled"]
    backend = cfg["backend"]
    return {"schema": "limber/poseidon-proof-size/v2", "metric_kind": "mixed_payload_estimate",
            "config_sha256": sha, "comparison_key": cfg["comparison_key"],
            "wire_id": md["protocol_wire_id"], "backend": backend, "k": k,
            "hashes_per_field": cfg["workload"]["hashes_per_field"], "instance": inst,
            "components": {"commitments_bytes": 4096 + 32 * k,
                           "eval_arg_bytes": 120000 + 1000 * k + inst,
                           "commitments_sha256": hexdigest("commitments", backend, k, inst),
                           "eval_arg_sha256": hexdigest("eval_arg", backend, k, inst)},
            "analytical_remainder": {"sumcheck_rounds": 14, "remainder_elements": 3 * 14,
                                     "element_bits": 128, "remainder_bits": 3 * 14 * 128},
            "prime_sampler_id": md["prime_sampler_id"], "verified": True}


def attempt_result(d, cfg, sha, status, error):
    outputs = []
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        outputs.append({"name": name, "size": os.path.getsize(full),
                        "sha256": pc.sha256_file(full)})
    write(d, "attempt-result.json", {
        "schema": "limber/poseidon-attempt-result/v2", "attempt": "preflight",
        "mode": cfg["mode"], "config_sha256": sha, "status": status, "outputs": outputs,
        "error": error, "compiled": metadata()})


def injected_preflight_failure(k, inst) -> bool:
    return os.environ.get("STUB_FAIL_PREFLIGHT") == "%d:%d" % (k, inst)


def form_preflight(path, sha, d):
    cfg, why = load_config(path, sha)
    if cfg is None:
        return usage(why)
    why = check_artifact_dir(d)
    if why:
        return usage(why)
    why = check_config_binding(cfg)
    if why:
        return validation(why)
    mode = cfg["mode"]
    threads = cfg["workload"]["threads"]
    if mode in ("normal", "psize"):
        k, inst = cfg["k"], cfg["workload"]["instance"]
        fail = injected_preflight_failure(k, inst)
        write(d, "preflight.json", preflight_object(cfg, sha, k, inst, threads, fail))
        if fail:
            attempt_result(d, cfg, sha, "failed", {"stage": "preflight", "code":
                                                   "ComponentsDiffer", "message": "injected"})
            return 3
        write(d, "proof-size.json", proof_size_object(cfg, sha, k, inst))
        attempt_result(d, cfg, sha, "ok", None)
        return 0
    if mode == "sweep":
        sweep = cfg["sweep"]
        matrix = []
        first_error = None
        for cand in sweep["candidates"]:
            for inst in sweep["instances"]:
                fail = injected_preflight_failure(cand, inst)
                pre = preflight_object(cfg, sha, cand, inst, threads, fail)
                matrix.append({"candidate": cand, "instance": inst,
                               "dimensions": sweep["dimensions_by_candidate"][str(cand)],
                               "status": "failed" if fail else "ok", "preflight": pre})
                if fail and first_error is None:
                    first_error = {"stage": "preflight", "code": "ComponentsDiffer",
                                   "message": "injected failure for k%d inst%d" % (cand, inst)}
        write(d, "sweep-metadata.json", {"schema": "limber/poseidon-sweep-metadata/v2",
                                         "config_sha256": sha,
                                         "candidates": sweep["candidates"],
                                         "instances": sweep["instances"], "matrix": matrix})
        attempt_result(d, cfg, sha, "failed" if first_error else "ok", first_error)
        return 3 if first_error else 0
    return usage("unknown mode %r" % mode)


def write_criterion(out, cfg, groups, block, k, inst):
    backend = cfg["backend"]
    hashes = cfg["workload"]["hashes_per_field"]
    threads = cfg["workload"]["threads"]
    dims = cfg["dimensions_by_candidate"][str(k)]
    prefix = "%s/mixed3/Hpf%d-total%d/c2^%dv2^%d/k%d/inst%d/thr%d" % (
        backend, hashes, 3 * hashes, dims["log_cons"], dims["log_vars"], k, inst, threads)
    for group in groups:
        primary = group in ("prove_e2e", "verify_core")
        func = prefix + ("/primary/blk%s" % block if primary else "/diagnostic/blkdiag")
        # Timings grow with k so that the smallest k is the unique best candidate.
        base = 100000000.5 + (k - 7) * 4000000.0
        point = base + inst * 1000 + (int(block) if primary else 0) * 10
        n = names(group, func, "")
        bench = {"group_id": group, "function_id": func, "value_str": "",
                 "throughput": None, "full_id": n["full_id"],
                 "directory_name": n["directory_name"], "title": n["title"]}
        est = {}
        for stat in ("mean", "median", "median_abs_dev", "slope", "std_dev"):
            est[stat] = {"confidence_interval": {"confidence_level": 0.95,
                                                 "lower_bound": point - 500000.0,
                                                 "upper_bound": point + 500000.0},
                         "point_estimate": point, "standard_error": 1000.0}
        leafdir = os.path.join(out, *[c for c in n["directory_name"].split("/") if c])
        for sub in ("base", "new"):
            dd = os.path.join(leafdir, sub)
            os.makedirs(dd)
            with open(os.path.join(dd, "benchmark.json"), "w") as f:
                json.dump(bench, f)
            with open(os.path.join(dd, "estimates.json"), "w") as f:
                json.dump(est, f)
            with open(os.path.join(dd, "sample.json"), "w") as f:
                json.dump({"sampling_mode": "Linear", "iters": [2.0, 4.0],
                           "times": [point * 2, point * 4]}, f)
            with open(os.path.join(dd, "tukey.json"), "w") as f:
                json.dump([point * 0.9, point * 0.95, point * 1.05, point * 1.1], f)


def form_child(path, sha, d):
    cc, why = load_config(path, sha)
    if cc is None:
        return usage(why)
    why = check_artifact_dir(d)
    if why:
        return usage(why)
    role = os.environ.get("STUB_SYSTEM_ROLE", "hyrax")
    if cc.get("schema") != CHILD_CONFIG_SCHEMA or cc.get("system_role") != role:
        return validation("child-config is not for this bench binary")
    cfg = cc["parent_config"]
    if pc.sha256_hex(pc.canonical_json_bytes(cfg)) != cc["parent_config_sha256"]:
        return validation("parent_config digest mismatch")
    if sorted(cc) != ["coordinate", "order", "ordinal", "parent_config", "parent_config_sha256",
                      "schema", "session_id", "system_role"]:
        return validation("child-config keys are not the contract set")
    why = check_config_binding(cfg)
    if why:
        return validation(why)
    coord = cc["coordinate"]
    if coord.get("kind") == "sweep" and (cc["order"] is not None or sorted(coord) != [
            "block", "candidate", "instance", "kind", "ordinal"]):
        return validation("sweep coordinate/order shape")
    if coord["kind"] == "primary":
        groups, block, k, inst = [coord["metric"]], coord["block"], cfg["k"], \
            cfg["workload"]["instance"]
    elif coord["kind"] == "diagnostic":
        groups, block, k, inst = list(pc.diagnostic_groups_for(role)), "diag", cfg["k"], \
            cfg["workload"]["instance"]
    elif coord["kind"] == "sweep":
        if not isinstance(coord["candidate"], int):
            return validation("sweep candidate must be the integer k")
        groups, block, k, inst = ["prove_e2e"], coord["block"], coord["candidate"], \
            coord["instance"]
    else:
        return usage("unknown coordinate kind")
    sys.stderr.write("[stub] child %s k=%d inst=%d block=%s\n" % (coord["kind"], k, inst, block))
    if os.environ.get("STUB_FAIL_CHILD") == "%d:%d:%s" % (k, inst, block):
        sys.stderr.write("[stub] injected child failure\n")
        return 101
    write_criterion(os.path.join(d, "criterion"), cfg, groups, block, k, inst)
    return 0


def main():
    args = sys.argv[1:]
    if not args:
        return 0  # smoke
    if args[-1] != "--bench" or args.count("--bench") != 1:
        return usage("exactly one trailing --bench is required")
    args = args[:-1]
    if args == ["--print-protocol-metadata"]:
        sys.stdout.write(pc.canonical_json_bytes(metadata()).decode("ascii"))
        return 0
    if len(args) == 8 and args[0] == "--run-config" and args[2] == "--config-sha256" and \
            args[4:6] == ["--attempt", "preflight"] and args[6] == "--artifact-dir":
        return form_preflight(args[1], args[3], args[7])
    if len(args) == 6 and args[0] == "--child-config" and args[2] == "--child-config-sha256" \
            and args[4] == "--artifact-dir":
        return form_child(args[1], args[3], args[5])
    return usage("unrecognized argument form %r" % (args,))


if __name__ == "__main__":
    sys.exit(main())
