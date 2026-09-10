#!/usr/bin/env python3
"""Re-execute TUNE-1 over a finalized limber sweep archive (plan v10, section 9).

    summarize_poseidon_sweep.py <sweep archive> [--out FILE] [--quiet]

The sweep runner writes `tuning-result.json` while it runs; this tool is the independent
re-derivation the plan requires before a sweep may become a tuning origin (it replaces the
v9 `summarize_poseidon_ksweep.py`). It

1. validates the archive (detached manifest digest, artifact index, exact sweep file set,
   durability confirmation), requires `mode = sweep` and `status = complete`;
2. re-reads every child's Criterion `estimates.json` through the exact-decimal parser and the
   observed 0.7.0 path grammar, rebuilds the `2N x 9 x N` process table from the archived
   `sweep-metadata.json` schedule (N = 7 candidate k values);
3. re-executes the TUNE-1 statistic/decision (simplicity order = ascending k) and re-renders
   the canonical `tuning-result.json` object with the provenance digests recomputed from the
   archive (run-config digest, source snapshot/lock/dependency-source IDs, sweep-metadata
   digest, every child config/result digest and every consumed `estimates.json`, whose
   `median.confidence_interval.confidence_level` must be exactly 0.95);
4. requires the re-rendered bytes to equal the archived `tuning-result.json`, its detached
   `tuning-result.sha256` and the manifest's `tuning_id`.

Exit 0 and prints `tuning_id <hex>` on success; `--out` additionally writes the re-rendered
canonical bytes. Any mismatch is a non-zero exit with a stable `[stage/error_code]` message.
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    from scripts import artifact_fs as af
    from scripts import poseidon_common as pc
    from scripts import poseidon_runner as runner
    from scripts import poseidon_tune1 as t1
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import artifact_fs as af  # noqa: E402
    import poseidon_common as pc  # noqa: E402
    import poseidon_runner as runner  # noqa: E402
    import poseidon_tune1 as t1  # noqa: E402


def load_json_file(path: str, what: str):
    with open(path, "rb") as f:
        return pc.load_canonical_json(f.read(), what)


def rederive(archive: str) -> tuple:
    """Return `(manifest, recomputed_result_bytes, archived_result_bytes, tuning_id)`."""
    archive = os.path.realpath(archive)
    required, globs = runner.expected_payload_set("sweep", "complete")
    manifest, manifest_sha, status = af.verify_artifact_dir(archive, required, globs)
    if status != "complete" or manifest.get("mode") != "sweep":
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "%s is not a complete sweep archive" % archive)
    cfg = load_json_file(os.path.join(archive, "run-config.json"), "run-config.json")
    config_id = pc.read_detached_digest(os.path.join(archive, "run-config.sha256"))
    if pc.sha256_file(os.path.join(archive, "run-config.json")) != config_id or \
            manifest.get("config_id") != config_id:
        raise pc.RunnerError("aggregate", "IntegrityMismatch", "run-config digest mismatch")
    runner.check_child_trees(archive, "sweep")
    meta_path = os.path.join(archive, "sweep-metadata.json")
    meta = load_json_file(meta_path, "sweep-metadata.json")
    if meta.get("schema") != runner.SWEEP_METADATA_SCHEMA or \
            meta.get("config_sha256") != config_id:
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "sweep-metadata.json does not bind this run config")
    backend = cfg["backend"]
    if backend not in pc.BACKENDS or cfg.get("system_role") != backend or \
            cfg.get("workload", {}).get("backend") != backend:
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "archived backend/system_role are inconsistent")
    group = {"backend": backend}
    candidates = list(meta["candidates"])
    instances = list(meta["instances"])
    sweep = cfg.get("sweep") or {}
    if candidates != list(pc.CANDIDATES) or instances != list(pc.TUNING_INSTANCES) or \
            sweep.get("candidates") != candidates or sweep.get("instances") != instances or \
            sweep.get("simplicity_order") != list(pc.SIMPLICITY_ORDER):
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "sweep candidates/instances are not the TUNE-1 lists")
    dims = sweep.get("dimensions_by_candidate")
    if not isinstance(dims, dict) or sorted(dims) != sorted(str(c) for c in candidates) or \
            cfg.get("dimensions_by_candidate") != dims:
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "archived dimensions_by_candidate is not the candidate map")
    orders = t1.block_orders(candidates)
    if sweep.get("block_orders") != [{"block": b, "order": o} for b, o in enumerate(orders)] \
            or sweep.get("blocks") != len(orders):
        raise pc.RunnerError("aggregate", "TuningResultInvalid", "archived block schedule "
                             "differs from the TUNE-1 rule")
    matrix = meta.get("matrix")
    pairs = sorted((c, i) for c in candidates for i in instances)
    threads = cfg["workload"]["threads"]
    sampler = cfg["ids"]["compiled"].get("prime_sampler_id")
    if not isinstance(matrix, list) or \
            sorted((m.get("candidate"), m.get("instance")) for m in matrix) != pairs or \
            not all(m.get("status") == "ok" and not runner.validate_preflight(
                m.get("preflight"), {"backend": backend, "instance": m.get("instance"),
                                     "k": m.get("candidate")}, config_id, threads, sampler)
                    for m in matrix):
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "preflight matrix is incomplete or has inadmissible entries")
    processes = {}
    children = []
    estimates_files = []
    for (b, o, cand, inst) in t1.schedule(candidates, instances):
        rel = "children/block-%d/%s" % (b, t1.child_dir_name(o, cand, inst))
        child_dir = os.path.join(archive, *rel.split("/"))
        if not os.path.isdir(child_dir):
            raise pc.RunnerError("aggregate", "TuningResultInvalid", "missing child %s" % rel)
        expectation = {"groups": {"prove_e2e"}, "backend": backend,
                       "hashes": pc.CANONICAL_HASHES, "instance": inst, "k": cand,
                       "threads": 1, "kind": "primary", "block": b,
                       "dimensions": dims[str(cand)]}
        recs = pc.validate_criterion_tree(os.path.join(child_dir, "criterion"), expectation)
        if len(recs) != 1:
            raise pc.RunnerError("aggregate", "TuningResultInvalid",
                                 "child %s registered %d ids" % (rel, len(recs)))
        est = recs[0]["estimates"]["median"]
        processes[(cand, inst, b)] = {"point": est["point_estimate"],
                                      "lower": est["lower_bound"],
                                      "upper": est["upper_bound"]}
        est_rel = rel + "/criterion/" + recs[0]["estimates_path"]
        full = os.path.join(archive, *est_rel.split("/"))
        estimates_files.append([est_rel, os.path.getsize(full), pc.sha256_file(full)])
        child_cfg = load_json_file(os.path.join(child_dir, "child-config.json"),
                                   "child-config.json")
        if child_cfg.get("parent_config_sha256") != config_id or \
                child_cfg.get("order") is not None or child_cfg.get("coordinate") != {
                    "kind": "sweep", "block": b, "ordinal": o, "candidate": cand,
                    "instance": inst}:
            raise pc.RunnerError("aggregate", "TuningResultInvalid",
                                 "child %s does not bind the parent config/coordinate" % rel)
        children.append({"block": b, "ordinal": o, "candidate": cand, "instance": inst,
                         "path": rel,
                         "child_config_sha256": pc.sha256_file(
                             os.path.join(child_dir, "child-config.json")),
                         "child_result_sha256": pc.sha256_file(
                             os.path.join(child_dir, "child-result.json"))})
    ids = dict(cfg["ids"]["compiled"])
    ids["tracked_specs"] = cfg["ids"]["tracked_specs"]
    result = t1.tuning_result(group, ids, candidates, instances, processes, {
        "run_config_sha256": config_id,
        "source_sha": cfg["source"]["git_sha"],
        "source_snapshot_id": cfg["source"]["source_snapshot_id"],
        "lockfile_sha256": cfg["lockfile"]["sha256"],
        "dependency_source_id": cfg["dependency_sources"]["dependency_source_id"],
        "sweep_metadata_sha256": pc.sha256_file(meta_path),
        "children": children,
        "estimates_files": sorted(estimates_files),
    }, simplicity_order=list(pc.SIMPLICITY_ORDER))
    recomputed = pc.canonical_json_bytes(result)
    with open(os.path.join(archive, "tuning-result.json"), "rb") as f:
        archived = f.read()
    tuning_id = pc.read_detached_digest(os.path.join(archive, "tuning-result.sha256"))
    t1.recheck_tuning_result(pc.load_canonical_json(archived, "tuning-result.json"))
    if recomputed != archived:
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "re-derived tuning result differs from the archived bytes")
    if pc.sha256_hex(archived) != tuning_id or manifest.get("tuning_id") != tuning_id:
        raise pc.RunnerError("aggregate", "TuningResultInvalid",
                             "tuning-result.sha256 / manifest tuning_id do not match")
    return manifest, recomputed, archived, tuning_id


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="summarize_poseidon_sweep.py",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("archive", help="finalized sweep archive directory")
    parser.add_argument("--out", help="write the re-derived canonical tuning-result.json here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest, recomputed, _, tuning_id = rederive(args.archive)
        if args.out:
            pc.write_bytes(args.out, recomputed)
        if not args.quiet:
            result = pc.load_canonical_json(recomputed, "tuning-result.json")
            print("tuning_id %s" % tuning_id)
            print("backend %s selected k=%s (%s) eligible_for %s" % (
                result["group"]["backend"], result["decision"]["selected"],
                result["decision"]["selection_basis"], manifest.get("eligible_for")))
        return 0
    except pc.RunnerError as exc:
        pc.eprint("summarize_poseidon_sweep: error %s" % exc)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
