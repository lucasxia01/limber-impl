#!/usr/bin/env python3
"""Assemble a same-source tuning epoch from eligible limber sweep archives (plan v10,
section 9; limber contract section 6).

    assemble_poseidon_tuning.py --group-set limber-both --store DIR --out BUNDLE \\
        ARCHIVE [ARCHIVE ...]

For the explicit ordered group set (`limber-both` = the `hyrax` and `brakedown` backends, in
that order), exactly one sweep archive per backend must be given. Each archive is validated
(manifest digest, index, sweep file set and child trees, durability confirmation), must
carry `eligible_for` containing `tuning`, must re-derive under `summarize_poseidon_sweep`,
and is imported into the content-addressed store as `<store>/<origin manifest sha256>/`:
copy to an exclusively created sibling staging directory, rehash, fsync, no-replace rename.
An existing object, including a rename-race winner, is reused only after the same full
rehash, bottom-up fsync and store-directory fsync. The store must be an absolute,
pre-existing, writable, symlink-free directory whose canonical path is disjoint in both
directions from every input archive, the source worktree and the bundle-output parent.
Every origin must share one source SHA/snapshot, one lockfile SHA-256, one compiled
capability tuple and the shared corpus/protocol/timing/security IDs; the assembler then
writes the canonical `tuning-bundle.json` (source/lock/snapshot IDs, per-group
`dependency_source_id`, origin manifest hash, `tuning_id` and selected `k`; plus
`tuning-bundle.sha256`) and prints `tuning_epoch_id <hex>`. `scripts/apply_poseidon_tuning.py`
renders the tracked defaults module from that bundle. The Zinc orchestrator resolves a
group with `bundle_group(bundle, backend=role)`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

try:
    from scripts import artifact_fs as af
    from scripts import poseidon_common as pc
    from scripts import poseidon_runner as runner
    from scripts import summarize_poseidon_sweep as summarize
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import artifact_fs as af  # noqa: E402
    import poseidon_common as pc  # noqa: E402
    import poseidon_runner as runner  # noqa: E402
    import summarize_poseidon_sweep as summarize  # noqa: E402

BUNDLE_SCHEMA = "limber/poseidon-tuning-bundle/v1"
GENERATED_MODULE_PATH = "src/poseidon_tuned_defaults.rs"
GROUP_SETS = {
    "limber-both": ("hyrax", "brakedown"),
}
CAPABILITY_KEYS = ("protocol_wire_id", "transcript_domain_separator", "prime_sampler_id",
                   "benchmark_coins_id")
SHARED_ID_KEYS = ("kat_fixture_sha256", "tuning_corpus_id", "tuning_protocol_id",
                  "timing_schema_id", "security_accounting_id", "comparison_schema_id")


def _overlaps(a: str, b: str) -> bool:
    a, b = a.rstrip("/"), b.rstrip("/")
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def check_store(path: str, disjoint_from=()) -> str:
    if not os.path.isabs(path) or os.path.realpath(path) != path.rstrip("/") or \
            not os.path.isdir(path) or os.path.islink(path):
        raise pc.RunnerError("env", "OutputParentInvalid",
                             "--store must be an absolute, existing, symlink-free directory")
    if not os.access(path, os.W_OK | os.X_OK):
        raise pc.RunnerError("env", "OutputParentInvalid", "--store is not writable")
    real = os.path.realpath(path)
    for what, other in disjoint_from:
        if other and _overlaps(real, os.path.realpath(other)):
            raise pc.RunnerError("env", "PathOverlap", "--store %s overlaps the %s %s" %
                                 (real, what, other))
    return path.rstrip("/")


def import_origin(store: str, archive: str, manifest_sha: str) -> str:
    """Copy `archive` into the store under its manifest digest (no-replace), or reuse an
    identical existing object (a rename-race winner included) after a full rehash and
    durability confirmation."""
    required, globs = runner.expected_payload_set("sweep", "complete")
    final = os.path.join(store, manifest_sha)
    if os.path.lexists(final):
        af.verify_and_reuse_existing(final, required, globs, manifest_sha)
        runner.check_child_trees(final, "sweep")
        return final
    staging = os.path.join(store, ".%s.staging" % manifest_sha)
    if os.path.lexists(staging):
        raise pc.RunnerError("staging", "StagingExists", "%s exists" % staging)
    shutil.copytree(archive, staging, symlinks=False)
    # The archive itself was fully verified by the re-derivation; the staged copy must be
    # the same manifest-bound file set byte for byte (symlinks/unexpected files rejected).
    if pc.hashed_file_list(staging) != pc.hashed_file_list(archive) or \
            pc.sha256_file(os.path.join(staging, "manifest.json")) != manifest_sha:
        shutil.rmtree(staging)
        raise pc.RunnerError("link", "IntegrityMismatch", "copied origin differs from the "
                             "archive")
    try:
        af.commit_staging(staging, final)
    except pc.RunnerError as exc:
        if exc.error_code != "RenameRefused":
            raise
        # Rename race: another importer won; discard our copy and reuse the winner only
        # after the same full rehash and durability confirmation.
        shutil.rmtree(staging, ignore_errors=True)
    manifest, sha, status = af.verify_and_reuse_existing(final, required, globs, manifest_sha)
    runner.check_child_trees(final, "sweep")
    if status != "complete":
        raise pc.RunnerError("link", "IntegrityMismatch", "imported origin is not complete")
    return final


def origin_record(archive: str) -> dict:
    manifest, recomputed, _, tuning_id = summarize.rederive(archive)
    if "tuning" not in manifest.get("eligible_for", []):
        raise pc.RunnerError("link", "BundleInvalid",
                             "%s is not tuning-eligible (%s)" % (archive, manifest.get(
                                 "failure_reasons")))
    result = pc.load_canonical_json(recomputed, "tuning-result.json")
    cfg = summarize.load_json_file(os.path.join(archive, "run-config.json"), "run-config.json")
    compiled = cfg["ids"]["compiled"]
    return {
        "archive": os.path.realpath(archive),
        "manifest_sha256": pc.read_detached_digest(os.path.join(archive, "manifest.sha256")),
        "tuning_id": tuning_id,
        "group": result["group"]["backend"],
        "selected": result["decision"]["selected"],
        "selection_basis": result["decision"]["selection_basis"],
        "source_sha": cfg["source"]["git_sha"],
        "source_snapshot_id": cfg["source"]["source_snapshot_id"],
        "lockfile_sha256": cfg["lockfile"]["sha256"],
        "dependency_source_id": cfg["dependency_sources"]["dependency_source_id"],
        "capability": {k: compiled.get(k) for k in CAPABILITY_KEYS},
        "shared_ids": {k: compiled.get(k) for k in SHARED_ID_KEYS},
    }


def assemble(group_set: str, store: str, archives) -> tuple:
    groups = GROUP_SETS[group_set]
    records = [origin_record(a) for a in archives]
    by_group = {}
    for rec in records:
        if rec["group"] in by_group:
            raise pc.RunnerError("link", "BundleInvalid", "two archives for backend %s" %
                                 rec["group"])
        by_group[rec["group"]] = rec
    missing = [g for g in groups if g not in by_group]
    extra = [g for g in by_group if g not in groups]
    if missing or extra:
        raise pc.RunnerError("link", "BundleInvalid", "group set %s: missing %s, extra %s" %
                             (group_set, missing, extra))
    first = by_group[groups[0]]
    for g in groups:
        rec = by_group[g]
        for key in ("source_sha", "source_snapshot_id", "lockfile_sha256", "capability",
                    "shared_ids"):
            if rec[key] != first[key]:
                raise pc.RunnerError("link", "BundleInvalid",
                                     "origin %s differs from the epoch in %s" % (g, key))
    entries = []
    for g in groups:
        rec = by_group[g]
        import_origin(store, rec["archive"], rec["manifest_sha256"])
        entries.append({"backend": g, "tuning_id": rec["tuning_id"],
                        "origin_manifest_sha256": rec["manifest_sha256"],
                        "dependency_source_id": rec["dependency_source_id"],
                        "selected": rec["selected"],
                        "selection_basis": rec["selection_basis"]})
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "group_set": group_set,
        "source_sha": first["source_sha"],
        "source_snapshot_id": first["source_snapshot_id"],
        "lockfile_sha256": first["lockfile_sha256"],
        "capability": first["capability"],
        "shared_ids": first["shared_ids"],
        "generated_module_path": GENERATED_MODULE_PATH,
        "groups": entries,
    }
    data = pc.canonical_json_bytes(bundle)
    return bundle, data, pc.sha256_hex(data)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="assemble_poseidon_tuning.py",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--group-set", required=True, choices=sorted(GROUP_SETS))
    parser.add_argument("--store", required=True, help="content-addressed tuning store")
    parser.add_argument("--out", required=True, help="path of the bundle to write")
    parser.add_argument("archives", nargs="+")
    args = parser.parse_args(argv)
    try:
        out = os.path.abspath(args.out)
        store = check_store(args.store, [("source worktree", pc.repo_root()),
                                         ("bundle output parent", os.path.dirname(out))]
                            + [("input archive", a) for a in args.archives])
        if os.path.lexists(args.out):
            raise pc.RunnerError("staging", "StagingExists", "%s exists" % args.out)
        _, data, epoch_id = assemble(args.group_set, store, args.archives)
        pc.write_bytes(args.out, data)
        pc.write_detached_digest(os.path.splitext(args.out)[0] + ".sha256", epoch_id)
        print("tuning_epoch_id %s" % epoch_id)
        return 0
    except pc.RunnerError as exc:
        pc.eprint("assemble_poseidon_tuning: error %s" % exc)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
