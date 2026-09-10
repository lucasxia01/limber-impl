#!/usr/bin/env python3
"""Closed build environment, source/dependency closures and evidence build (plan v10, §9).

Package C of the Python tooling split, limber adaptation. The module is the Zinc+
`scripts/poseidon_build_env.py` verbatim except for the system-profile hooks it takes from
`poseidon_common`: the evidence build accepts the limber roles (`hyrax`, `brakedown`) and
validates the `poseidon_modp` bench artifact of the single `limber` package
(`pc.evidence_artifact_rules`), the bootstrap `cargo metadata` command never carries a
feature (`pc.bootstrap_feature_tokens`), the bundled-zstd evidence requirement is Zinc-only
(`pc.REQUIRE_BUNDLED_ZSTD`), and the rustc argv grammar is read from the shared timing schema
under `rustc_argv_grammar.roles.<pc.GRAMMAR_ROLE>`; while that role key is absent (today the
schema pins only `zinc`) the evidence build fails closed with `RustcArgvInvalid` naming the
missing role, so the integrator has to capture the limber grammar from a real evidence build
and extend the schema before a real run can pass. Standard library only (Python >= 3.11). Every public
function raises `poseidon_common.RunnerError`; the codes it uses are listed in
`BUILD_ENV_ERROR_CODES` and live in the reserved build-env section of
`poseidon_common.ERROR_CODES`.

What is here:
  * `source_snapshot` / `recheck_source_snapshot`: an independent `os.scandir` walk compared
    against `git ls-tree -r HEAD` and the index, byte for byte (clean mode), or the all-file
    closure plus HEAD/index/diff state (dirty override).
  * `reject_inherited_environment`, `resolve_toolchain`, `resolve_native_tools`,
    `closed_environment`, `identity_projection`: the closed subprocess environment.
  * `cargo_config_audit` / `recheck_cargo_config`: Cargo's discovery order with the `term.*`
    allowlist and the pinned `[profile.bench]` block.
  * `dependency_source_audit` / `rehash_dependency_closure` / `replay_metadata`: the offline
    bootstrap `cargo metadata`, lockfile cross-check, registry `.crate`/unpacked-tree and git
    checkout audits, and the canonical `dependency-sources.json` document.
  * `evidence_build`, `build_profile_document`, `rehash_executable`, `rehash_all`,
    `run_cargo_subprocess`: the evidence build, the rustc argv grammar check, the native tool
    line check and the pre/post rehash wrapper for every later Cargo subprocess.

Raw absolute locators never enter an identity: `identity_projection` maps the audited roots
to `<SOURCE>`, `<CARGO_HOME>`, `<HOME>`, `<SYSROOT>`, `<METADATA_TARGET>`, `<ROLE_TARGET>`,
`<ROLE_TEMP>`, `<STAGING>` and `<CHILD_OUTPUT>`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib

try:
    from scripts import poseidon_common as pc
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import poseidon_common as pc  # noqa: E402

# Error codes this module raises (exit codes as in poseidon_common.ERROR_CODES). They are
# registered in the reserved build-env section of that table; if a future edit of the table
# drops one, it is re-registered here so RunnerError construction never fails.
BUILD_ENV_ERROR_CODES = {
    "RejectedEnvironment": 2, "SourceSnapshotInvalid": 3, "CargoConfigRejected": 3,
    "CargoConfigChanged": 3, "DependencySourceInvalid": 3, "DependencyClosureChanged": 3,
    "MetadataReplayMismatch": 3, "ToolchainMismatch": 3, "ToolchainChanged": 3,
    "NativeToolInvalid": 3, "NativeClosureChanged": 3, "EvidenceBuildFailed": 4,
    "ExecutableChanged": 4, "RustcArgvInvalid": 4, "IdentityProjectionInvalid": 4,
}
for _code, _exit in BUILD_ENV_ERROR_CODES.items():
    pc.ERROR_CODES.setdefault(_code, _exit)

DEPENDENCY_SOURCES_SCHEMA = "zinc-plus/dependency-sources/v1"
BUILD_PROFILE_SCHEMA = "zinc-plus/build-profile/v1"
CARGO_CONFIG_AUDIT_SCHEMA = "zinc-plus/cargo-config-audit/v1"
REJECTED_INPUTS_SCHEMA = "zinc-plus/rejected-inputs-audit/v1"
REGISTRY_MARKER_BYTES = b'{"v":1}'
REGISTRY_MARKER_SHA256 = "afbf9d0f3560b0fd7795e81c42a0a79ee6b6fc67e064f77826aee642cad28d91"
GIT_MARKER_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
INDEX_DIR_PREFIX = "index.crates.io-"
PROJECTION_TOKENS = ("<SOURCE>", "<CARGO_HOME>", "<HOME>", "<SYSROOT>", "<METADATA_TARGET>",
                     "<ROLE_TARGET>", "<ROLE_TEMP>", "<STAGING>", "<CHILD_OUTPUT>")
REJECTED_ENV_EXACT = (
    "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
    "CARGO", "CARGO_TARGET_DIR", "CARGO_INCREMENTAL", "CARGO_BUILD_TARGET",
    "CARGO_BUILD_RUSTFLAGS", "PYTHONPYCACHEPREFIX",
)
REJECTED_ENV_RULES = (
    "CARGO_PROFILE_*", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER", "CARGO", "CARGO_TARGET_DIR", "CARGO_INCREMENTAL",
    "CARGO_BUILD_TARGET", "CARGO_BUILD_*RUSTC*", "CARGO_BUILD_RUSTFLAGS",
    "CARGO_TARGET_*_{RUSTFLAGS,RUNNER,LINKER}", "PYTHONPYCACHEPREFIX",
)
RECHECK_POLICY = [
    "after_evidence_build", "before_metadata", "after_metadata", "before_kat", "after_kat",
    "before_tune_corpus", "after_tune_corpus", "before_preflight", "after_preflight",
    "before_each_child", "after_each_child", "after_terminal_attempt",
]
HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
ENV_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
RUNNING_PREFIX = "     Running `"
NATIVE_RUNNING_RE = re.compile(r"^\[([^\] ]+) ([^\]]+)\] running: (.*)$")
NATIVE_ENVREAD_RE = re.compile(r'^\[([^\] ]+) ([^\]]+)\] (CC|CXX|AR|RANLIB) = (None|Some\("(.*)"\))$')
BUILD_SCRIPT_LINE_RE = re.compile(r"^\[([^\] ]+) ([^\]]+)\] (.*)$")


def _err(stage, code, message):
    return pc.RunnerError(stage, code, message)


def _canon(obj) -> bytes:
    return pc.canonical_json_bytes(obj)


def _sha_obj(obj) -> str:
    return pc.sha256_hex(_canon(obj))


def _sha_stream(fobj) -> tuple:
    h = hashlib.sha256()
    size = 0
    while True:
        chunk = fobj.read(1 << 20)
        if not chunk:
            break
        size += len(chunk)
        h.update(chunk)
    return h.hexdigest(), size


def _git_blob_sha1(data: bytes) -> str:
    h = hashlib.sha1(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def _bytes_key(entry):
    return entry[0].encode("utf-8", "surrogateescape")


def _under(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/")


def _strict_under(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path.startswith(root + "/")


def _run(args, cwd, env, stage, code, what=None, stdin_devnull=True):
    try:
        return subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False,
                              stdin=subprocess.DEVNULL if stdin_devnull else None)
    except OSError as exc:
        raise _err(stage, code, "cannot execute %s: %s" % (what or args[0], exc)) from None


def _run_ok(args, cwd, env, stage, code, what=None) -> bytes:
    proc = _run(args, cwd, env, stage, code, what)
    if proc.returncode != 0:
        raise _err(stage, code, "%s exited %d: %s" %
                   (what or " ".join(args), proc.returncode,
                    proc.stderr.decode("utf-8", "replace").strip()[-1500:]))
    return proc.stdout


def _git(root: str, *args, stage="source", code="GitCommandFailed") -> bytes:
    return _run_ok(["git", "-C", root, *args], root, None, stage, code, "git " + " ".join(args))


def _git_mode(st_mode: int) -> str:
    if stat.S_ISLNK(st_mode):
        return "120000"
    if stat.S_ISDIR(st_mode):
        return "040000"
    return "100755" if st_mode & 0o111 else "100644"


# ---------------------------------------------------------------------------
# Independent filesystem walk.
# ---------------------------------------------------------------------------

def _admit_root_git(root: str, stage: str, code: str) -> dict:
    """Prove that `<root>/.git` is this checkout's own in-root administrative directory."""
    git_path = os.path.join(root, ".git")
    try:
        st = os.lstat(git_path)
    except FileNotFoundError:
        raise _err(stage, code, "%s has no .git directory" % root) from None
    if stat.S_ISLNK(st.st_mode):
        raise _err(stage, code, "%s is a symlink (Git indirection rejected)" % git_path)
    if not stat.S_ISDIR(st.st_mode):
        raise _err(stage, code, "%s is not a directory (gitfile / linked worktree rejected)"
                   % git_path)
    real_root = os.path.realpath(root)
    git_dir = _git(root, "rev-parse", "--git-dir", stage=stage).decode().strip()
    common_dir = _git(root, "rev-parse", "--git-common-dir", stage=stage).decode().strip()
    toplevel = _git(root, "rev-parse", "--show-toplevel", stage=stage).decode().strip()
    if not os.path.isabs(git_dir):
        git_dir = os.path.join(root, git_dir)
    if not os.path.isabs(common_dir):
        common_dir = os.path.join(root, common_dir)
    real_git = os.path.realpath(git_path)
    if os.path.realpath(git_dir) != real_git:
        raise _err(stage, code, "git rev-parse --git-dir resolves to %s, not %s" %
                   (git_dir, git_path))
    if os.path.realpath(common_dir) != real_git:
        raise _err(stage, code, "%s is a linked worktree (common dir %s)" % (root, common_dir))
    if not _strict_under(real_git, real_root):
        raise _err(stage, code, "%s resolves outside the checkout" % git_path)
    if os.path.realpath(toplevel) != real_root:
        raise _err(stage, code, "%s is not the top level of its repository (%s)" %
                   (root, toplevel))
    return {"path": ".git", "kind": "dir", "git_dir": git_dir}


def _walk_tree(root: str, stage: str, code: str, skip_root_git: bool = False,
               skip_dirs=()) -> tuple:
    """`(entries, directories)` below `root`: entries are `[rel, kind, mode, size_or_target,
    sha256]` for regular files and symlinks, sorted bytewise; directories are the relative
    paths of every directory walked. Special files, escaping symlinks, `.git` anywhere (except
    the already-admitted root one) and symlinked directories are rejected.
    """
    root = os.path.abspath(root)
    skip_real = {os.path.realpath(d) for d in skip_dirs}
    entries = []
    directories = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            it = os.scandir(current)
        except OSError as exc:
            raise _err(stage, code, "cannot read %s: %s" % (current, exc)) from None
        with it:
            children = sorted(it, key=lambda e: e.name.encode("utf-8", "surrogateescape"))
        for entry in children:
            full = os.path.join(current, entry.name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise _err(stage, code, "cannot lstat %s: %s" % (rel, exc)) from None
            if entry.name == ".git":
                if current == root and skip_root_git and stat.S_ISDIR(st.st_mode):
                    continue
                raise _err(stage, code, "Git indirection %s rejected (nested repository, "
                           "gitfile or worktree)" % rel)
            if stat.S_ISLNK(st.st_mode):
                target = os.readlink(full)
                base = target if os.path.isabs(target) else os.path.join(current, target)
                if not _under(os.path.normpath(base), root):
                    raise _err(stage, code, "symlink %s -> %s escapes the root" % (rel, target))
                entries.append([rel, "symlink", "120000", target,
                                pc.sha256_hex(target.encode("utf-8", "surrogateescape"))])
            elif stat.S_ISDIR(st.st_mode):
                if os.path.realpath(full) in skip_real:
                    continue
                directories.append(rel)
                stack.append(full)
            elif stat.S_ISREG(st.st_mode):
                with open(full, "rb") as f:
                    digest, size = _sha_stream(f)
                if size != st.st_size:
                    raise _err(stage, code, "%s changed size while being hashed" % rel)
                entries.append([rel, "file", _git_mode(st.st_mode), size, digest])
            else:
                raise _err(stage, code, "special file %s rejected" % rel)
    entries.sort(key=_bytes_key)
    directories.sort(key=lambda p: p.encode("utf-8", "surrogateescape"))
    return entries, directories


def _implied_dirs(paths) -> set:
    out = set()
    for rel in paths:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            out.add("/".join(parts[:i]))
    return out


def _parse_ls_tree(data: bytes, stage: str, code: str) -> dict:
    """`git ls-tree -r -z` -> {path: (mode, sha)}; any non-blob entry (gitlink) is fatal."""
    out = {}
    for record in data.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, kind, sha = meta.decode("ascii").split(" ")
        rel = path.decode("utf-8", "surrogateescape")
        if kind != "blob" or mode not in ("100644", "100755", "120000"):
            raise _err(stage, code, "tree entry %s has kind %s mode %s (gitlink/submodule or "
                       "unsupported entry rejected)" % (rel, kind, mode))
        out[rel] = (mode, sha)
    return out


def _parse_ls_files(data: bytes, stage: str, code: str) -> dict:
    """`git ls-files -s -z` -> {path: (mode, sha)}; gitlinks and unmerged stages are fatal."""
    out = {}
    for record in data.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, sha, stage_no = meta.decode("ascii").split(" ")
        rel = path.decode("utf-8", "surrogateescape")
        if mode == "160000":
            raise _err(stage, code, "index entry %s is a gitlink (submodule rejected)" % rel)
        if stage_no != "0":
            raise _err(stage, code, "index entry %s is unmerged" % rel)
        if rel in out:
            raise _err(stage, code, "duplicate index entry %s" % rel)
        out[rel] = (mode, sha)
    return out


def _compare_walk_to_tree(root: str, tree: dict, entries: list, directories: list,
                          stage: str, code: str, extra_allowed=()) -> list:
    """Every walked entry must be a tree blob with equal mode/bytes (or a listed extra);
    every tree blob must be walked; no empty directory. Returns the extras encountered."""
    walked = {e[0]: e for e in entries}
    extras = []
    for rel, entry in walked.items():
        if rel not in tree:
            if rel in extra_allowed:
                extras.append(rel)
                continue
            raise _err(stage, code, "path %s is not in the Git tree (untracked or ignored "
                       "entries are rejected)" % rel)
        mode, sha = tree[rel]
        kind = entry[1]
        if mode == "120000":
            if kind != "symlink":
                raise _err(stage, code, "%s is a %s but the tree records a symlink" % (rel, kind))
            data = entry[3].encode("utf-8", "surrogateescape")
        else:
            if kind != "file":
                raise _err(stage, code, "%s is a %s but the tree records a file" % (rel, kind))
            if entry[2] != mode:
                raise _err(stage, code, "%s has mode %s but the tree records %s" %
                           (rel, entry[2], mode))
            with open(os.path.join(root, rel), "rb") as f:
                data = f.read()
            if pc.sha256_hex(data) != entry[4]:
                raise _err(stage, code, "%s changed while being audited" % rel)
        if _git_blob_sha1(data) != sha:
            raise _err(stage, code, "%s bytes differ from tree blob %s" % (rel, sha))
    missing = sorted(set(tree) - set(walked))
    if missing:
        raise _err(stage, code, "tree entries missing from the working tree: %s" %
                   ", ".join(missing[:10]))
    implied = _implied_dirs(walked)
    empty = [d for d in directories if d not in implied]
    if empty:
        raise _err(stage, code, "directories without tracked content: %s" %
                   ", ".join(empty[:10]))
    return extras


# ---------------------------------------------------------------------------
# Source snapshot (clean) and dirty closure.
# ---------------------------------------------------------------------------

def source_snapshot(repo: str, allow_dirty: bool, external_roots=()) -> dict:
    """Clean: the independent walk must equal `HEAD` tree == index, byte for byte. Dirty
    override (`allow_dirty=True`): the closure of every regular file/symlink below the
    repository except `.git` and `external_roots`, plus HEAD/index/diff state."""
    stage, code = "source", "SourceSnapshotInvalid"
    repo = os.path.abspath(repo)
    if not os.path.isdir(repo):
        raise _err(stage, code, "%s is not a directory" % repo)
    external_roots = sorted(os.path.abspath(r) for r in external_roots)
    for ext in external_roots:
        if _under(repo, ext):
            raise _err(stage, code, "external root %s contains the repository" % ext)
    _admit_root_git(repo, stage, code)
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    tree = _git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
    if not HEX40_RE.match(commit) or not HEX40_RE.match(tree):
        raise _err(stage, "GitCommandFailed", "unexpected HEAD %r / tree %r" % (commit, tree))
    tree_entries = _parse_ls_tree(_git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD"),
                                  stage, code)
    index_entries = _parse_ls_files(_git(repo, "ls-files", "-s", "-z"), stage, code)
    if "Cargo.lock" not in tree_entries:
        raise _err(stage, "LockfileUntracked", "Cargo.lock is not in the HEAD tree")
    lock_blob = tree_entries["Cargo.lock"][1]
    # External roots (output parent, stores) are skipped only under the dirty
    # override; a clean snapshot walks everything, so an external root inside
    # the checkout would surface as an extra path.
    entries, directories = _walk_tree(repo, stage, code, skip_root_git=True,
                                      skip_dirs=(external_roots if allow_dirty else []))
    walked = {e[0]: e for e in entries}
    if "Cargo.lock" not in walked or walked["Cargo.lock"][1] != "file":
        raise _err(stage, "LockfileUntracked", "Cargo.lock is not a regular file in the tree")
    lock_sha256 = walked["Cargo.lock"][4]
    closure_sha256 = _sha_obj(entries)
    if not allow_dirty:
        if index_entries != tree_entries:
            changed = sorted(set(index_entries.items()) ^ set(tree_entries.items()))
            raise _err(stage, "DirtyWorktree", "index differs from HEAD tree at %s" %
                       ", ".join(sorted({p for p, _ in changed})[:10]))
        try:
            _compare_walk_to_tree(repo, tree_entries, entries, directories, stage, code)
        except pc.RunnerError as exc:
            if exc.error_code == code:
                raise _err(stage, "DirtyWorktree", exc.message) from None
            raise
        identity = {"commit": commit, "tree": tree, "lock_blob": lock_blob,
                    "lock_sha256": lock_sha256, "closure": entries}
        return {
            "commit": commit, "tree": tree, "lock_blob": lock_blob, "lock_sha256": lock_sha256,
            "closure": entries, "closure_sha256": closure_sha256,
            "source_snapshot_id": _sha_obj(identity), "dirty": False,
            "dirty_closure_sha256": None, "head_index_diff_sha256": None,
            "index_worktree_diff_sha256": None, "index_sha256": None, "external_roots": [],
        }
    index_bytes = _git(repo, "ls-files", "-s", "-z")
    head_index = _git(repo, "diff-index", "--cached", "-z", "HEAD")
    index_worktree = _git(repo, "diff-files", "-z")
    identity = {"commit": commit, "tree": tree, "lock_blob": lock_blob,
                "lock_sha256": lock_sha256, "closure": entries, "dirty": True,
                "index_sha256": pc.sha256_hex(index_bytes),
                "head_index_diff_sha256": pc.sha256_hex(head_index),
                "index_worktree_diff_sha256": pc.sha256_hex(index_worktree)}
    return {
        "commit": commit, "tree": tree, "lock_blob": lock_blob, "lock_sha256": lock_sha256,
        "closure": entries, "closure_sha256": closure_sha256,
        "source_snapshot_id": _sha_obj(identity), "dirty": True,
        "dirty_closure_sha256": closure_sha256,
        "head_index_diff_sha256": identity["head_index_diff_sha256"],
        "index_worktree_diff_sha256": identity["index_worktree_diff_sha256"],
        "index_sha256": identity["index_sha256"], "external_roots": external_roots,
    }


def recheck_source_snapshot(repo: str, snapshot: dict) -> None:
    """Recompute the entire snapshot and require every recorded value to be unchanged."""
    try:
        now = source_snapshot(repo, snapshot["dirty"], snapshot.get("external_roots", ()))
    except pc.RunnerError as exc:
        if exc.error_code in ("DirtyWorktree", "SourceSnapshotInvalid", "LockfileUntracked"):
            raise _err("source", "SourceChanged", "source state changed: " + exc.message) from None
        raise
    for key in ("commit", "tree", "lock_blob", "lock_sha256", "dirty", "closure_sha256",
                "dirty_closure_sha256", "head_index_diff_sha256", "index_worktree_diff_sha256",
                "index_sha256", "source_snapshot_id"):
        if now.get(key) != snapshot.get(key):
            detail = ""
            if key == "closure_sha256":
                before = {e[0]: e for e in snapshot["closure"]}
                after = {e[0]: e for e in now["closure"]}
                changed = sorted(set(before) ^ set(after)) or sorted(
                    p for p in before if before[p] != after.get(p))
                detail = " (paths: %s)" % ", ".join(changed[:10])
            raise _err("source", "SourceChanged", "source snapshot %s changed%s" % (key, detail))


# ---------------------------------------------------------------------------
# Inherited environment, toolchain, native tools, closed environment, projection.
# ---------------------------------------------------------------------------

def _rejected_env_name(name: str) -> bool:
    if name in REJECTED_ENV_EXACT or name.startswith("CARGO_PROFILE_"):
        return True
    if name.startswith("CARGO_BUILD_") and "RUSTC" in name:
        return True
    if name.startswith("CARGO_TARGET_") and name.endswith(("_RUSTFLAGS", "_RUNNER", "_LINKER")):
        return True
    return False


def reject_inherited_environment(env) -> dict:
    """Fatal on any rejected inherited name; returns the rejected-input audit otherwise."""
    hits = sorted(n for n in env if _rejected_env_name(n))
    if hits:
        raise _err("env", "RejectedEnvironment",
                   "rejected inherited environment names: %s" % ", ".join(hits))
    return {"schema": REJECTED_INPUTS_SCHEMA, "checked_names": list(REJECTED_ENV_RULES),
            "inherited_name_count": len(env), "rejected": []}


def _launcher_record(path: str, stage: str, code: str) -> dict:
    """A PATH launcher: its own kind, readlink chain (never dereferenced for identity) and
    the bytes hash of the regular file finally reached."""
    chain = []
    current = path
    seen = set()
    while True:
        try:
            st = os.lstat(current)
        except OSError as exc:
            raise _err(stage, code, "launcher %s: %s" % (current, exc)) from None
        if stat.S_ISLNK(st.st_mode):
            target = os.readlink(current)
            chain.append(target)
            if current in seen or len(chain) > 32:
                raise _err(stage, code, "symlink loop at %s" % current)
            seen.add(current)
            current = target if os.path.isabs(target) else os.path.join(
                os.path.dirname(current), target)
            continue
        if not stat.S_ISREG(st.st_mode):
            raise _err(stage, code, "launcher %s is not a regular file" % current)
        break
    resolved = os.path.normpath(current)
    resolved_sha = pc.sha256_file(resolved)
    return {"path": path, "kind": "symlink" if chain else "file", "link_chain": chain,
            "symlink_target": chain[0] if chain else None,
            "sha256": None if chain else resolved_sha, "resolved_path": resolved,
            "resolved_sha256": resolved_sha, "resolved_size": os.path.getsize(resolved)}


def _find_on_path(name: str, path_value: str, stage: str, code: str) -> str:
    for directory in path_value.split(":"):
        if not directory:
            continue
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise _err(stage, code, "no executable %s on PATH" % name)


def _sysroot_closure(sysroot: str) -> list:
    stage, code = "toolchain", "ToolchainMismatch"
    closure = []
    for rel in ("bin/cargo", "bin/rustc"):
        full = os.path.join(sysroot, rel)
        st = os.lstat(full)
        if not stat.S_ISREG(st.st_mode):
            raise _err(stage, code, "%s is not a regular file" % full)
        closure.append([rel, "file", st.st_size, pc.sha256_file(full)])
    lib = os.path.join(sysroot, "lib")
    if os.path.islink(lib) or not os.path.isdir(lib):
        raise _err(stage, code, "%s is not a plain directory" % lib)
    entries, _ = _walk_tree(lib, stage, code)
    for rel, kind, _mode, size_or_target, sha in entries:
        closure.append(["lib/" + rel, kind, size_or_target, sha])
    closure.sort(key=_bytes_key)
    return closure


def resolve_toolchain(env, timing_schema: dict, cwd: str | None = None) -> dict:
    """Resolve the PATH launchers without dereferencing their final symlinks, ask the rustc
    launcher for its sysroot, and bind `<sysroot>/bin/{cargo,rustc}` as the executables.

    `cwd` (added keyword) is where the launchers are asked (rustup honours directory
    overrides such as `rust-toolchain.toml`); it defaults to the current directory.
    """
    stage, code = "toolchain", "ToolchainMismatch"
    cwd = os.getcwd() if cwd is None else cwd
    pinned = timing_schema["toolchain"]
    path_value = env.get("PATH")
    if not path_value:
        raise _err(stage, "ToolchainUnavailable", "PATH is empty")
    cargo_launcher = _launcher_record(_find_on_path("cargo", path_value, stage,
                                                    "ToolchainUnavailable"), stage, code)
    rustc_launcher = _launcher_record(_find_on_path("rustc", path_value, stage,
                                                    "ToolchainUnavailable"), stage, code)
    launcher_env = dict(env)
    sysroot = _run_ok([rustc_launcher["path"], "--print", "sysroot"], cwd, launcher_env,
                      stage, "ToolchainUnavailable", "rustc --print sysroot").decode(
        "utf-8", "replace").strip()
    if not os.path.isabs(sysroot) or not os.path.isdir(sysroot):
        raise _err(stage, code, "rustc sysroot %r is not an absolute directory" % sysroot)
    cargo_path = os.path.join(sysroot, "bin", "cargo")
    rustc_path = os.path.join(sysroot, "bin", "rustc")
    for p in (cargo_path, rustc_path):
        st = os.lstat(p) if os.path.lexists(p) else None
        if st is None or not stat.S_ISREG(st.st_mode):
            raise _err(stage, code, "%s is not a regular file" % p)
    tool_env = {"PATH": "%s:/usr/bin:/bin" % os.path.join(sysroot, "bin"), "LANG": "C",
                "LC_ALL": "C"}
    for keep in ("HOME",):
        if keep in env:
            tool_env[keep] = env[keep]
    cargo_version = _run_ok([cargo_path, "-V"], cwd, tool_env, stage, "ToolchainUnavailable",
                            "cargo -V").decode("utf-8", "replace").strip()
    if cargo_version != pinned["cargo_version"]:
        raise _err(stage, code, "cargo -V is %r, the accepted identity is %r" %
                   (cargo_version, pinned["cargo_version"]))
    launcher_cargo_version = _run_ok([cargo_launcher["path"], "-V"], cwd, launcher_env, stage,
                                     "ToolchainUnavailable", "launcher cargo -V").decode(
        "utf-8", "replace").strip()
    if launcher_cargo_version != cargo_version:
        raise _err(stage, code, "the cargo launcher selects %r but the sysroot cargo is %r" %
                   (launcher_cargo_version, cargo_version))
    rustc_vv = _run_ok([rustc_path, "-vV"], cwd, tool_env, stage, "ToolchainUnavailable",
                       "rustc -vV").decode("utf-8", "replace").strip()
    launcher_rustc_vv = _run_ok([rustc_launcher["path"], "-vV"], cwd, launcher_env, stage,
                                "ToolchainUnavailable", "launcher rustc -vV").decode(
        "utf-8", "replace").strip()
    if launcher_rustc_vv != rustc_vv:
        raise _err(stage, code, "the rustc launcher and the sysroot rustc differ")
    fields = {}
    for line in rustc_vv.split("\n")[1:]:
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    host = fields.get("host")
    if not host:
        raise _err(stage, code, "rustc -vV has no host line")
    if host != pinned["host_triple"]:
        raise _err(stage, code, "host %r is not the pinned %r" % (host, pinned["host_triple"]))
    rustc_version = rustc_vv.split("\n")[0]
    if pinned.get("rustc_version") and rustc_version != pinned["rustc_version"]:
        raise _err(stage, code, "rustc is %r, the accepted identity is %r" %
                   (rustc_version, pinned["rustc_version"]))
    closure = _sysroot_closure(sysroot)
    return {
        "cargo_launcher": cargo_launcher, "rustc_launcher": rustc_launcher,
        "sysroot": sysroot, "cargo_path": cargo_path, "cargo_sha256": pc.sha256_file(cargo_path),
        "cargo_version": cargo_version, "rustc_path": rustc_path,
        "rustc_sha256": pc.sha256_file(rustc_path), "rustc_vV": rustc_vv,
        "rustc_version": rustc_version, "host_triple": host,
        "toolchain_closure": closure, "toolchain_closure_sha256": _sha_obj(closure),
        "wrapper_chain": [],
    }


def recheck_toolchain(toolchain: dict) -> str:
    """Recompute the sysroot closure and both binary hashes; raises on any change."""
    closure = _sysroot_closure(toolchain["sysroot"])
    digest = _sha_obj(closure)
    if digest != toolchain["toolchain_closure_sha256"]:
        before = {e[0]: e for e in toolchain["toolchain_closure"]}
        after = {e[0]: e for e in closure}
        changed = sorted(set(before) ^ set(after)) or sorted(
            p for p in before if before[p] != after.get(p))
        raise _err("toolchain", "ToolchainChanged", "toolchain closure changed at %s" %
                   ", ".join(changed[:10]))
    for launcher in (toolchain["cargo_launcher"], toolchain["rustc_launcher"]):
        now = _launcher_record(launcher["path"], "toolchain", "ToolchainChanged")
        if now != launcher:
            raise _err("toolchain", "ToolchainChanged", "launcher %s changed" % launcher["path"])
    return digest


def _probe(args, env, what) -> str:
    return _run_ok(args, "/", env, "toolchain", "NativeToolInvalid", what).decode(
        "utf-8", "replace").strip()


def _token_after(tokens: list, flag: str, count: int):
    """The `count` tokens following `flag` (None when absent): deterministic linker facts."""
    if flag not in tokens:
        return None
    i = tokens.index(flag) + 1
    return tokens[i:i + count]


def _native_closure(record: dict) -> str:
    return _sha_obj({k: v for k, v in record.items() if k != "native_closure_sha256"})


def _collect_native(timing_schema: dict) -> dict:
    stage, code = "toolchain", "NativeToolInvalid"
    if platform.system() != "Darwin":
        raise _err(stage, "UnsupportedPlatform",
                   "native tool resolution is implemented for Darwin only (got %s)" %
                   platform.system())
    probe_env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
    launchers = {}
    xcrun_find = {}
    for name in ("cc", "c++", "ar", "ranlib"):
        path = "/usr/bin/" + name
        launchers[name] = _launcher_record(path, stage, code)
        found = _probe(["xcrun", "--find", name], probe_env, "xcrun --find " + name)
        if not os.path.isabs(found) or not os.path.isfile(found):
            raise _err(stage, code, "xcrun --find %s -> %r is not a file" % (name, found))
        xcrun_find[name] = _launcher_record(found, stage, code)
    developer_dir = _probe(["xcode-select", "-p"], probe_env, "xcode-select -p")
    sdk_version = _probe(["xcrun", "--show-sdk-version"], probe_env, "xcrun --show-sdk-version")
    sdk_path = _probe(["xcrun", "--show-sdk-path"], probe_env, "xcrun --show-sdk-path")
    if not os.path.isabs(sdk_path) or not os.path.isdir(sdk_path):
        raise _err(stage, code, "SDK path %r is not a directory" % sdk_path)
    sdk_chain = []
    current = sdk_path
    while os.path.islink(current):
        target = os.readlink(current)
        sdk_chain.append(target)
        current = target if os.path.isabs(target) else os.path.join(os.path.dirname(current),
                                                                    target)
        if len(sdk_chain) > 32:
            raise _err(stage, code, "SDK symlink loop")
    settings = os.path.join(sdk_path, "SDKSettings.json")
    settings_sha = pc.sha256_file(settings) if os.path.isfile(settings) else None
    cc_version = _probe([launchers["cc"]["path"], "--version"], probe_env,
                        "cc --version").split("\n")[0]
    cxx_version = _probe([launchers["c++"]["path"], "--version"], probe_env,
                         "c++ --version").split("\n")[0]
    trace = _run([launchers["cc"]["path"], "-###", "-x", "c", "/dev/null", "-o", "/dev/null"],
                 "/", probe_env, stage, code, "cc -###")
    if trace.returncode != 0:
        raise _err(stage, code, "cc -### exited %d" % trace.returncode)
    ld_lines = []
    for line in trace.stderr.decode("utf-8", "replace").split("\n"):
        line = line.strip()
        if not line.startswith('"'):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if tokens and os.path.basename(tokens[0]) == "ld":
            ld_lines.append(tokens)
    if len(ld_lines) != 1:
        raise _err(stage, code, "expected exactly one ld invocation in the cc -### trace, "
                   "found %d" % len(ld_lines))
    ld_path = ld_lines[0][0]
    if not os.path.isabs(ld_path) or not os.path.isfile(ld_path):
        raise _err(stage, code, "linker %r is not a regular file" % ld_path)
    record = {
        "platform": "Darwin",
        "launchers": launchers, "xcrun_find": xcrun_find,
        "developer_dir": developer_dir,
        "sdk": {"version": sdk_version, "path": sdk_path, "link_chain": sdk_chain,
                "realpath": os.path.realpath(sdk_path), "settings_sha256": settings_sha},
        "cc_version": cc_version, "cxx_version": cxx_version,
        "ld": {"path": ld_path, "sha256": pc.sha256_file(ld_path),
               "syslibroot": _token_after(ld_lines[0], "-syslibroot", 1),
               "platform_version": _token_after(ld_lines[0], "-platform_version", 3)},
    }
    record["native_closure_sha256"] = _native_closure(record)
    return record


def resolve_native_tools(toolchain: dict, timing_schema: dict) -> dict:
    """Darwin: the four `/usr/bin` launchers (no final-symlink dereference), xcrun/CLT
    resolution, developer dir, SDK version/path, the linker from `cc -###`, closure hash, and
    the names/values to inject (`CC`, `CXX`, `AR`, `RANLIB`, the target linker variable)."""
    record = _collect_native(timing_schema)
    host = toolchain["host_triple"]
    linker_var = "CARGO_TARGET_%s_LINKER" % host.upper().replace("-", "_")
    launchers = record["launchers"]
    out = {
        "cc": launchers["cc"]["path"], "cxx": launchers["c++"]["path"],
        "ar": launchers["ar"]["path"], "ranlib": launchers["ranlib"]["path"],
        "linker_var_name": linker_var, "linker": launchers["cc"]["path"],
        "env": {"CC": launchers["cc"]["path"], "CXX": launchers["c++"]["path"],
                "AR": launchers["ar"]["path"], "RANLIB": launchers["ranlib"]["path"],
                linker_var: launchers["cc"]["path"]},
    }
    out.update(record)
    return out


def recheck_native_tools(native: dict) -> str:
    """Re-resolve every launcher/dispatcher/SDK fact and require the same closure hash."""
    record = _collect_native({})
    if record["native_closure_sha256"] != native["native_closure_sha256"]:
        raise _err("toolchain", "NativeClosureChanged", "native tool/SDK closure changed")
    return record["native_closure_sha256"]


def closed_environment(role: dict, toolchain: dict, native: dict, cargo_home: str, home: str,
                       target_dir: str, temp_dir: str, threads: int, timing_schema: dict) -> dict:
    """A fresh mapping with exactly the allowlisted names (nothing inherited)."""
    stage, code = "env", "RejectedEnvironment"
    for name, value in (("cargo_home", cargo_home), ("home", home), ("target_dir", target_dir),
                        ("temp_dir", temp_dir)):
        if not isinstance(value, str) or not os.path.isabs(value):
            raise _err(stage, code, "%s must be an absolute path (got %r)" % (name, value))
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise _err(stage, code, "threads must be a positive integer (got %r)" % (threads,))
    sysroot = toolchain["sysroot"]
    allow = timing_schema["closed_environment"]["allowlist"]
    linker_var = native["linker_var_name"]
    env = {
        "HOME": home,
        "PATH": "%s:/usr/bin:/bin" % os.path.join(sysroot, "bin"),
        "CARGO_HOME": cargo_home, "TMPDIR": temp_dir,
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "NO_COLOR": "1", "CARGO_TERM_COLOR": "never",
        "RUSTC": toolchain["rustc_path"], "RUSTFLAGS": allow["RUSTFLAGS"],
        "CARGO_TARGET_DIR": target_dir, linker_var: native["linker"],
        "CC": native["cc"], "CXX": native["cxx"], "AR": native["ar"], "RANLIB": native["ranlib"],
        "RAYON_NUM_THREADS": str(threads), "CC_ENABLE_DEBUG_OUTPUT": "1",
        "CARGO_NET_OFFLINE": "true", "PYTHONDONTWRITEBYTECODE": "1",
    }
    template = timing_schema["closed_environment"]["linker_variable_template"]
    expected_names = {template if n == template else n for n in allow}
    expected_names.discard(template)
    expected_names.add(linker_var)
    if set(env) != expected_names:
        raise _err(stage, code, "closed environment names %s differ from the allowlist %s" %
                   (sorted(env), sorted(expected_names)))
    for name, fixed in allow.items():
        if name == template or fixed.startswith("<"):
            continue
        if env[name] != fixed:
            raise _err(stage, code, "%s must be %r" % (name, fixed))
    if env["RUSTC"] != os.path.join(sysroot, "bin", "rustc"):
        raise _err(stage, code, "RUSTC must be <SYSROOT>/bin/rustc")
    return env


def _projection_roots(roots: dict) -> list:
    pairs = []
    for key, raw in roots.items():
        token = key if key.startswith("<") else "<%s>" % key
        if token not in PROJECTION_TOKENS:
            raise _err("config", "IdentityProjectionInvalid", "unknown projection token %s"
                       % token)
        if raw is None:
            continue
        if not isinstance(raw, str) or not os.path.isabs(raw):
            raise _err("config", "IdentityProjectionInvalid",
                       "root for %s must be an absolute path (got %r)" % (token, raw))
        candidates = {os.path.normpath(raw), os.path.realpath(raw)}
        for cand in candidates:
            pairs.append((cand.rstrip("/") or "/", token))
    order = {t: i for i, t in enumerate(PROJECTION_TOKENS)}
    pairs.sort(key=lambda p: (-len(p[0]), order[p[1]], p[0]))
    return pairs


_PRE_BOUNDARY = " \t:=,;'\"()[]"
_POST_BOUNDARY = " \t/:;,'\"#)]"


def project_string(value: str, roots: dict) -> str:
    """Replace every audited root prefix in `value` (longest first) by its token."""
    pairs = _projection_roots(roots)
    out = value
    for raw, token in pairs:
        pos = 0
        result = []
        while True:
            idx = out.find(raw, pos)
            if idx < 0:
                result.append(out[pos:])
                break
            end = idx + len(raw)
            before_ok = idx == 0 or out[idx - 1] in _PRE_BOUNDARY or out[:idx].endswith("://")
            after_ok = end == len(out) or out[end] in _POST_BOUNDARY
            if before_ok and after_ok:
                result.append(out[pos:idx])
                result.append(token)
                pos = end
            else:
                result.append(out[pos:end])
                pos = end
        out = "".join(result)
    return out


def project_value(value, roots: dict):
    if isinstance(value, str):
        return project_string(value, roots)
    if isinstance(value, list):
        return [project_value(v, roots) for v in value]
    if isinstance(value, dict):
        return {k: project_value(v, roots) for k, v in value.items()}
    return value


def identity_projection(env: dict, roots: dict) -> dict:
    """The raw closed environment (or any JSON-like mapping) with root prefixes replaced by
    their typed tokens; only the projected mapping may enter configs and identities."""
    return {name: project_value(value, roots) for name, value in env.items()}


def assert_no_raw_locators(value, roots: dict, what: str) -> None:
    """Fail closed when any string still names an audited raw root (or a file:// path)."""
    pairs = _projection_roots(roots)

    def visit(v, path):
        if isinstance(v, str):
            for raw, token in pairs:
                if raw in v:
                    raise _err("config", "IdentityProjectionInvalid",
                               "%s contains the raw %s locator at %s" % (what, token, path))
            if "file://" in v and "file://<" not in v:
                raise _err("config", "IdentityProjectionInvalid",
                           "%s contains a raw file:// locator at %s" % (what, path))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                visit(item, "%s[%d]" % (path, i))
        elif isinstance(v, dict):
            for k, item in v.items():
                visit(item, "%s.%s" % (path, k))
    visit(value, "$")


# ---------------------------------------------------------------------------
# Cargo configuration discovery and the pinned bench profile block.
# ---------------------------------------------------------------------------

def _config_candidate_paths(invocation_dir: str, cargo_home: str) -> list:
    dirs = []
    d = os.path.abspath(invocation_dir)
    while True:
        dirs.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    out = []
    seen = set()
    for base in dirs:
        for name in ("config.toml", "config"):
            path = os.path.join(base, ".cargo", name)
            seen.add(os.path.realpath(path))
            out.append(path)
    for name in ("config.toml", "config"):
        path = os.path.join(os.path.abspath(cargo_home), name)
        if os.path.realpath(path) not in seen:
            out.append(path)
    return out


def _flatten_toml(obj, prefix="") -> dict:
    out = {}
    for key, value in obj.items():
        dotted = prefix + key if not prefix else "%s.%s" % (prefix, key)
        if isinstance(value, dict):
            out.update(_flatten_toml(value, dotted))
        else:
            out[dotted] = value
    return out


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)}


def _audit_config_candidate(path: str, allowed_keys, forbidden_tables) -> dict:
    stage, code = "toolchain", "CargoConfigRejected"
    record = {"path": path, "exists": False, "kind": None, "is_symlink": False, "size": None,
              "sha256": None, "keys": {}}
    if not os.path.lexists(path):
        return record
    st = os.lstat(path)
    record["exists"] = True
    if stat.S_ISLNK(st.st_mode):
        record["is_symlink"] = True
        raise _err(stage, code, "Cargo config candidate %s is a symlink" % path)
    if stat.S_ISDIR(st.st_mode):
        record["kind"] = "dir"
        raise _err(stage, code, "Cargo config candidate %s is a directory" % path)
    if not stat.S_ISREG(st.st_mode):
        raise _err(stage, code, "Cargo config candidate %s is a special file" % path)
    record["kind"] = "file"
    with open(path, "rb") as f:
        data = f.read()
    record["size"] = len(data)
    record["sha256"] = pc.sha256_hex(data)
    try:
        doc = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _err(stage, code, "Cargo config %s does not parse: %s" % (path, exc)) from None
    for table in forbidden_tables:
        if table in doc:
            raise _err(stage, code, "Cargo config %s sets the forbidden table [%s]" %
                       (path, table))
    flat = _flatten_toml(doc)
    for key in sorted(flat):
        if key not in allowed_keys:
            raise _err(stage, code, "Cargo config %s sets %s, which is outside the allowlist"
                       % (path, key))
    record["keys"] = {k: _json_safe(v) for k, v in flat.items()}
    return record


def _profile_block(manifest_text: str) -> str:
    """The exact `[profile.bench]` block text: header line through the last non-blank line
    before the next table header, trailing blank lines removed, one trailing LF."""
    lines = manifest_text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.strip() == "[profile.bench]"]
    if len(starts) != 1:
        raise _err("toolchain", "CargoConfigRejected",
                   "Cargo.toml must contain exactly one [profile.bench] header (found %d)"
                   % len(starts))
    start = starts[0]
    end = start + 1
    while end < len(lines) and not lines[end].lstrip().startswith("["):
        end += 1
    block = lines[start:end]
    while block and block[-1].strip() == "":
        block.pop()
    return "\n".join(block) + "\n"


def _audit_manifest_profile(manifest_path: str, timing_schema: dict) -> dict:
    stage, code = "toolchain", "CargoConfigRejected"
    pinned = timing_schema["bench_profile"]
    try:
        with open(manifest_path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise _err(stage, code, "cannot read %s: %s" % (manifest_path, exc)) from None
    if os.path.islink(manifest_path):
        raise _err(stage, code, "%s is a symlink" % manifest_path)
    try:
        text = data.decode("utf-8")
        doc = tomllib.loads(text)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _err(stage, code, "%s does not parse: %s" % (manifest_path, exc)) from None
    bench = doc.get("profile", {}).get("bench")
    if not isinstance(bench, dict):
        raise _err(stage, code, "%s has no [profile.bench] table" % manifest_path)
    if "package" in bench or any(isinstance(v, dict) for v in bench.values()):
        raise _err(stage, code, "[profile.bench.package.*] / nested overrides are rejected")
    block = _profile_block(text)
    if block != pinned["manifest_block"]:
        raise _err(stage, code, "[profile.bench] block differs from the pinned block:\n%s"
                   % block)
    expected = {"opt-level": pinned["opt_level"], "debug": pinned["debug"],
                "strip": pinned["strip"], "debug-assertions": pinned["debug_assertions"],
                "overflow-checks": pinned["overflow_checks"], "lto": pinned["lto"],
                "codegen-units": pinned["codegen_units"], "incremental": pinned["incremental"]}
    if bench != expected:
        raise _err(stage, code, "[profile.bench] values %r differ from %r" % (bench, expected))
    return {"path": manifest_path, "sha256": pc.sha256_hex(data), "profile_block": block,
            "profile_block_sha256": pc.sha256_hex(block.encode("utf-8"))}


def cargo_config_audit(invocation_dir: str, cargo_home: str, timing_schema: dict,
                       repo: str | None = None, rustflags: str | None = None) -> dict:
    """Cargo's discovery order (invocation dir, every ancestor, then `CARGO_HOME`), the
    `term.*` allowlist, and the pinned `[profile.bench]` block of `<repo>/Cargo.toml`
    (`repo` defaults to `invocation_dir`). `rustflags` defaults to the pinned canonical value.
    """
    stage, code = "toolchain", "CargoConfigRejected"
    allow = timing_schema["cargo_config_allowlist"]
    allowed_keys = set(allow["allowed_keys"])
    forbidden = list(allow["forbidden_tables"])
    invocation_dir = os.path.abspath(invocation_dir)
    cargo_home = os.path.abspath(cargo_home)
    repo = invocation_dir if repo is None else os.path.abspath(repo)
    candidates = []
    by_dir = {}
    for path in _config_candidate_paths(invocation_dir, cargo_home):
        record = _audit_config_candidate(path, allowed_keys, forbidden)
        candidates.append(record)
        if record["exists"]:
            by_dir.setdefault(os.path.dirname(path), []).append(path)
    for directory, paths in by_dir.items():
        if len(paths) > 1:
            raise _err(stage, code, "ambiguous Cargo config pair in %s: %s" %
                       (directory, ", ".join(os.path.basename(p) for p in paths)))
    pinned_flags = timing_schema["closed_environment"]["allowlist"]["RUSTFLAGS"]
    rustflags = pinned_flags if rustflags is None else rustflags
    if rustflags != pinned_flags:
        raise _err(stage, code, "RUSTFLAGS %r is not the canonical %r" % (rustflags, pinned_flags))
    if re.search(r"-C\s*panic", rustflags) or "-Cpanic" in rustflags:
        raise _err(stage, code, "a -C panic override in RUSTFLAGS is rejected")
    manifest = _audit_manifest_profile(os.path.join(repo, "Cargo.toml"), timing_schema)
    audit = {
        "schema": CARGO_CONFIG_AUDIT_SCHEMA,
        "python": {"version": sys.version.split()[0], "implementation":
                   platform.python_implementation(),
                   "dont_write_bytecode": bool(sys.dont_write_bytecode)},
        "invocation_dir": invocation_dir, "cargo_home": cargo_home, "repo": repo,
        "candidates": candidates,
        "discovered": [c["path"] for c in candidates if c["exists"]],
        "origins": {k: c["path"] for c in candidates if c["exists"] for k in c["keys"]},
        "allowed_keys": sorted(allowed_keys), "forbidden_tables": forbidden,
        "rustflags": rustflags, "manifest": manifest,
    }
    audit["audit_id"] = _sha_obj({k: v for k, v in audit.items() if k != "audit_id"})
    return audit


def recheck_cargo_config(audit: dict, timing_schema: dict | None = None) -> str:
    """Repeat the full discovery and require the frozen audit to be reproduced exactly."""
    schema = timing_schema
    if schema is None:
        schema = {"cargo_config_allowlist": {"allowed_keys": audit["allowed_keys"],
                                             "forbidden_tables": audit["forbidden_tables"]},
                  "closed_environment": {"allowlist": {"RUSTFLAGS": audit["rustflags"]}},
                  "bench_profile": _profile_fields_from_block(audit["manifest"]["profile_block"])}
    now = cargo_config_audit(audit["invocation_dir"], audit["cargo_home"], schema,
                             repo=audit["repo"], rustflags=audit["rustflags"])
    if now["audit_id"] != audit["audit_id"]:
        before = {c["path"]: c for c in audit["candidates"]}
        after = {c["path"]: c for c in now["candidates"]}
        changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
        if now["manifest"] != audit["manifest"]:
            changed.append(audit["manifest"]["path"])
        raise _err("toolchain", "CargoConfigChanged", "Cargo configuration changed: %s" %
                   ", ".join(changed[:10]))
    return now["audit_id"]


def _profile_fields_from_block(block: str) -> dict:
    doc = tomllib.loads(block)["profile"]["bench"]
    return {"manifest_block": block, "opt_level": doc["opt-level"], "debug": doc["debug"],
            "strip": doc["strip"], "debug_assertions": doc["debug-assertions"],
            "overflow_checks": doc["overflow-checks"], "lto": doc["lto"],
            "codegen_units": doc["codegen-units"], "incremental": doc["incremental"]}


# ---------------------------------------------------------------------------
# Dependency-source closure: lockfile, registry archives/unpacked trees, git checkouts.
# ---------------------------------------------------------------------------

_DEP_STAGE, _DEP_CODE = "dependency", "DependencySourceInvalid"


def _parse_lock(lock_bytes: bytes) -> dict:
    """`{(name, version, source_or_None): {"checksum", "dependencies"}}` from Cargo.lock."""
    try:
        doc = tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _err(_DEP_STAGE, _DEP_CODE, "Cargo.lock does not parse: %s" % exc) from None
    packages = {}
    for pkg in doc.get("package", []):
        key = (pkg.get("name"), pkg.get("version"), pkg.get("source"))
        if not key[0] or not key[1]:
            raise _err(_DEP_STAGE, _DEP_CODE, "Cargo.lock package without name/version")
        if key in packages:
            raise _err(_DEP_STAGE, _DEP_CODE, "duplicate Cargo.lock entry %s %s" % key[:2])
        packages[key] = {"checksum": pkg.get("checksum"),
                         "dependencies": list(pkg.get("dependencies", []))}
    return {"version": doc.get("version"), "packages": packages}


def _lock_dep_matches(dep: str, name: str, version: str, source) -> bool:
    parts = dep.split(" ", 2)
    if parts[0] != name:
        return False
    if len(parts) >= 2 and parts[1] != version:
        return False
    if len(parts) == 3:
        inner = parts[2]
        if not (inner.startswith("(") and inner.endswith(")")):
            return False
        if inner[1:-1] != (source or ""):
            return False
    return True


def _registry_index_dir(cargo_home: str) -> str:
    cache = os.path.join(cargo_home, "registry", "cache")
    try:
        names = sorted(n for n in os.listdir(cache) if n.startswith(INDEX_DIR_PREFIX)
                       and os.path.isdir(os.path.join(cache, n)))
    except OSError as exc:
        raise _err(_DEP_STAGE, _DEP_CODE, "cannot list %s: %s" % (cache, exc)) from None
    if len(names) != 1:
        raise _err(_DEP_STAGE, _DEP_CODE, "expected exactly one %s* directory under %s, found "
                   "%s" % (INDEX_DIR_PREFIX, cache, names))
    src = os.path.join(cargo_home, "registry", "src", names[0])
    if os.path.islink(src) or not os.path.isdir(src):
        raise _err(_DEP_STAGE, _DEP_CODE, "%s is not a plain directory" % src)
    return names[0]


def _tar_member_rel(member, prefix: str) -> str | None:
    name = member.name
    if name.startswith("/") or name.startswith("\\"):
        raise _err(_DEP_STAGE, _DEP_CODE, "archive member %r is absolute" % name)
    parts = [p for p in name.split("/")]
    if parts and parts[-1] == "" and member.isdir():
        parts = parts[:-1]
    if any(p in ("", ".", "..") for p in parts):
        raise _err(_DEP_STAGE, _DEP_CODE, "archive member %r has an empty/./.. component" % name)
    if parts[0] != prefix:
        raise _err(_DEP_STAGE, _DEP_CODE, "archive member %r is outside the package prefix %s"
                   % (name, prefix))
    return "/".join(parts[1:])


def _audit_crate_archive(archive: str, prefix: str, unpacked_root: str) -> dict:
    """Stream the `.crate`, reject unsafe members, and require the unpacked tree to equal it
    (plus the root `.cargo-ok` marker)."""
    st = os.lstat(archive) if os.path.lexists(archive) else None
    if st is None or not stat.S_ISREG(st.st_mode):
        raise _err(_DEP_STAGE, _DEP_CODE, "%s is not a regular non-symlink file" % archive)
    members = {}
    explicit_dirs = {}
    root_member = False
    try:
        with tarfile.open(archive, "r:gz", errorlevel=2) as tf:
            for m in tf:
                rel = _tar_member_rel(m, prefix)
                if rel == "":
                    if not m.isdir() or root_member:
                        raise _err(_DEP_STAGE, _DEP_CODE, "archive %s has a non-directory or "
                                   "duplicate package-root member" % archive)
                    root_member = True
                    continue
                if rel in members or rel in explicit_dirs:
                    raise _err(_DEP_STAGE, _DEP_CODE, "duplicate archive member %s" % rel)
                if m.isdir():
                    explicit_dirs[rel] = m.mode & 0o777
                elif m.isfile():
                    f = tf.extractfile(m)
                    if f is None:
                        raise _err(_DEP_STAGE, _DEP_CODE, "cannot read member %s" % rel)
                    digest, size = _sha_stream(f)
                    members[rel] = ["file", "100755" if m.mode & 0o111 else "100644", size,
                                    digest]
                elif m.issym():
                    target = m.linkname
                    joined = os.path.normpath(os.path.join(os.path.dirname(rel), target))
                    if os.path.isabs(target) or joined == ".." or joined.startswith("../"):
                        raise _err(_DEP_STAGE, _DEP_CODE, "symlink member %s -> %s escapes the "
                                   "package root" % (rel, target))
                    members[rel] = ["symlink", "120000", target,
                                    pc.sha256_hex(target.encode("utf-8"))]
                else:
                    raise _err(_DEP_STAGE, _DEP_CODE, "archive member %s has unsupported type "
                               "%r (hard link/special member rejected)" % (rel, m.type))
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise _err(_DEP_STAGE, _DEP_CODE, "cannot read archive %s: %s" % (archive, exc)) from None
    expected_dirs = set(explicit_dirs) | _implied_dirs(list(members) + list(explicit_dirs))
    if os.path.islink(unpacked_root) or not os.path.isdir(unpacked_root):
        raise _err(_DEP_STAGE, _DEP_CODE, "%s is not a plain directory" % unpacked_root)
    entries, directories = _walk_tree(unpacked_root, _DEP_STAGE, _DEP_CODE)
    if set(directories) != expected_dirs:
        diff = sorted(set(directories) ^ expected_dirs)
        raise _err(_DEP_STAGE, _DEP_CODE, "unpacked directory set of %s differs from the "
                   "archive: %s" % (unpacked_root, ", ".join(diff[:10])))
    recorded_dirs = []
    for rel in sorted(expected_dirs):
        mode = os.lstat(os.path.join(unpacked_root, rel)).st_mode & 0o777
        if rel in explicit_dirs and mode != explicit_dirs[rel]:
            raise _err(_DEP_STAGE, _DEP_CODE, "directory %s has mode %o, archive says %o" %
                       (rel, mode, explicit_dirs[rel]))
        recorded_dirs.append([rel, "%o" % mode, rel in explicit_dirs])
    marker = None
    on_disk = {e[0]: e for e in entries}
    for rel, entry in on_disk.items():
        if rel in members:
            kind, mode, size_or_target, digest = members[rel]
            if entry[1] != kind or entry[2] != mode or entry[3] != size_or_target or \
                    entry[4] != digest:
                raise _err(_DEP_STAGE, _DEP_CODE, "unpacked %s/%s differs from the archive "
                           "member" % (prefix, rel))
            continue
        if rel == ".cargo-ok" and entry[1] == "file" and entry[3] == len(REGISTRY_MARKER_BYTES) \
                and entry[4] == REGISTRY_MARKER_SHA256:
            marker = {"path": rel, "kind": "file", "mode": entry[2], "size": entry[3],
                      "sha256": entry[4]}
            continue
        raise _err(_DEP_STAGE, _DEP_CODE, "extra unpacked entry %s/%s" % (prefix, rel))
    missing = sorted(set(members) - set(on_disk))
    if missing:
        raise _err(_DEP_STAGE, _DEP_CODE, "archive members missing from %s: %s" %
                   (unpacked_root, ", ".join(missing[:10])))
    if marker is None:
        raise _err(_DEP_STAGE, _DEP_CODE, "%s lacks the .cargo-ok marker" % unpacked_root)
    return {"entries": entries, "directories": recorded_dirs, "marker": marker,
            "root_member": root_member, "member_count": len(members)}


def _audit_git_checkout(root: str, commit: str) -> dict:
    stage, code = _DEP_STAGE, _DEP_CODE
    if os.path.islink(root) or not os.path.isdir(root):
        raise _err(stage, code, "git checkout %s is not a plain directory" % root)
    _admit_root_git(root, stage, code)
    head = _git(root, "rev-parse", "HEAD", stage=stage).decode().strip()
    if head != commit:
        raise _err(stage, code, "checkout %s is at %s, the lock selects %s" % (root, head, commit))
    tree = _git(root, "rev-parse", "HEAD^{tree}", stage=stage).decode().strip()
    tree_entries = _parse_ls_tree(_git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD",
                                       stage=stage), stage, code)
    index_entries = _parse_ls_files(_git(root, "ls-files", "-s", "-z", stage=stage), stage, code)
    if index_entries != tree_entries:
        raise _err(stage, code, "checkout %s index differs from its HEAD tree" % root)
    entries, directories = _walk_tree(root, stage, code, skip_root_git=True)
    extras = _compare_walk_to_tree(root, tree_entries, entries, directories, stage, code,
                                   extra_allowed=(".cargo-ok",))
    marker = None
    for entry in entries:
        if entry[0] == ".cargo-ok":
            if entry[1] != "file" or entry[3] != 0 or entry[4] != GIT_MARKER_SHA256:
                raise _err(stage, code, "%s/.cargo-ok is not the zero-byte marker" % root)
            marker = {"path": ".cargo-ok", "kind": "file", "mode": entry[2], "size": 0,
                      "sha256": entry[4]}
    if marker is None or extras != [".cargo-ok"]:
        raise _err(stage, code, "%s lacks the zero-byte .cargo-ok marker" % root)
    return {"commit": head, "tree": tree, "entries": entries, "marker": marker}


def _parse_git_source(source: str) -> dict:
    body = source[len("git+"):]
    locator, sep, commit = body.partition("#")
    if not sep or not HEX40_RE.match(commit):
        raise _err(_DEP_STAGE, _DEP_CODE, "git source %r does not name a full commit" % source)
    url, _, query = locator.partition("?")
    name = url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    if not name:
        raise _err(_DEP_STAGE, _DEP_CODE, "git source %r has no repository name" % source)
    return {"url": url, "query": query, "commit": commit, "repo_name": name, "locator": locator}


def _metadata_argv(toolchain: dict, features) -> list:
    try:
        feature_tokens = pc.bootstrap_feature_tokens(features)
    except pc.RunnerError as exc:
        raise _err(_DEP_STAGE, _DEP_CODE, "unsupported feature set %r: %s" %
                   (features, exc.message)) from None
    argv = [toolchain["cargo_path"], "metadata", "--locked", "--offline", "--format-version",
            "1", "--filter-platform", toolchain["host_triple"]]
    argv += feature_tokens
    return argv


def _check_metadata_target(path: str, repo: str, cargo_home: str, sysroot: str) -> None:
    if not os.path.isabs(path) or os.path.islink(path) or not os.path.isdir(path):
        raise _err(_DEP_STAGE, "TempDirInvalid", "%s must be an absolute non-symlink directory"
                   % path)
    if os.listdir(path):
        raise _err(_DEP_STAGE, "TempDirInvalid", "%s is not empty" % path)
    real = os.path.realpath(path)
    for other in (repo, cargo_home, sysroot):
        real_other = os.path.realpath(other)
        if _under(real, real_other) or _under(real_other, real):
            raise _err(_DEP_STAGE, "PathOverlap", "%s overlaps %s" % (path, other))


def _run_metadata(argv, repo, env, metadata_target) -> dict:
    run_env = dict(env)
    run_env["CARGO_TARGET_DIR"] = metadata_target
    proc = _run(argv, repo, run_env, _DEP_STAGE, "MetadataUnavailable", "cargo metadata")
    if proc.returncode != 0:
        raise _err(_DEP_STAGE, "MetadataUnavailable", "cargo metadata exited %d: %s" %
                   (proc.returncode, proc.stderr.decode("utf-8", "replace").strip()[-1500:]))
    try:
        obj = json.loads(proc.stdout.decode("utf-8"), parse_float=pc._reject_float)
    except (UnicodeDecodeError, ValueError) as exc:
        raise _err(_DEP_STAGE, "MetadataMalformed", "cargo metadata output: %s" % exc) from None
    if not isinstance(obj, dict) or obj.get("version") != 1:
        raise _err(_DEP_STAGE, "MetadataMalformed", "cargo metadata is not format-version 1")
    for key in ("packages", "resolve", "workspace_root", "target_directory"):
        if key not in obj:
            raise _err(_DEP_STAGE, "MetadataMalformed", "cargo metadata lacks %s" % key)
    return obj


def dependency_source_audit(repo: str, snapshot: dict, env: dict, toolchain: dict,
                            cargo_home: str, metadata_target: str, features: list,
                            timing_schema: dict) -> dict:
    """Bootstrap `cargo metadata` in the closed environment, cross-check every package and
    edge against the unchanged `Cargo.lock`, audit registry archives/unpacked trees and git
    checkouts, and build the canonical `dependency-sources.json` document."""
    stage, code = _DEP_STAGE, _DEP_CODE
    repo = os.path.abspath(repo)
    cargo_home = os.path.abspath(cargo_home)
    metadata_target = os.path.abspath(metadata_target)
    _check_metadata_target(metadata_target, repo, cargo_home, toolchain["sysroot"])
    if env.get("CARGO_NET_OFFLINE") != "true" or env.get("CARGO_HOME") != cargo_home:
        raise _err(stage, code, "the closed environment must carry CARGO_NET_OFFLINE=true and "
                   "CARGO_HOME=%s" % cargo_home)
    with open(os.path.join(repo, "Cargo.lock"), "rb") as f:
        lock_bytes = f.read()
    lock_sha256 = pc.sha256_hex(lock_bytes)
    if lock_sha256 != snapshot["lock_sha256"]:
        raise _err(stage, code, "Cargo.lock differs from the source snapshot")
    lock = _parse_lock(lock_bytes)
    argv = _metadata_argv(toolchain, features)
    metadata = _run_metadata(argv, repo, env, metadata_target)
    roots = {"SOURCE": repo, "CARGO_HOME": cargo_home, "METADATA_TARGET": metadata_target}
    ids = {}
    for pkg in metadata["packages"]:
        if pkg["id"] in ids:
            raise _err(stage, code, "duplicate package id %s" % pkg["id"])
        ids[pkg["id"]] = pkg
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    if set(nodes) != set(ids):
        raise _err(stage, code, "resolve nodes and packages differ: %s" %
                   sorted(set(nodes) ^ set(ids))[:5])
    lock_packages = lock["packages"]
    by_id_lock = {}
    for pid, pkg in ids.items():
        key = (pkg["name"], pkg["version"], pkg.get("source"))
        if key not in lock_packages:
            raise _err(stage, code, "package %s %s (%s) is not in Cargo.lock" % key)
        by_id_lock[pid] = key
    for pid, node in nodes.items():
        lock_deps = lock_packages[by_id_lock[pid]]["dependencies"]
        for dep_id in node.get("dependencies", []):
            if dep_id not in ids:
                raise _err(stage, code, "edge %s -> %s names an unknown package" % (pid, dep_id))
            name, version, source = by_id_lock[dep_id]
            if not any(_lock_dep_matches(d, name, version, source) for d in lock_deps):
                raise _err(stage, code, "edge %s -> %s is not in Cargo.lock" % (pid, dep_id))
    snapshot_paths = {e[0]: e for e in snapshot["closure"]}
    index_dir = None
    git_cache = {}
    packages = []
    raw_packages = {}
    zstd_features = None
    for pid in sorted(ids, key=lambda p: project_string(p, roots).encode()):
        pkg = ids[pid]
        name, version, source = by_id_lock[pid]
        manifest = pkg["manifest_path"]
        lock_entry = lock_packages[(name, version, source)]
        record = {"id": project_string(pid, roots), "name": name, "version": version,
                  "lock_source": source, "lock_checksum": lock_entry["checksum"],
                  "manifest_path": project_string(manifest, roots), "registry": None,
                  "git": None, "entries": []}
        if name == "zstd-sys":
            zstd_features = sorted(nodes[pid].get("features", []))
            if "pkg-config" in zstd_features:
                raise _err(stage, code, "zstd-sys resolves with the pkg-config feature")
        if source is None:
            if not pid.startswith("path+file://") or not _strict_under(manifest, repo):
                raise _err(stage, code, "path package %s is outside the repository" % pid)
            rel_manifest = os.path.relpath(manifest, repo).replace(os.sep, "/")
            if rel_manifest not in snapshot_paths:
                raise _err(stage, code, "manifest %s is not in the source snapshot" % rel_manifest)
            root = os.path.dirname(manifest)
            rel_root = os.path.relpath(root, repo).replace(os.sep, "/")
            for target in pkg.get("targets", []):
                if not _strict_under(target.get("src_path", ""), repo):
                    raise _err(stage, code, "target %s of %s is outside the repository" %
                               (target.get("name"), pid))
            prefix = "" if rel_root == "." else rel_root + "/"
            record["source_kind"] = "path"
            record["root"] = project_string(root, roots)
            record["entries"] = [[e[0][len(prefix):]] + e[1:] for e in snapshot["closure"]
                                 if e[0].startswith(prefix)]
            raw_packages[record["id"]] = {"kind": "path", "root": root, "prefix": prefix}
        elif source == CRATES_IO_SOURCE:
            checksum = lock_entry["checksum"]
            if not pc.is_hex64(checksum):
                raise _err(stage, code, "registry package %s has no lock checksum" % pid)
            if index_dir is None:
                index_dir = _registry_index_dir(cargo_home)
            dirname = "%s-%s" % (name, version)
            archive = os.path.join(cargo_home, "registry", "cache", index_dir, dirname + ".crate")
            unpacked = os.path.join(cargo_home, "registry", "src", index_dir, dirname)
            if manifest != os.path.join(unpacked, "Cargo.toml"):
                raise _err(stage, code, "registry package %s manifest %s is not %s/Cargo.toml"
                           % (pid, manifest, unpacked))
            if not os.path.lexists(archive) or os.path.islink(archive):
                raise _err(stage, code, "archive %s is missing or a symlink (offline setup "
                           "error, never a fetch)" % archive)
            archive_sha = pc.sha256_file(archive)
            if archive_sha != checksum:
                raise _err(stage, code, "archive %s has SHA-256 %s, the lock says %s" %
                           (archive, archive_sha, checksum))
            audit = _audit_crate_archive(archive, dirname, unpacked)
            record["source_kind"] = "registry"
            record["registry"] = {
                "index": index_dir, "archive": project_string(archive, roots),
                "archive_sha256": archive_sha, "archive_size": os.path.getsize(archive),
                "unpacked_root": project_string(unpacked, roots),
                "directories": audit["directories"], "marker": audit["marker"],
                "root_member": audit["root_member"], "member_count": audit["member_count"]}
            record["entries"] = audit["entries"]
            raw_packages[record["id"]] = {"kind": "registry", "archive": archive,
                                          "root": unpacked, "prefix": dirname}
        elif source.startswith("git+"):
            src = _parse_git_source(source)
            checkouts = os.path.join(cargo_home, "git", "checkouts")
            rel = os.path.relpath(manifest, checkouts).replace(os.sep, "/")
            parts = rel.split("/")
            if rel.startswith("..") or len(parts) < 3 or parts[-1] != "Cargo.toml":
                raise _err(stage, code, "git package %s manifest %s is not under %s" %
                           (pid, manifest, checkouts))
            if not parts[0].startswith(src["repo_name"] + "-"):
                raise _err(stage, code, "checkout directory %s does not belong to %s" %
                           (parts[0], src["repo_name"]))
            root = os.path.join(checkouts, parts[0], parts[1])
            if root not in git_cache:
                git_cache[root] = _audit_git_checkout(root, src["commit"])
            audit = git_cache[root]
            record["source_kind"] = "git"
            record["git"] = {"url": src["url"], "query": src["query"], "commit": src["commit"],
                             "tree": audit["tree"], "checkout_root": project_string(root, roots),
                             "package_root": project_string(os.path.dirname(manifest), roots),
                             "marker": audit["marker"]}
            record["entries"] = audit["entries"]
            raw_packages[record["id"]] = {"kind": "git", "root": root, "commit": src["commit"]}
        else:
            raise _err(stage, code, "package %s has an unsupported source %r (source "
                       "replacement or unknown kind)" % (pid, source))
        packages.append(record)
    normalized = project_value(metadata, roots)
    metadata_sha = _sha_obj(normalized)
    doc_roots = dict(roots)
    doc_roots["SYSROOT"] = toolchain["sysroot"]
    document = {
        "schema": DEPENDENCY_SOURCES_SCHEMA,
        "cargo_version": toolchain["cargo_version"], "host_triple": toolchain["host_triple"],
        "metadata_command": [project_string(a, doc_roots) for a in argv],
        "features": list(features), "lock_version": lock["version"],
        "lock_sha256": lock_sha256, "lock_package_count": len(lock_packages),
        "resolved_package_count": len(packages), "registry_index": index_dir,
        "zstd_sys_features": zstd_features, "packages": packages,
        "metadata_normalized_sha256": metadata_sha,
        "tokens": ["<SOURCE>", "<CARGO_HOME>", "<METADATA_TARGET>", "<SYSROOT>"],
    }
    scan_roots = dict(doc_roots)
    if env.get("HOME"):
        scan_roots["HOME"] = env["HOME"]
    assert_no_raw_locators(document, scan_roots, "dependency-sources.json")
    data = _canon(document)
    closure_sha = _sha_obj([[p["id"], p["entries"]] for p in packages])
    return {
        "document": document, "bytes": data, "dependency_source_id": pc.sha256_hex(data),
        "closure_sha256": closure_sha, "metadata_normalized_sha256": metadata_sha,
        "raw_locators": {"repo": repo, "cargo_home": cargo_home,
                         "metadata_target": metadata_target, "index_dir": index_dir,
                         "argv": argv, "sysroot": toolchain["sysroot"],
                         "packages": raw_packages},
    }


def rehash_dependency_closure(audit: dict) -> str:
    """Recompute every recorded entry hash (archives, unpacked trees, markers, git trees)
    and compare with the frozen document; returns the closure hash."""
    stage, code = _DEP_STAGE, "DependencyClosureChanged"
    raw = audit["raw_locators"]["packages"]
    packages = audit["document"]["packages"]
    rebuilt = []
    for record in packages:
        loc = raw.get(record["id"])
        if loc is None:
            raise _err(stage, code, "no raw locator for %s" % record["id"])
        try:
            if loc["kind"] == "path":
                skip_git = os.path.realpath(loc["root"]) == os.path.realpath(
                    audit["raw_locators"]["repo"])
                entries, _ = _walk_tree(loc["root"], stage, code, skip_root_git=skip_git)
            elif loc["kind"] == "registry":
                if pc.sha256_file(loc["archive"]) != record["registry"]["archive_sha256"]:
                    raise _err(stage, code, "archive %s changed" % loc["archive"])
                now = _audit_crate_archive(loc["archive"], loc["prefix"], loc["root"])
                if now["directories"] != record["registry"]["directories"] or \
                        now["marker"] != record["registry"]["marker"]:
                    raise _err(stage, code, "unpacked tree %s changed" % loc["root"])
                entries = now["entries"]
            else:
                now = _audit_git_checkout(loc["root"], loc["commit"])
                if now["tree"] != record["git"]["tree"] or now["marker"] != record["git"]["marker"]:
                    raise _err(stage, code, "git checkout %s changed" % loc["root"])
                entries = now["entries"]
        except pc.RunnerError as exc:
            if exc.error_code == code:
                raise
            raise _err(stage, code, exc.message) from None
        if entries != record["entries"]:
            before = {e[0]: e for e in record["entries"]}
            after = {e[0]: e for e in entries}
            changed = sorted(set(before) ^ set(after)) or sorted(
                p for p in before if before[p] != after.get(p))
            raise _err(stage, code, "dependency closure of %s changed at %s" %
                       (record["id"], ", ".join(changed[:10])))
        rebuilt.append([record["id"], entries])
    digest = _sha_obj(rebuilt)
    if digest != audit["closure_sha256"]:
        raise _err(stage, code, "dependency closure hash changed")
    return digest


def replay_metadata(audit: dict, env: dict, metadata_target: str | None = None) -> None:
    """Rerun the exact metadata command with a fresh empty target, rehashing the closure
    before and after, and require identical normalized output."""
    raw = audit["raw_locators"]
    created = metadata_target is None
    if created:
        metadata_target = tempfile.mkdtemp(prefix="poseidon-meta-replay-",
                                           dir=env.get("TMPDIR") or None)
    try:
        metadata_target = os.path.abspath(metadata_target)
        _check_metadata_target(metadata_target, raw["repo"], raw["cargo_home"], raw["sysroot"])
        rehash_dependency_closure(audit)
        metadata = _run_metadata(raw["argv"], raw["repo"], env, metadata_target)
        roots = {"SOURCE": raw["repo"], "CARGO_HOME": raw["cargo_home"],
                 "METADATA_TARGET": metadata_target}
        digest = _sha_obj(project_value(metadata, roots))
        rehash_dependency_closure(audit)
        if digest != audit["metadata_normalized_sha256"]:
            raise _err(_DEP_STAGE, "MetadataReplayMismatch",
                       "the replayed cargo metadata output differs from the frozen output")
    finally:
        if created:
            shutil.rmtree(metadata_target, ignore_errors=True)


# ---------------------------------------------------------------------------
# Evidence build: Cargo JSON messages, the bench rustc argv grammar, native tool lines.
# ---------------------------------------------------------------------------

_EV_STAGE = "evidence"


def _parse_running_records(text: str) -> list:
    """Every `     Running \\`...\\`` record of a `-vv` log (values may span lines and contain
    backticks inside single quotes)."""
    records = []
    pos = 0
    while True:
        idx = text.find(RUNNING_PREFIX, pos)
        if idx < 0:
            break
        if idx > 0 and text[idx - 1] != "\n":
            pos = idx + len(RUNNING_PREFIX)
            continue
        i = idx + len(RUNNING_PREFIX)
        quoted = False
        end = -1
        while i < len(text):
            ch = text[i]
            if quoted:
                if ch == "'":
                    quoted = False
            elif ch == "'":
                quoted = True
            elif ch == "\\":
                i += 1
            elif ch == "`":
                end = i
                break
            i += 1
        if end < 0:
            raise _err(_EV_STAGE, "EvidenceBuildFailed", "unterminated Running record in the log")
        inner = text[idx + len(RUNNING_PREFIX):end]
        try:
            tokens = shlex.split(inner)
        except ValueError as exc:
            raise _err(_EV_STAGE, "EvidenceBuildFailed", "cannot parse Running record: %s" %
                       exc) from None
        env = {}
        k = 0
        while k < len(tokens) and ENV_TOKEN_RE.match(tokens[k]):
            name, _, value = tokens[k].partition("=")
            env[name] = value
            k += 1
        if k >= len(tokens):
            raise _err(_EV_STAGE, "EvidenceBuildFailed", "Running record without a program")
        records.append({"env": env, "program": tokens[k], "args": tokens[k + 1:],
                        "raw_sha256": pc.sha256_hex(inner.encode("utf-8", "surrogateescape"))})
        pos = end + 1
    return records


def _target_tokens(target_dir: str) -> list:
    return sorted({os.path.normpath(target_dir), os.path.realpath(target_dir)}, key=len,
                  reverse=True)


def _replace_target(token: str, targets: list) -> str:
    for raw in targets:
        if token == raw or token.startswith(raw + "/"):
            return "<ROLE_TARGET>" + token[len(raw):]
        idx = token.find("=" + raw)
        if idx >= 0 and (len(token) == idx + 1 + len(raw) or token[idx + 1 + len(raw)] == "/"):
            return token[:idx + 1] + "<ROLE_TARGET>" + token[idx + 1 + len(raw):]
    return token


_META_RE = re.compile(r"^metadata=[0-9a-f]{16}$")
_EXTRA_RE = re.compile(r"^extra-filename=-[0-9a-f]{16}$")
_EXTERN_RE = re.compile(r"^([A-Za-z0-9_]+)=<ROLE_TARGET>/release/deps/lib([A-Za-z0-9_]+)-[0-9a-f]{16}\.rlib$")
_NATIVE_RE = re.compile(r"^native=<ROLE_TARGET>/release/build/([A-Za-z0-9_.+-]+)-[0-9a-f]{16}/out$")


def normalize_rustc_argv(args: list, target_dir: str, native_cc: str = None) -> list:
    """Ephemeral out-dir/metadata/extra-filename/extern/native values -> tokens.

    `native_cc` is the injected CC launcher path: cargo forwards the injected
    target-linker variable as `-C linker=<path>`, normalized to `<NATIVE_CC>`.
    """
    targets = _target_tokens(target_dir)
    out = []
    for token in args:
        token = _replace_target(token, targets)
        if native_cc is not None and token == "linker=%s" % native_cc:
            token = "linker=<NATIVE_CC>"
        elif _META_RE.match(token):
            token = "metadata=<META>"
        elif _EXTRA_RE.match(token):
            token = "extra-filename=-<EXTRA>"
        else:
            m = _EXTERN_RE.match(token)
            if m:
                token = "%s=<ROLE_TARGET>/release/deps/lib%s-<HASH>.rlib" % (m.group(1), m.group(2))
            else:
                m = _NATIVE_RE.match(token)
                if m:
                    token = "native=<ROLE_TARGET>/release/build/%s-<HASH>/out" % m.group(1)
        out.append(token)
    return out


def rustc_argv_grammar(timing_schema: dict, role: str) -> dict:
    """The pinned rustc argv grammar of the limber bench (`rustc_argv_grammar.roles.<GRAMMAR_
    ROLE>` of the shared timing schema). Fails closed with `RustcArgvInvalid` naming the
    missing role while the schema does not pin it (the integrator captures it from a real
    evidence build; until then only the stub environment can run)."""
    grammar = timing_schema.get("rustc_argv_grammar")
    roles = grammar.get("roles") if isinstance(grammar, dict) else None
    if not isinstance(roles, dict):
        raise _err(_EV_STAGE, "RustcArgvInvalid", "timing-schema-v2.json has no "
                   "rustc_argv_grammar.roles table")
    key = pc.GRAMMAR_ROLE
    if key not in roles or not isinstance(roles[key], dict) or \
            not isinstance(roles[key].get("features"), dict):
        raise _err(_EV_STAGE, "RustcArgvInvalid",
                   "no rustc argv grammar is pinned for role %r (runner role %r) in "
                   "specs/poseidon/timing-schema-v2.json: rustc_argv_grammar.roles has %s; "
                   "capture the limber bench's -vv evidence build and add "
                   "rustc_argv_grammar.roles.%s before a real run can pass" %
                   (key, role, sorted(roles), key))
    return roles[key]


def _check_rustc_argv(normalized: list, grammar: dict, features: list) -> None:
    role_grammar = grammar["features"]
    key = "none" if not features else "+".join(features)
    pinned = role_grammar.get(key)
    if pinned is None:
        raise _err(_EV_STAGE, "RustcArgvInvalid", "no rustc argv grammar is pinned for the "
                   "feature set %r" % (features,))
    expected = pinned["tokens"]
    for i, token in enumerate(normalized):
        if i >= len(expected):
            raise _err(_EV_STAGE, "RustcArgvInvalid", "extra rustc token %r at %d" % (token, i))
        if token != expected[i]:
            raise _err(_EV_STAGE, "RustcArgvInvalid", "rustc token %d is %r, the grammar "
                       "expects %r" % (i, token, expected[i]))
    if len(normalized) < len(expected):
        raise _err(_EV_STAGE, "RustcArgvInvalid", "rustc argv is missing %r at %d" %
                   (expected[len(normalized)], len(normalized)))


def _split_native_command(command: str) -> tuple:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise _err(_EV_STAGE, "NativeToolInvalid", "cannot parse native line: %s" % exc) from None
    cwd = None
    if tokens[:1] == ["cd"] and len(tokens) >= 4 and tokens[2] == "&&":
        cwd = tokens[1]
        tokens = tokens[3:]
    if tokens[:1] == ["env"]:
        tokens = tokens[1:]
    while tokens:
        if tokens[0] == "-u" and len(tokens) >= 2:
            tokens = tokens[2:]
        elif ENV_TOKEN_RE.match(tokens[0]):
            tokens = tokens[1:]
        else:
            break
    if not tokens:
        raise _err(_EV_STAGE, "NativeToolInvalid", "native line without a program: %r" % command)
    return cwd, tokens[0], tokens[1:]


def _check_native_lines(stdout_lines: list, env: dict, target_dir: str, allowed_roots: list,
                        targets: list) -> tuple:
    injected = {env["CC"]: "cc", env["CXX"]: "cxx", env["AR"]: "ar", env["RANLIB"]: "ranlib"}
    lines = []
    env_reads = {}
    for line in stdout_lines:
        m = NATIVE_ENVREAD_RE.match(line)
        if m:
            pkg, name, value, inner = m.group(1), m.group(3), m.group(4), m.group(5)
            if value == "None":
                raise _err(_EV_STAGE, "NativeToolInvalid", "%s read %s = None: the injected "
                           "launcher was not observed" % (pkg, name))
            if inner != env[name]:
                raise _err(_EV_STAGE, "NativeToolInvalid", "%s read %s = %r, injected %r" %
                           (pkg, name, inner, env[name]))
            env_reads.setdefault(name, set()).add(inner)
            continue
        m = NATIVE_RUNNING_RE.match(line)
        if not m:
            continue
        pkg, version, command = m.group(1), m.group(2), m.group(3)
        cwd, program, args = _split_native_command(command)
        if program == "xcrun":
            if args != ["--show-sdk-version", "--sdk", "macosx"]:
                raise _err(_EV_STAGE, "NativeToolInvalid", "%s ran an unexpected xcrun command"
                           % pkg)
            klass = "sdk_probe"
        elif program in injected:
            klass = injected[program]
        else:
            raise _err(_EV_STAGE, "NativeToolInvalid", "%s invoked %r, which is not an injected "
                       "tool" % (pkg, program))
        for path in ([cwd] if cwd else []) + [a for a in args if a.startswith("/")]:
            if path == "/dev/null" or any(_under(path, r) for r in allowed_roots):
                continue
            raise _err(_EV_STAGE, "NativeToolInvalid", "%s native line names %s outside the "
                       "audited roots" % (pkg, path))
        lines.append({"package": pkg, "version": version, "class": klass, "program": program,
                      "cwd": _replace_target(cwd, targets) if cwd else None,
                      "args": [_replace_target(a, targets) for a in args]})
    return lines, {k: sorted(v) for k, v in env_reads.items()}


def _check_executable(path, target_dir: str) -> dict:
    stage, code = _EV_STAGE, "EvidenceBuildFailed"
    if not isinstance(path, str) or not os.path.isabs(path):
        raise _err(stage, code, "compiler-artifact executable %r is not absolute" % (path,))
    rel = os.path.relpath(path, target_dir)
    if rel.startswith("..") or os.path.isabs(rel):
        raise _err(stage, code, "executable %s is not below the target %s" % (path, target_dir))
    current = target_dir
    for part in rel.split(os.sep):
        current = os.path.join(current, part)
        st = os.lstat(current)
        if stat.S_ISLNK(st.st_mode):
            raise _err(stage, code, "executable path component %s is a symlink" % current)
    if not stat.S_ISREG(st.st_mode):
        raise _err(stage, code, "executable %s is not a regular file" % path)
    if not _strict_under(os.path.realpath(path), os.path.realpath(target_dir)):
        raise _err(stage, code, "executable %s resolves outside the target" % path)
    return {"relative_path": rel.replace(os.sep, "/"), "size": st.st_size,
            "sha256": pc.sha256_file(path)}


def evidence_build(repo: str, env: dict, target_dir: str, features: list, role: str,
                   timing_schema: dict, dependency_audit: dict | None = None) -> dict:
    """Run the exact evidence command in the closed environment and validate its outputs.

    `dependency_audit` (added keyword): the frozen dependency-source audit whose raw package
    roots bound every `CARGO_MANIFEST_DIR`; without it the audited roots are the registry
    `src/<index>` and `git/checkouts` prefixes of the closed `CARGO_HOME`.
    """
    stage, code = _EV_STAGE, "EvidenceBuildFailed"
    if role not in pc.SYSTEM_ROLES:
        raise _err(stage, code, "evidence_build implements the limber roles %s only (got %r)" %
                   (list(pc.SYSTEM_ROLES), role))
    grammar = rustc_argv_grammar(timing_schema, role)
    repo = os.path.abspath(repo)
    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir) or os.path.islink(target_dir):
        raise _err(stage, "TempDirInvalid", "%s is not a non-symlink directory" % target_dir)
    if os.listdir(target_dir):
        raise _err(stage, "TempDirInvalid", "role target %s is not empty" % target_dir)
    if _under(os.path.realpath(target_dir), os.path.realpath(repo)):
        raise _err(stage, "PathOverlap", "role target %s is inside the repository" % target_dir)
    if env.get("CARGO_TARGET_DIR") != target_dir:
        raise _err(stage, code, "CARGO_TARGET_DIR must be the role target %s" % target_dir)
    for name in ("RUSTC", "CC", "CXX", "AR", "RANLIB", "CARGO_HOME"):
        if not env.get(name):
            raise _err(stage, code, "closed environment lacks %s" % name)
    argv = pc.evidence_command(features)
    argv[0] = os.path.join(os.path.dirname(env["RUSTC"]), "cargo")
    proc = _run(argv, repo, env, stage, code, "evidence build")
    log_bytes = b"--- stdout ---\n" + proc.stdout + b"\n--- stderr ---\n" + proc.stderr
    if proc.returncode != 0:
        raise _err(stage, code, "evidence build exited %d: %s" %
                   (proc.returncode, proc.stderr.decode("utf-8", "replace").strip()[-1500:]))
    rules = pc.evidence_artifact_rules(timing_schema)
    stdout_text = proc.stdout.decode("utf-8", "replace")
    stdout_lines = stdout_text.split("\n")
    counts = {"json": 0, "build_script": 0, "blank": 0}
    artifacts = []
    finished = []
    build_script_msgs = {}
    for line in stdout_lines:
        if line == "":
            counts["blank"] += 1
        elif line.startswith("{"):
            counts["json"] += 1
            try:
                msg = json.loads(line)
            except ValueError as exc:
                raise _err(stage, code, "malformed Cargo JSON message: %s" % exc) from None
            reason = msg.get("reason")
            if reason == "compiler-artifact":
                target = msg.get("target", {})
                if target.get("kind") == rules["target_kind"] and \
                        target.get("name") == rules["target_name"]:
                    artifacts.append(msg)
            elif reason == "build-finished":
                finished.append(msg)
            elif reason == "build-script-executed":
                build_script_msgs[msg.get("package_id", "")] = msg
        elif BUILD_SCRIPT_LINE_RE.match(line):
            counts["build_script"] += 1
        else:
            raise _err(stage, code, "unclassified stdout line: %r" % line[:200])
    if len(finished) != 1 or finished[0].get("success") is not True:
        raise _err(stage, code, "expected exactly one successful build-finished message")
    if len(artifacts) != rules["count"]:
        raise _err(stage, code, "expected exactly %d bench compiler-artifact for %s, found %d"
                   % (rules["count"], rules["target_name"], len(artifacts)))
    artifact = artifacts[0]
    if artifact.get("fresh") is not False:
        raise _err(stage, code, "the bench compiler-artifact is Fresh (warm cache rejected)")
    profile = artifact.get("profile", {})
    for key, expected in rules["profile"].items():
        if profile.get(key) != expected:
            raise _err(stage, code, "compiler-artifact profile.%s is %r, expected %r" %
                       (key, profile.get(key), expected))
    if sorted(artifact.get("features", [])) != sorted(features):
        raise _err(stage, code, "compiler-artifact features %r differ from %r" %
                   (artifact.get("features"), features))
    roots = {"SOURCE": repo, "CARGO_HOME": env["CARGO_HOME"], "ROLE_TARGET": target_dir,
             "SYSROOT": os.path.dirname(os.path.dirname(env["RUSTC"]))}
    if env.get("HOME"):
        roots["HOME"] = env["HOME"]
    src_path = project_string(artifact.get("target", {}).get("src_path", ""), roots)
    if src_path != rules["src_path"]:
        raise _err(stage, code, "bench src_path %r is not %r" % (src_path, rules["src_path"]))
    executable = _check_executable(artifact.get("executable"), target_dir)
    if artifact.get("executable") not in artifact.get("filenames", []):
        raise _err(stage, code, "executable is not among the artifact filenames")
    # Source roots for CARGO_MANIFEST_DIR / native paths.
    if dependency_audit is not None:
        dep_roots = [loc["root"] for loc in dependency_audit["raw_locators"]["packages"].values()
                     if loc["kind"] != "path"]
        policy = "dependency_audit"
    else:
        dep_roots = [os.path.join(env["CARGO_HOME"], "registry", "src"),
                     os.path.join(env["CARGO_HOME"], "git", "checkouts")]
        policy = "cargo_home_prefixes"
    manifest_roots = [repo] + dep_roots
    records = _parse_running_records(proc.stderr.decode("utf-8", "replace"))
    manifest_dirs = set()
    bench_records = []
    for rec in records:
        base = os.path.basename(rec["program"])
        if base == "rustc" or base.startswith("rustc-"):
            if rec["program"] != env["RUSTC"]:
                raise _err(stage, "RustcArgvInvalid", "rustc invoked as %r, injected RUSTC is %r "
                           "(wrapper or foreign compiler rejected)" % (rec["program"], env["RUSTC"]))
            args = rec["args"]
            if any(args[i:i + 2] == ["--crate-name", rules["target_name"]]
                   for i in range(len(args) - 1)):
                bench_records.append(rec)
        elif not _strict_under(rec["program"], target_dir):
            raise _err(stage, code, "Running record invokes %r, neither rustc nor a build script"
                       " in the role target" % rec["program"])
        manifest = rec["env"].get("CARGO_MANIFEST_DIR")
        if manifest is None:
            raise _err(stage, code, "Running record without CARGO_MANIFEST_DIR")
        if not any(_under(manifest, r) for r in manifest_roots):
            raise _err(stage, code, "CARGO_MANIFEST_DIR %s is outside the audited source roots"
                       % manifest)
        manifest_dirs.add(project_string(manifest, roots))
    if len(bench_records) != 1:
        raise _err(stage, "RustcArgvInvalid", "expected exactly one bench rustc invocation, "
                   "found %d" % len(bench_records))
    bench = bench_records[0]
    normalized = normalize_rustc_argv(bench["args"], target_dir, native_cc=env.get("CC"))
    _check_rustc_argv(normalized, grammar, features)
    # Native tool lines.
    native_roots = [target_dir, os.path.realpath(target_dir), repo] + dep_roots
    for extra in timing_schema["evidence"].get("native_path_roots", []):
        native_roots.append(extra)
    targets = _target_tokens(target_dir)
    native_lines, env_reads = _check_native_lines(stdout_lines, env, target_dir, native_roots,
                                                  targets)
    zstd = {"bundled_compile_lines": 0, "linked_libs": None,
            "required": bool(pc.REQUIRE_BUNDLED_ZSTD)}
    for line in native_lines:
        if line["package"] == "zstd-sys" and line["class"] == "cc" and "-c" in line["args"]:
            src = line["args"][line["args"].index("-c") + 1]
            if src.startswith("zstd/lib/"):
                zstd["bundled_compile_lines"] += 1
    for pid, msg in build_script_msgs.items():
        if "#zstd-sys@" in pid:
            zstd["linked_libs"] = msg.get("linked_libs")
    # The bundled-zstd requirement is Zinc's (its proof stream is zstd-compressed); limber
    # has no zstd dependency, so the schema flag is gated by the system profile.
    if pc.REQUIRE_BUNDLED_ZSTD and timing_schema["evidence"].get("require_bundled_zstd", True):
        if zstd["bundled_compile_lines"] == 0 or zstd["linked_libs"] != ["static=zstd"]:
            raise _err(stage, "NativeToolInvalid", "the log does not show the bundled zstd "
                       "source build (%r)" % zstd)
    return {
        "argv": argv, "argv_normalized": [project_string(a, roots) for a in argv],
        "log_bytes": log_bytes, "log_sha256": pc.sha256_hex(log_bytes), "exit": 0,
        "executable": executable,
        "artifact": {"package_id": project_string(artifact.get("package_id", ""), roots),
                     "features": sorted(artifact.get("features", [])), "profile": profile,
                     "src_path": src_path, "fresh": False},
        "rustc_program": project_string(bench["program"], roots),
        "rustc_argv_normalized": normalized, "rustc_argv_raw_sha256": bench["raw_sha256"],
        "running_records": len(records), "native_lines": native_lines,
        "native_env_reads": env_reads, "manifest_dirs": sorted(manifest_dirs),
        "source_root_policy": policy, "zstd_sys": zstd, "stdout_lines": counts,
        "features": list(features), "role": role, "target_initially_empty": True,
    }


def rehash_executable(target_dir: str, executable: dict) -> str:
    path = os.path.join(target_dir, *executable["relative_path"].split("/"))
    try:
        now = _check_executable(path, target_dir)
    except pc.RunnerError as exc:
        raise _err(_EV_STAGE, "ExecutableChanged", exc.message) from None
    except OSError as exc:
        raise _err(_EV_STAGE, "ExecutableChanged", "%s: %s" % (path, exc)) from None
    if now["size"] != executable["size"] or now["sha256"] != executable["sha256"]:
        raise _err(_EV_STAGE, "ExecutableChanged", "evidenced executable %s changed" %
                   executable["relative_path"])
    return now["sha256"]


def build_profile_document(evidence: dict, snapshot: dict, dependency: dict, cargo_config: dict,
                           toolchain: dict, native: dict, env_raw: dict, roots: dict,
                           rejected_audit: dict, compiled_metadata: dict, timing_schema: dict,
                           features: list, role: str = "hyrax") -> dict:
    """The canonical `build-profile.json` object (plan v10 lines 1880-1899)."""
    probes_required = timing_schema["bench_profile"]["probes"]
    probes = {}
    for key, expected in probes_required.items():
        value = compiled_metadata.get(key)
        if value != expected:
            raise _err(_EV_STAGE, "EvidenceBuildFailed", "compiled probe %s is %r, required %r"
                       % (key, value, expected))
        probes[key] = value
    proj = lambda v: project_value(v, roots)  # noqa: E731
    launchers = {"cargo": proj(toolchain["cargo_launcher"]),
                 "rustc": proj(toolchain["rustc_launcher"])}
    doc = {
        "schema": BUILD_PROFILE_SCHEMA, "role": role, "features": list(features),
        "log_sha256": evidence["log_sha256"], "command": evidence["argv_normalized"],
        "exit": evidence["exit"],
        "manifest": {"sha256": cargo_config["manifest"]["sha256"],
                     "profile_block_sha256": cargo_config["manifest"]["profile_block_sha256"],
                     "profile_block": cargo_config["manifest"]["profile_block"]},
        "dependency_source_id": dependency["dependency_source_id"],
        "dependency_sources_sha256": pc.sha256_hex(dependency["bytes"]),
        "dependency_closure_sha256": dependency["closure_sha256"],
        "source": {"commit": snapshot["commit"], "tree": snapshot["tree"],
                   "source_snapshot_id": snapshot["source_snapshot_id"],
                   "dirty": snapshot["dirty"], "lock_sha256": snapshot["lock_sha256"],
                   "closure_sha256": snapshot["closure_sha256"]},
        "target": {"path": "<ROLE_TARGET>", "initially_empty": True, "symlink": False},
        "launchers": launchers,
        "toolchain": {"cargo_path": proj(toolchain["cargo_path"]),
                      "cargo_sha256": toolchain["cargo_sha256"],
                      "cargo_version": toolchain["cargo_version"],
                      "rustc_path": proj(toolchain["rustc_path"]),
                      "rustc_sha256": toolchain["rustc_sha256"],
                      "rustc_vV": toolchain["rustc_vV"], "host_triple": toolchain["host_triple"],
                      "sysroot": proj(toolchain["sysroot"]),
                      "toolchain_closure_sha256": toolchain["toolchain_closure_sha256"]},
        "native": proj({k: v for k, v in native.items() if k != "env"}),
        "environment": {"raw": dict(env_raw), "projected": identity_projection(env_raw, roots)},
        "cargo_config": proj({k: v for k, v in cargo_config.items()}),
        "rustflags": env_raw["RUSTFLAGS"],
        "rejected_inputs": rejected_audit, "wrapper_chain": [],
        "bench_rustc_invocation": {"program": evidence["rustc_program"],
                                   "argv_normalized": evidence["rustc_argv_normalized"],
                                   "raw_sha256": evidence["rustc_argv_raw_sha256"]},
        "native_lines": evidence["native_lines"], "native_env_reads": evidence["native_env_reads"],
        "manifest_dirs": evidence["manifest_dirs"],
        "source_root_policy": evidence["source_root_policy"], "zstd_sys": evidence["zstd_sys"],
        "artifact": evidence["artifact"], "executable": dict(evidence["executable"]),
        "recheck_policy": list(timing_schema["evidence"]["recheck_policy"]),
        "probes": probes,
    }
    _canon(doc)  # must be canonical-JSON encodable
    return doc


def rehash_all(audit_bundle: dict) -> dict:
    """Pre/post audit hashes for a Cargo subprocess: source snapshot, dependency closure,
    Cargo config, toolchain closure, native/SDK closure, evidenced executable."""
    snapshot = audit_bundle.get("snapshot", audit_bundle.get("source"))
    recheck_source_snapshot(audit_bundle["repo"], snapshot)
    out = {"source": snapshot["source_snapshot_id"],
           "dependency": rehash_dependency_closure(audit_bundle["dependency"]),
           "cargo_config": recheck_cargo_config(audit_bundle["cargo_config"],
                                                audit_bundle.get("timing_schema")),
           "toolchain": recheck_toolchain(audit_bundle["toolchain"]),
           "native": recheck_native_tools(audit_bundle["native"]) if
           audit_bundle.get("native") is not None else None,
           "executable": None}
    executable = audit_bundle.get("executable")
    if executable is not None:
        out["executable"] = rehash_executable(audit_bundle["target_dir"], executable)
    return out


def run_cargo_subprocess(argv: list, env: dict, cwd: str, audit_bundle: dict, log_sink) -> dict:
    """Pre-rehash, run (`stdin=DEVNULL`), post-rehash. `log_sink` is either a callable
    `sink(name, bytes)` or a mapping `{"stdout": path, "stderr": path}`."""
    pre = rehash_all(audit_bundle)
    start_utc = pc.utc_rfc3339_ns()
    start_mono = time.monotonic_ns()
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise _err("gates", "InternalError", "cannot execute %s: %s" % (argv[0], exc)) from None
    out, err = proc.communicate()
    end_mono = time.monotonic_ns()
    end_utc = pc.utc_rfc3339_ns()
    if callable(log_sink):
        log_sink("stdout", out)
        log_sink("stderr", err)
    else:
        pc.write_bytes(log_sink["stdout"], out)
        pc.write_bytes(log_sink["stderr"], err)
    post = rehash_all(audit_bundle)
    rc = proc.returncode
    return {"argv": list(argv), "exit": rc if rc >= 0 else None, "signal": -rc if rc < 0 else None,
            "pid": proc.pid, "start_utc": start_utc, "end_utc": end_utc,
            "start_mono": start_mono, "end_mono": end_mono, "pre": pre, "post": post,
            "stdout_sha256": pc.sha256_hex(out), "stderr_sha256": pc.sha256_hex(err),
            "stdout_size": len(out), "stderr_size": len(err)}
