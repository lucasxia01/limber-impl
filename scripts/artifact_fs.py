#!/usr/bin/env python3
"""Atomic no-replace directory publication and artifact integrity (plan v9, section 9).

- `rename_dir_noreplace(src, dst)`: `renamex_np(RENAME_EXCL)` on macOS, `renameat2(
  RENAME_NOREPLACE)` on Linux, fail closed elsewhere. A prior existence check is diagnostic.
- `fsync_tree_bottom_up(root)`: every regular file, then every directory bottom-up
  including `root` ("durable before rename").
- `commit_staging(staging, final)`: the publication commit. A pre-rename error leaves
  staging in place; a post-rename parent fsync failure raises `DurabilityUnconfirmed`
  (the final directory may be visible; nothing is claimed about staging; no rollback).
- Artifact index (`artifacts.sha256`), detached manifest pairs and `verify_artifact_dir`;
  `verify_indexed_dir` applies the same integrity checks without a file-set policy (for
  archives whose payload policy belongs to another repository) and
  `verify_and_reuse_existing` is the content-store rule for an object that already exists
  (rename-race winner included): full rehash, bottom-up fsync and store-directory fsync.

Payload paths are ASCII `[A-Za-z0-9._/-]+` with no empty, `.` or `..` component and are
sorted bytewise. Symlinks anywhere below a root are rejected.
"""
from __future__ import annotations

import ctypes
import errno
import fnmatch
import os
import platform
import re
import stat
import sys

try:
    from scripts import poseidon_common as pc
except ImportError:  # executed as a plain file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import poseidon_common as pc  # noqa: E402

RENAME_EXCL = 0x4          # macOS renamex_np flag
RENAME_NOREPLACE = 1       # Linux renameat2 flag
AT_FDCWD = -100
SYS_RENAMEAT2 = {"x86_64": 316, "aarch64": 276}

INDEX_NAME = "artifacts.sha256"
MANIFEST_PAIR = ("manifest.json", "manifest.sha256")
FAILED_PAIR = ("manifest.failed.json", "manifest.failed.sha256")
SEAL_FILES = (INDEX_NAME,) + MANIFEST_PAIR + FAILED_PAIR
PAYLOAD_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
INDEX_LINE_RE = re.compile(r"^([0-9a-f]{64})  (0|[1-9][0-9]*)  ([A-Za-z0-9._/-]+)$")


# ---------------------------------------------------------------------------
# Rename and durability primitives.
# ---------------------------------------------------------------------------

def _libc():
    return ctypes.CDLL(None, use_errno=True)


def rename_dir_noreplace(src: str, dst: str) -> None:
    """Atomically rename `src` to `dst`, failing if `dst` exists (any kind of entry)."""
    if os.path.lexists(dst):  # diagnostic only; the syscall is what guarantees no-replace
        raise pc.RunnerError("commit", "RenameRefused", "destination %s already exists" % dst)
    b_src, b_dst = os.fsencode(src), os.fsencode(dst)
    if sys.platform == "darwin":
        libc = _libc()
        try:
            fn = libc.renamex_np
        except AttributeError:
            raise pc.RunnerError("commit", "UnsupportedPlatform",
                                 "renamex_np is unavailable") from None
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        rc = fn(b_src, b_dst, RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        libc = _libc()
        try:
            fn = libc.renameat2
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_uint]
            fn.restype = ctypes.c_int
            rc = fn(AT_FDCWD, b_src, AT_FDCWD, b_dst, RENAME_NOREPLACE)
        except AttributeError:
            number = SYS_RENAMEAT2.get(platform.machine())
            if number is None:
                raise pc.RunnerError("commit", "UnsupportedPlatform",
                                     "no renameat2 syscall number for %s" %
                                     platform.machine()) from None
            syscall = libc.syscall
            syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_char_p, ctypes.c_uint]
            syscall.restype = ctypes.c_long
            rc = syscall(number, AT_FDCWD, b_src, AT_FDCWD, b_dst, RENAME_NOREPLACE)
    else:
        raise pc.RunnerError("commit", "UnsupportedPlatform",
                             "no no-replace rename on platform %s" % sys.platform)
    if rc != 0:
        err = ctypes.get_errno()
        code = "RenameRefused" if err == errno.EEXIST else "RenameRefused"
        raise pc.RunnerError("commit", code, "no-replace rename %s -> %s failed: %s" %
                             (src, dst, os.strerror(err)))


def fsync_file(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _walk_checked(root: str):
    """Yield (dirpath, dirnames, filenames) bottom-up, rejecting symlinks and special files."""
    if os.path.islink(root):
        raise pc.RunnerError("seal", "SymlinkRejected", "root %s is a symlink" % root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise pc.RunnerError("seal", "SymlinkRejected",
                                     "symlink %s" % os.path.join(dirpath, name))
        for name in filenames:
            st = os.lstat(os.path.join(dirpath, name))
            if stat.S_ISLNK(st.st_mode):
                raise pc.RunnerError("seal", "SymlinkRejected",
                                     "symlink %s" % os.path.join(dirpath, name))
            if not stat.S_ISREG(st.st_mode):
                raise pc.RunnerError("seal", "UnsafePath",
                                     "not a regular file: %s" % os.path.join(dirpath, name))
        yield dirpath, dirnames, filenames


def fsync_tree_bottom_up(root: str) -> None:
    """fsync every regular file, then every directory bottom-up including `root`."""
    entries = list(_walk_checked(root))
    try:
        for dirpath, _, filenames in entries:
            for name in filenames:
                fsync_file(os.path.join(dirpath, name))
        for dirpath, _, _ in entries:
            fsync_dir(dirpath)
    except OSError as exc:
        raise pc.RunnerError("commit", "DurabilitySyncFailed", "fsync failed: %s" % exc) from None


def commit_staging(staging: str, final: str) -> None:
    """Durable-before-rename walk, no-replace rename, then fsync of the open parent fd."""
    if os.path.dirname(os.path.abspath(staging)) != os.path.dirname(os.path.abspath(final)):
        raise pc.RunnerError("commit", "PathOverlap",
                             "staging and final directories must share one parent")
    fsync_tree_bottom_up(staging)
    parent = os.path.dirname(os.path.abspath(final))
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        raise pc.RunnerError("commit", "DurabilitySyncFailed",
                             "cannot open parent %s: %s" % (parent, exc)) from None
    try:
        rename_dir_noreplace(staging, final)  # a failure here leaves staging untouched
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise pc.RunnerError(
                "durability", "DurabilityUnconfirmed",
                "%s was renamed but the parent directory fsync failed (%s); the final "
                "directory may be visible with unconfirmed durability" % (final, exc)) from None
    finally:
        os.close(parent_fd)


def create_staging_exclusive(path: str) -> None:
    try:
        os.mkdir(path)
    except FileExistsError:
        raise pc.RunnerError("staging", "StagingExists", "%s already exists" % path) from None


# ---------------------------------------------------------------------------
# Path safety, index, manifest pair.
# ---------------------------------------------------------------------------

def check_payload_path(rel: str) -> str:
    if not isinstance(rel, str) or not rel.isascii() or not PAYLOAD_PATH_RE.match(rel):
        raise pc.RunnerError("seal", "UnsafePath", "payload path %r is not ASCII [A-Za-z0-9._/-]+"
                             % (rel,))
    for component in rel.split("/"):
        if component in ("", ".", ".."):
            raise pc.RunnerError("seal", "UnsafePath",
                                 "payload path %r has an empty, . or .. component" % rel)
    return rel


def payload_files(root: str, exclude=SEAL_FILES) -> list:
    """Bytewise-sorted payload paths below `root` (symlinks rejected, seal files excluded)."""
    out = []
    for rel in pc.list_files(root):
        check_payload_path(rel)
        if rel in exclude:
            continue
        out.append(rel)
    return sorted(out, key=lambda p: p.encode("ascii"))


def format_index_line(sha: str, size: int, rel: str) -> str:
    return "%s  %d  %s\n" % (sha, size, rel)


def write_artifact_index(root: str, exclude=SEAL_FILES) -> tuple:
    """Write `artifacts.sha256`; returns `(index_bytes, index_sha256, entries)`."""
    entries = []
    lines = []
    for rel in payload_files(root, exclude):
        full = os.path.join(root, rel)
        sha = pc.sha256_file(full)
        size = os.path.getsize(full)
        entries.append((rel, size, sha))
        lines.append(format_index_line(sha, size, rel))
    data = "".join(lines).encode("ascii")
    pc.write_bytes(os.path.join(root, INDEX_NAME), data)
    return data, pc.sha256_hex(data), entries


def parse_artifact_index(data: bytes) -> list:
    """Strictly parse index bytes into `[(rel, size, sha)]`, checking order and uniqueness."""
    text = data.decode("ascii", "strict") if data else ""
    if data and not text.endswith("\n"):
        raise pc.RunnerError("seal", "IntegrityMismatch", "artifact index lacks a final LF")
    entries = []
    previous = None
    for line in text.split("\n")[:-1] if text else []:
        m = INDEX_LINE_RE.match(line)
        if not m:
            raise pc.RunnerError("seal", "IntegrityMismatch", "malformed index line %r" % line)
        sha, size, rel = m.group(1), int(m.group(2)), m.group(3)
        check_payload_path(rel)
        key = rel.encode("ascii")
        if previous is not None and key <= previous:
            raise pc.RunnerError("seal", "IntegrityMismatch",
                                 "index is not strictly bytewise sorted at %r" % rel)
        previous = key
        entries.append((rel, size, sha))
    return entries


def seal_manifest(root: str, manifest_obj: dict, failed: bool = False) -> str:
    """Write the manifest pair; `artifact_index_sha256` must match the existing index."""
    names = FAILED_PAIR if failed else MANIFEST_PAIR
    index_path = os.path.join(root, INDEX_NAME)
    if not os.path.isfile(index_path):
        raise pc.RunnerError("seal", "SealFailed", "artifact index is missing")
    if manifest_obj.get("artifact_index_sha256") != pc.sha256_file(index_path):
        raise pc.RunnerError("seal", "SealFailed",
                             "manifest artifact_index_sha256 does not match the index")
    for name in SEAL_FILES[1:]:
        if os.path.lexists(os.path.join(root, name)):
            raise pc.RunnerError("seal", "SealFailed", "%s already exists" % name)
    data = pc.canonical_json_bytes(manifest_obj)
    digest = pc.sha256_hex(data)
    pc.write_bytes(os.path.join(root, names[0]), data)
    pc.write_detached_digest(os.path.join(root, names[1]), digest)
    return digest


def _match_allowed(rel: str, required_files, allowed_globs) -> bool:
    if rel in required_files:
        return True
    return any(fnmatch.fnmatchcase(rel, g) for g in allowed_globs)


def verify_artifact_dir(root: str, required_files, allowed_globs, confirm_durability=True):
    """Rehash index and every indexed file, check the exact file set, confirm durability.

    Returns `(manifest, manifest_sha256, status)` where `status` is `complete` or `failed`
    according to the manifest pair found. Raises `RunnerError` on any mismatch.
    """
    root = os.path.abspath(root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise pc.RunnerError("seal", "SymlinkRejected", "%s is not a plain directory" % root)
    if os.path.basename(root).endswith(".staging") or os.path.basename(root).startswith("."):
        raise pc.RunnerError("seal", "ManifestInvalid", "%s is a staging/temporary name" % root)
    present = set(pc.list_files(root))
    has_ok = all(n in present for n in MANIFEST_PAIR)
    has_failed = all(n in present for n in FAILED_PAIR)
    partial = any(n in present for n in MANIFEST_PAIR + FAILED_PAIR)
    if has_ok == has_failed or (partial and not (has_ok or has_failed)):
        raise pc.RunnerError("seal", "ManifestInvalid",
                             "exactly one complete manifest pair is required")
    names = MANIFEST_PAIR if has_ok else FAILED_PAIR
    status = "complete" if has_ok else "failed"
    with open(os.path.join(root, names[0]), "rb") as f:
        manifest_bytes = f.read()
    expected = pc.read_detached_digest(os.path.join(root, names[1]))
    actual = pc.sha256_hex(manifest_bytes)
    if actual != expected:
        raise pc.RunnerError("seal", "IntegrityMismatch",
                             "%s digest %s != detached %s" % (names[0], actual, expected))
    manifest = pc.load_canonical_json(manifest_bytes, names[0])
    if not isinstance(manifest, dict) or manifest.get("status") != status:
        raise pc.RunnerError("seal", "ManifestInvalid",
                             "manifest status does not match the pair kind %s" % status)
    if INDEX_NAME not in present:
        raise pc.RunnerError("seal", "FileSetMismatch", "artifact index is missing")
    with open(os.path.join(root, INDEX_NAME), "rb") as f:
        index_bytes = f.read()
    if manifest.get("artifact_index_sha256") != pc.sha256_hex(index_bytes):
        raise pc.RunnerError("seal", "IntegrityMismatch", "artifact index hash mismatch")
    entries = parse_artifact_index(index_bytes)
    indexed = {rel for rel, _, _ in entries}
    payload = present - {INDEX_NAME} - set(names)
    if payload != indexed:
        extra = sorted(payload - indexed)
        missing = sorted(indexed - payload)
        raise pc.RunnerError("seal", "FileSetMismatch",
                             "index/payload mismatch: unindexed %s, missing %s" %
                             (extra, missing))
    for rel, size, sha in entries:
        full = os.path.join(root, rel)
        if os.path.getsize(full) != size or pc.sha256_file(full) != sha:
            raise pc.RunnerError("seal", "IntegrityMismatch", "payload %s was modified" % rel)
    missing_required = [r for r in required_files if r not in payload]
    if missing_required:
        raise pc.RunnerError("seal", "FileSetMismatch",
                             "required payloads missing: %s" % missing_required)
    unexpected = [r for r in sorted(payload) if not _match_allowed(r, required_files,
                                                                    allowed_globs)]
    if unexpected:
        raise pc.RunnerError("seal", "FileSetMismatch", "unexpected payloads: %s" % unexpected)
    if confirm_durability:
        fsync_tree_bottom_up(root)
        try:
            fsync_dir(os.path.dirname(root))
        except OSError as exc:
            raise pc.RunnerError("commit", "DurabilitySyncFailed",
                                 "parent fsync failed: %s" % exc) from None
    return manifest, actual, status


def verify_indexed_dir(root: str, confirm_durability: bool = True):
    """Integrity-only verification: manifest pair, index, every indexed file (any payload)."""
    return verify_artifact_dir(root, [], ["*"], confirm_durability=confirm_durability)


def verify_and_reuse_existing(path: str, required_files, allowed_globs,
                              expected_manifest_sha256=None):
    """An existing content-store object is reused only after the same full rehash,
    bottom-up fsync and store-directory fsync as a fresh import; any mismatch is fatal."""
    manifest, sha, status = verify_artifact_dir(path, required_files, allowed_globs,
                                                confirm_durability=True)
    if expected_manifest_sha256 is not None and sha != expected_manifest_sha256:
        raise pc.RunnerError("link", "IntegrityMismatch",
                             "existing object %s rehashes to %s" % (path, sha))
    return manifest, sha, status
