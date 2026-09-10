#!/usr/bin/env python3
"""Stand-in for `scripts/poseidon_build_env.py` (package C) used by the unit tests (limber).

It implements the public API named in the Python API contract with canned but internally
consistent values, so the runner/assembler code paths that integrate package C can be
exercised without a Rust toolchain. Tests inject it by monkeypatching the `be` attribute of
the importing module (`poseidon_runner.be = stub_build_env`); the subprocess wrapper
`stub_limber_runner.py` does the same for out-of-process runs.

Test-only inputs live in `STUB_SETTINGS` (never in the runner's environment): the metadata
JSON the stub bench prints and the failure injections. `closed_environment` forwards them to
the stub cargo/bench as `STUB_*` names.

Every path-like value the runner may record is a real absolute path here; the runner is
responsible for projecting them through `identity_projection` before they enter a config.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from scripts import poseidon_common as pc  # noqa: E402

STUB_SETTINGS = {"metadata_json": None, "fail_gate": None, "fail_preflight": None,
                 "fail_child": None}
CREATED_TARGETS = []  # role targets seen by evidence_build (tests assert their removal)
STUB_ENV_NAMES = {"metadata_json": "STUB_METADATA_JSON", "fail_gate": "STUB_FAIL_GATE",
                  "fail_preflight": "STUB_FAIL_PREFLIGHT", "fail_child": "STUB_FAIL_CHILD"}
CARGO_VERSION = "cargo 1.98.1 (797e8a9bc 2026-08-05)"
RUSTC_VV = ("rustc 1.98.1 (48a229cea 2026-09-01)\nbinary: rustc\ncommit-hash: 48a229cea\n"
            "commit-date: 2026-09-01\nhost: aarch64-apple-darwin\nrelease: 1.98.1\n"
            "LLVM version: 21.1.0")
HOST_TRIPLE = "aarch64-apple-darwin"
LINKER_VAR = "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"
REJECTED_EXACT = frozenset((
    "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
    "CARGO", "CARGO_TARGET_DIR", "CARGO_INCREMENTAL", "CARGO_BUILD_TARGET",
    "CARGO_BUILD_RUSTFLAGS", "PYTHONPYCACHEPREFIX",
))
# The (not yet captured) limber bench rustc argv the stub evidence build reports.
STUB_RUSTC_ARGV = ["--crate-name", pc.BENCH_NAME, "--edition=2024", pc.BENCH_SRC_PATH, "-C",
                   "opt-level=3", "-C", "lto=fat", "-C", "codegen-units=1", "-C",
                   "target-cpu=native"]


def settings_from_environ(environ=None) -> None:
    """Populate `STUB_SETTINGS` from `STUB_*` names (used by the subprocess wrappers)."""
    environ = os.environ if environ is None else environ
    for key, name in STUB_ENV_NAMES.items():
        if environ.get(name):
            STUB_SETTINGS[key] = environ[name]


def _sha_file(path: str) -> str:
    return pc.sha256_file(path)


# --- source snapshot --------------------------------------------------------------------

def _walk_closure(repo: str) -> list:
    closure = []
    for dirpath, dirnames, filenames in os.walk(repo, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if not (dirpath == repo and d == ".git"))
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, repo).replace(os.sep, "/")
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                closure.append([rel, "symlink", "%o" % (st.st_mode & 0o777), os.readlink(full),
                                pc.sha256_hex(os.readlink(full).encode())])
            elif stat.S_ISREG(st.st_mode):
                closure.append([rel, "file", "%o" % (st.st_mode & 0o777), st.st_size,
                                _sha_file(full)])
            else:
                raise pc.RunnerError("source", "SourceSnapshotInvalid",
                                     "special file %s" % rel)
    return sorted(closure, key=lambda e: e[0].encode())


def source_snapshot(repo: str, allow_dirty: bool, external_roots=()) -> dict:
    commit = pc.git_head(repo)
    tree = pc.git(repo, "rev-parse", "HEAD^{tree}").strip()
    status = pc.git_status_porcelain(repo)
    dirty = bool(status)
    if dirty and not allow_dirty:
        raise pc.RunnerError("source", "DirtyWorktree",
                             "worktree is not the clean recorded tree (%d paths); set "
                             "POSEIDON_ALLOW_DIRTY=1 for an exploratory run" % len(status))
    if not pc.git_is_tracked(repo, "Cargo.lock"):
        raise pc.RunnerError("source", "LockfileUntracked", "Cargo.lock is not tracked")
    with open(os.path.join(repo, "Cargo.lock"), "rb") as f:
        lock = f.read()
    closure = _walk_closure(repo)
    closure_sha = pc.sha256_hex(pc.canonical_json_bytes(closure))
    lock_blob = pc.git(repo, "rev-parse", "HEAD:Cargo.lock").strip()
    ident = {"commit": commit, "tree": tree, "lock_blob": lock_blob,
             "lock_sha256": pc.sha256_hex(lock), "closure_sha256": closure_sha}
    return {
        "commit": commit, "tree": tree, "lock_blob": lock_blob,
        "lock_sha256": pc.sha256_hex(lock), "closure": closure, "closure_sha256": closure_sha,
        "source_snapshot_id": pc.sha256_hex(pc.canonical_json_bytes(ident)),
        "dirty": dirty,
        "dirty_closure_sha256": closure_sha if dirty else None,
        "head_index_diff_sha256": pc.sha256_hex("\n".join(status).encode()) if dirty else None,
    }


def recheck_source_snapshot(repo: str, snapshot: dict) -> None:
    now = source_snapshot(repo, allow_dirty=True)
    for key in ("commit", "tree", "lock_sha256", "closure_sha256", "dirty"):
        if now[key] != snapshot[key]:
            raise pc.RunnerError("source", "SourceChanged",
                                 "source snapshot %s changed (%r -> %r)" %
                                 (key, snapshot[key], now[key]))


# --- environment ------------------------------------------------------------------------

def _rejected_name(name: str) -> bool:
    if name in REJECTED_EXACT or name.startswith("CARGO_PROFILE_"):
        return True
    if name.startswith("CARGO_BUILD_") and "RUSTC" in name:
        return True
    if name.startswith("CARGO_TARGET_") and name.endswith(("_RUSTFLAGS", "_RUNNER", "_LINKER")):
        return True
    return False


def reject_inherited_environment(env) -> dict:
    hits = sorted(n for n in env if _rejected_name(n))
    if hits:
        raise pc.RunnerError("env", "RejectedEnvironment",
                             "rejected inherited environment names: %s" % ", ".join(hits))
    return {"schema": "zinc-plus/rejected-inputs-audit/v1", "inherited_names": len(env),
            "rejected": [], "rules": ["CARGO_PROFILE_*", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS",
                                      "RUSTC", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                                      "CARGO", "CARGO_TARGET_DIR", "CARGO_INCREMENTAL",
                                      "CARGO_BUILD_TARGET", "CARGO_BUILD_*RUSTC*",
                                      "CARGO_BUILD_RUSTFLAGS",
                                      "CARGO_TARGET_*_{RUSTFLAGS,RUNNER,LINKER}",
                                      "PYTHONPYCACHEPREFIX"]}


def resolve_toolchain(env, timing_schema: dict, cwd=None) -> dict:
    cargo = os.path.join(HERE, "stub_cargo.py")
    rustc = os.path.join(HERE, "stub_rustc.py")
    closure = [["bin/cargo", "file", os.path.getsize(cargo), _sha_file(cargo)],
               ["bin/rustc", "file", os.path.getsize(rustc), _sha_file(rustc)]]
    return {
        "cargo_launcher": {"path": cargo, "symlink_target": None, "sha256": _sha_file(cargo)},
        "rustc_launcher": {"path": rustc, "symlink_target": None, "sha256": _sha_file(rustc)},
        "sysroot": HERE, "cargo_path": cargo, "cargo_sha256": _sha_file(cargo),
        "cargo_version": CARGO_VERSION, "rustc_path": rustc, "rustc_sha256": _sha_file(rustc),
        "rustc_vV": RUSTC_VV, "host_triple": HOST_TRIPLE,
        "toolchain_closure_sha256": pc.sha256_hex(pc.canonical_json_bytes(closure)),
        "toolchain_closure": closure, "wrapper_chain": [],
    }


def recheck_toolchain(toolchain: dict) -> str:
    now = resolve_toolchain({}, {})
    if now["toolchain_closure_sha256"] != toolchain["toolchain_closure_sha256"]:
        raise pc.RunnerError("toolchain", "ToolchainChanged", "toolchain closure changed")
    return now["toolchain_closure_sha256"]


def resolve_native_tools(toolchain: dict, timing_schema: dict) -> dict:
    closure = [["cc", "/usr/bin/cc"], ["c++", "/usr/bin/c++"], ["ar", "/usr/bin/ar"],
               ["ranlib", "/usr/bin/ranlib"], ["sdk", "15.4"]]
    return {"cc": "/usr/bin/cc", "cxx": "/usr/bin/c++", "ar": "/usr/bin/ar",
            "ranlib": "/usr/bin/ranlib", "linker_var_name": LINKER_VAR, "linker": "/usr/bin/cc",
            "sdk": {"developer_dir": "/Library/Developer/CommandLineTools", "version": "15.4",
                    "sdkroot": "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"},
            "native_closure_sha256": pc.sha256_hex(pc.canonical_json_bytes(closure))}


def recheck_native_tools(native: dict) -> str:
    now = resolve_native_tools({}, {})
    if now["native_closure_sha256"] != native["native_closure_sha256"]:
        raise pc.RunnerError("toolchain", "NativeClosureChanged", "native closure changed")
    return now["native_closure_sha256"]


def closed_environment(role: dict, toolchain: dict, native: dict, cargo_home: str, home: str,
                       target_dir: str, temp_dir: str, threads: int, timing_schema: dict) -> dict:
    env = {
        "HOME": home,
        "PATH": "%s:%s:/usr/bin:/bin" % (os.path.dirname(sys.executable),
                                         os.path.join(toolchain["sysroot"], "bin")),
        "CARGO_HOME": cargo_home, "TMPDIR": temp_dir, "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        "NO_COLOR": "1", "CARGO_TERM_COLOR": "never", "RUSTC": toolchain["rustc_path"],
        "RUSTFLAGS": pc.RUSTFLAGS, "CARGO_TARGET_DIR": target_dir,
        native["linker_var_name"]: native["linker"], "CC": native["cc"], "CXX": native["cxx"],
        "AR": native["ar"], "RANLIB": native["ranlib"], "RAYON_NUM_THREADS": str(threads),
        "CC_ENABLE_DEBUG_OUTPUT": "1", "CARGO_NET_OFFLINE": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, name in STUB_ENV_NAMES.items():
        if STUB_SETTINGS.get(key):
            env[name] = STUB_SETTINGS[key]
    # The stub bench stands in for both backends' binary in tests: tell it which role it is.
    env["STUB_SYSTEM_ROLE"] = role.get("name", "hyrax")
    return env


def identity_projection(env: dict, roots: dict) -> dict:
    ordered = sorted(((r, t) for t, r in roots.items() if r), key=lambda x: -len(x[0]))
    out = {}
    for name, value in env.items():
        if name.startswith("STUB_"):
            continue
        projected = value
        for root, token in ordered:
            if value == root or value.startswith(root.rstrip("/") + "/"):
                projected = token + value[len(root.rstrip("/")):]
                break
        out[name] = projected
    return out


# --- cargo config discovery ---------------------------------------------------------------

def _config_candidates(invocation_dir: str, cargo_home: str) -> list:
    dirs = []
    d = invocation_dir
    while True:
        dirs.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    out = []
    for base in dirs:
        for name in ("config.toml", "config"):
            out.append(os.path.join(base, ".cargo", name))
    for name in ("config.toml", "config"):
        out.append(os.path.join(cargo_home, name))
    return out


def _candidate_state(path: str) -> dict:
    if os.path.islink(path):
        raise pc.RunnerError("toolchain", "CargoConfigRejected", "%s is a symlink" % path)
    if os.path.isfile(path):
        raise pc.RunnerError("toolchain", "CargoConfigRejected",
                             "unexpected Cargo config %s (stub allows none)" % path)
    return {"path": path, "exists": False, "kind": None, "sha256": None}


def cargo_config_audit(invocation_dir: str, cargo_home: str, timing_schema: dict, repo=None,
                       rustflags=None) -> dict:
    candidates = [_candidate_state(p) for p in _config_candidates(invocation_dir, cargo_home)]
    audit = {"schema": "zinc-plus/cargo-config-audit/v1", "python": sys.version.split()[0],
             "invocation_dir": invocation_dir, "cargo_home": cargo_home,
             "candidates": candidates, "discovered": [], "parsed": {}, "origins": {}}
    audit["audit_id"] = pc.sha256_hex(pc.canonical_json_bytes(
        {"candidates": candidates, "parsed": {}}))
    return audit


def recheck_cargo_config(audit: dict, timing_schema=None) -> str:
    now = cargo_config_audit(audit["invocation_dir"], audit["cargo_home"], {})
    if now["audit_id"] != audit["audit_id"]:
        raise pc.RunnerError("toolchain", "CargoConfigChanged", "Cargo configuration changed")
    return now["audit_id"]


# --- dependency sources -------------------------------------------------------------------

def dependency_source_audit(repo: str, snapshot: dict, env: dict, toolchain: dict,
                            cargo_home: str, metadata_target: str, features: list,
                            timing_schema: dict) -> dict:
    with open(os.path.join(repo, "Cargo.lock"), "r", encoding="utf-8") as f:
        packages = pc.parse_cargo_lock(f.read())
    entries = []
    closure = []
    for name in sorted(packages):
        for pkg in packages[name]:
            source = pkg.get("source")
            kind = "path" if source is None else ("git" if source.startswith("git+") else
                                                   "registry")
            proof = None
            if kind == "registry":
                proof = {"archive": "<CARGO_HOME>/registry/cache/index/%s-%s.crate" %
                         (name, pkg["version"]), "checksum": pkg.get("checksum"),
                         "unpacked": "<CARGO_HOME>/registry/src/index/%s-%s" %
                         (name, pkg["version"]), "cargo_ok": '{"v":1}'}
                closure.append(["<CARGO_HOME>/registry/src/index/%s-%s/.cargo-ok" %
                                (name, pkg["version"]), "file", "644", 7,
                                pc.sha256_hex(b'{"v":1}')])
            entries.append({"name": name, "version": pkg.get("version"), "source": source,
                            "checksum": pkg.get("checksum"), "kind": kind, "proof": proof})
    document = {"schema": "zinc-plus/dependency-sources/v1", "host_triple": HOST_TRIPLE,
                "features": list(features), "lock_sha256": snapshot["lock_sha256"],
                "commit": snapshot["commit"], "packages": entries, "closure": closure,
                "tokens": ["<SOURCE>", "<CARGO_HOME>", "<METADATA_TARGET>"],
                "bootstrap_command": ["cargo", "metadata", "--locked", "--offline",
                                      "--format-version", "1", "--filter-platform",
                                      HOST_TRIPLE] + pc.bootstrap_feature_tokens(features)}
    data = pc.canonical_json_bytes(document)
    return {"document": document, "bytes": data, "dependency_source_id": pc.sha256_hex(data),
            "closure_sha256": pc.sha256_hex(pc.canonical_json_bytes(closure)),
            "metadata_normalized_sha256": pc.sha256_hex(pc.canonical_json_bytes(
                [e["name"] + "@" + str(e["version"]) for e in entries])),
            "raw_locators": {"cargo_home": cargo_home, "metadata_target": metadata_target,
                             "closure_sha256": pc.sha256_hex(pc.canonical_json_bytes(closure))}}


def rehash_dependency_closure(audit: dict) -> str:
    expected = audit.get("closure_sha256")
    actual = audit.get("raw_locators", {}).get("closure_sha256", expected)
    if actual != expected:
        raise pc.RunnerError("dependency", "DependencyClosureChanged",
                             "dependency closure changed")
    return actual


def replay_metadata(audit: dict, env: dict, metadata_target=None) -> None:
    return None


# --- evidence build ---------------------------------------------------------------------

def evidence_build(repo: str, env: dict, target_dir: str, features: list, role: str,
                   timing_schema: dict, dependency_audit=None) -> dict:
    if role not in pc.SYSTEM_ROLES:
        raise pc.RunnerError("evidence", "EvidenceBuildFailed", "unknown role %r" % (role,))
    CREATED_TARGETS.append(target_dir)
    argv = pc.evidence_command(features)
    argv[0] = os.path.join(HERE, "stub_cargo.py")
    proc = subprocess.run(argv, cwd=repo, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise pc.RunnerError("evidence", "EvidenceBuildFailed", "stub evidence build exited %d"
                             % proc.returncode)
    artifact = None
    for line in proc.stdout.decode("utf-8", "replace").split("\n"):
        if line.startswith("{"):
            msg = json.loads(line)
            if msg.get("reason") == "compiler-artifact" and msg.get("executable"):
                artifact = msg
    if artifact is None:
        raise pc.RunnerError("evidence", "EvidenceBuildFailed", "no bench compiler-artifact")
    exe = artifact["executable"]
    rel = os.path.relpath(exe, target_dir).replace(os.sep, "/")
    if rel.startswith("..") or os.path.islink(exe):
        raise pc.RunnerError("evidence", "EvidenceBuildFailed", "executable outside the target")
    argv_norm = list(pc.evidence_command(features))
    argv_norm[0] = "<SYSROOT>/bin/cargo"
    rustc_argv = list(STUB_RUSTC_ARGV)
    return {"log_bytes": log, "log_sha256": pc.sha256_hex(log),
            "executable": {"relative_path": rel, "size": os.path.getsize(exe),
                           "sha256": _sha_file(exe)},
            "argv": pc.evidence_command(features), "argv_normalized": argv_norm,
            "rustc_program": "<SYSROOT>/bin/rustc", "rustc_argv_normalized": rustc_argv,
            "rustc_argv_raw_sha256": pc.sha256_hex(" ".join(rustc_argv).encode()),
            "native_lines": [], "native_env_reads": [], "manifest_dirs": ["<SOURCE>"],
            "source_root_policy": "repository or audited dependency root",
            "zstd_sys": {"bundled_compile_lines": 0, "linked_libs": None, "required": False},
            "artifact": {"target_kind": ["bench"], "target_name": pc.BENCH_NAME,
                         "fresh": False},
            "exit": proc.returncode}


def build_profile_document(evidence: dict, snapshot: dict, dependency: dict, cargo_config: dict,
                           toolchain: dict, native: dict, env_raw: dict, roots: dict,
                           rejected_audit: dict, compiled_metadata: dict, timing_schema: dict,
                           features: list, role: str = "hyrax") -> dict:
    """Same signature as the landed package C function; a reduced document."""
    probes = {k: compiled_metadata.get(k) for k in ("panic_strategy", "debug_assertions",
                                                     "overflow_checks")}
    proj = lambda v: identity_projection({"v": v}, roots)["v"]  # noqa: E731
    return {"schema": "zinc-plus/build-profile/v1", "role": role, "features": list(features),
            "log_sha256": evidence["log_sha256"], "command": evidence["argv_normalized"],
            "exit": evidence["exit"],
            "dependency_source_id": dependency["dependency_source_id"],
            "dependency_sources_sha256": pc.sha256_hex(dependency["bytes"]),
            "dependency_closure_sha256": dependency["closure_sha256"],
            "source": {"commit": snapshot["commit"], "tree": snapshot["tree"],
                       "source_snapshot_id": snapshot["source_snapshot_id"],
                       "dirty": snapshot["dirty"], "lock_sha256": snapshot["lock_sha256"],
                       "closure_sha256": snapshot["closure_sha256"]},
            "target": {"path": "<ROLE_TARGET>", "initially_empty": True, "symlink": False},
            "toolchain": {"cargo_path": proj(toolchain["cargo_path"]),
                          "cargo_sha256": toolchain["cargo_sha256"],
                          "cargo_version": toolchain["cargo_version"],
                          "rustc_path": proj(toolchain["rustc_path"]),
                          "rustc_sha256": toolchain["rustc_sha256"],
                          "sysroot": proj(toolchain["sysroot"]),
                          "toolchain_closure_sha256": toolchain["toolchain_closure_sha256"]},
            "native": {k: v for k, v in native.items()},
            "environment": {"raw": dict(env_raw), "projected": identity_projection(env_raw, roots)},
            "cargo_config": {"audit_id": cargo_config["audit_id"]},
            "rustflags": env_raw["RUSTFLAGS"], "rejected_inputs": rejected_audit,
            "wrapper_chain": [],
            "bench_rustc_invocation": {"program": evidence["rustc_program"],
                                       "argv_normalized": evidence["rustc_argv_normalized"],
                                       "raw_sha256": evidence["rustc_argv_raw_sha256"]},
            "native_lines": evidence["native_lines"], "executable": dict(evidence["executable"]),
            "recheck_policy": ["before and after metadata, each gate, the preflight and every "
                               "child; after the terminal attempt"],
            "probes": probes}


def rehash_executable(target_dir: str, executable: dict) -> str:
    path = os.path.join(target_dir, *executable["relative_path"].split("/"))
    if not os.path.isfile(path) or os.path.islink(path) or \
            os.path.getsize(path) != executable["size"] or _sha_file(path) != executable["sha256"]:
        raise pc.RunnerError("evidence", "ExecutableChanged",
                             "evidenced executable %s changed" % executable["relative_path"])
    return executable["sha256"]


def rehash_all(audit_bundle: dict) -> dict:
    recheck_source_snapshot(audit_bundle["repo"], audit_bundle["snapshot"])
    executable = audit_bundle.get("executable")
    return {"source": audit_bundle["snapshot"]["source_snapshot_id"],
            "dependency": rehash_dependency_closure(audit_bundle["dependency"]),
            "cargo_config": recheck_cargo_config(audit_bundle["cargo_config"]),
            "toolchain": recheck_toolchain(audit_bundle["toolchain"]),
            "native": recheck_native_tools(audit_bundle["native"]),
            "executable": rehash_executable(audit_bundle["target_dir"], executable)
            if executable is not None else None}


# --- subprocess wrapper -----------------------------------------------------------------

def run_cargo_subprocess(argv: list, env: dict, cwd: str, audit_bundle: dict, log_sink) -> dict:
    pre = rehash_all(audit_bundle)
    start_utc = pc.utc_rfc3339_ns()
    start_mono = time.monotonic_ns()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    end_mono = time.monotonic_ns()
    end_utc = pc.utc_rfc3339_ns()
    post = rehash_all(audit_bundle)
    if callable(log_sink):
        log_sink("stdout", out)
        log_sink("stderr", err)
    else:
        pc.write_bytes(log_sink["stdout"], out)
        pc.write_bytes(log_sink["stderr"], err)
    rc = proc.returncode
    return {"argv": list(argv), "exit": rc if rc >= 0 else None,
            "signal": -rc if rc < 0 else None,
            "pid": proc.pid, "start_utc": start_utc, "end_utc": end_utc,
            "start_mono": start_mono, "end_mono": end_mono, "pre": pre, "post": post,
            "stdout_sha256": pc.sha256_hex(out), "stderr_sha256": pc.sha256_hex(err)}
