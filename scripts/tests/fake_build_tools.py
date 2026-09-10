#!/usr/bin/env python3
"""Fixture builder and stub cargo/rustc for the poseidon_build_env tests (limber).

As a library: `build_fixture(root, grammar_tokens)` creates, below `root`,
  home/.cargo/bin/{rustup,cargo->rustup,rustc->rustup}   PATH launchers (symlinked, like rustup)
  sysroot/bin/{cargo,rustc}, sysroot/lib/...              the "toolchain" (sh wrappers)
  cargo-home/registry/{cache,src}/index.crates.io-.../    one registry crate (zstd-sys)
  cargo-home/git/checkouts/fakegit-<hash>/<short>/        one git checkout with .cargo-ok
  repo/                                                   a committed single-package crate
                                                          (`limber`, bench `poseidon_modp`)
  sysroot/fake-control.json                               what the stub cargo prints
and returns the paths plus the control object (tests mutate it and call `write_control`).

As a script (invoked by the sh wrappers): `fake_build_tools.py cargo <sysroot> ARGS...` and
`fake_build_tools.py rustc <sysroot> ARGS...` implement `cargo -V`, `cargo metadata`, the
evidence build (`cargo bench ... --no-run -vv`), `rustc -vV` and `rustc --print sysroot`.
Placeholders in the control file are substituted from the closed environment at run time:
`<TARGET>`, `<RUSTC>`, `<CC>`, `<CXX>`, `<AR>`, `<RANLIB>`, `<CARGO_HOME>`, `<SOURCE>` (cwd).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import stat
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
CARGO_VERSION = "cargo 1.98.1 (797e8a9bc 2026-08-05)"
RUSTC_VV = ("rustc 1.98.1 (48a229cea 2026-09-01)\nbinary: rustc\ncommit-hash: "
            "48a229ceaefd4985c50990b14116b6d856af0985\ncommit-date: 2026-09-01\n"
            "host: aarch64-apple-darwin\nrelease: 1.98.1\nLLVM version: 22.1.8")
CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
INDEX = "index.crates.io-1949cf8c6b5b557f"
CRATE_NAME, CRATE_VERSION = "zstd-sys", "2.1.0+zstd.1.5.7"
GIT_NAME, GIT_URL = "fakegit", "https://example.invalid/fakegit.git"
GIT_DIR_HASH = "0123456789abcdef"
PACKAGE = "limber"
BENCH = "poseidon_modp"
BENCH_SRC = "benches/poseidon_modp.rs"
PROFILE_BLOCK = ('[profile.bench]\nopt-level = 3\ndebug = false\nstrip = "none"\n'
                 'debug-assertions = false\noverflow-checks = false\nlto = "fat"\n'
                 'codegen-units = 1\nincremental = false\n')
EXECUTABLE_REL = "release/deps/poseidon_modp-62c678ee63335c9b"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: str, data, mode=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
    if mode is not None:
        os.chmod(path, mode)


def git(cwd, *args) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          cwd=cwd, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.decode().strip()


def _sh_wrapper(path: str, body: str) -> None:
    write(path, "#!/bin/sh\n" + body + "\n", 0o755)


# ---------------------------------------------------------------------------
# Fixture construction.
# ---------------------------------------------------------------------------

def make_crate(cache_dir: str, src_dir: str) -> dict:
    """A tiny `.crate` (regular files, one executable, one safe symlink), unpacked by hand."""
    prefix = "%s-%s" % (CRATE_NAME, CRATE_VERSION)
    files = {
        "Cargo.toml": ('[package]\nname = "%s"\nversion = "%s"\nedition = "2018"\nlinks = '
                       '"zstd"\n\n[features]\ndefault = ["legacy", "zdict_builder"]\nlegacy = []'
                       '\nzdict_builder = []\nstd = []\npkg-config = []\n' %
                       (CRATE_NAME, CRATE_VERSION), 0o644),
        "build.rs": ("fn main() {}\n", 0o644),
        "src/lib.rs": ("pub fn zstd() -> u32 { 1 }\n", 0o644),
        "scripts/gen.sh": ("#!/bin/sh\necho gen\n", 0o755),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel, (text, mode) in files.items():
            data = text.encode()
            info = tarfile.TarInfo(prefix + "/" + rel)
            info.size = len(data)
            info.mode = mode
            info.mtime = 1153000000
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo(prefix + "/src/alias.rs")
        link.type = tarfile.SYMTYPE
        link.linkname = "lib.rs"
        link.mtime = 1153000000
        link.mode = 0o777
        tf.addfile(link)
    archive = buf.getvalue()
    archive_path = os.path.join(cache_dir, INDEX, prefix + ".crate")
    write(archive_path, archive)
    root = os.path.join(src_dir, INDEX, prefix)
    for rel, (text, mode) in files.items():
        write(os.path.join(root, rel), text, mode)
    os.symlink("lib.rs", os.path.join(root, "src", "alias.rs"))
    write(os.path.join(root, ".cargo-ok"), b'{"v":1}')
    return {"archive": archive_path, "root": root, "checksum": sha256(archive), "prefix": prefix}


def make_git_checkout(cargo_home: str) -> dict:
    tmp = os.path.join(cargo_home, "git", "checkouts", "%s-%s" % (GIT_NAME, GIT_DIR_HASH), "tmp")
    os.makedirs(tmp)
    write(os.path.join(tmp, "Cargo.toml"),
          '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2021"\n' % GIT_NAME)
    write(os.path.join(tmp, "src", "lib.rs"), "pub fn g() {}\n")
    write(os.path.join(tmp, "tool.sh"), "#!/bin/sh\n", 0o755)
    git(tmp, "init", "-q")
    git(tmp, "add", "-A")
    git(tmp, "commit", "-q", "-m", "fixture")
    commit = git(tmp, "rev-parse", "HEAD")
    root = os.path.join(os.path.dirname(tmp), commit[:7])
    os.rename(tmp, root)
    write(os.path.join(root, ".cargo-ok"), b"")
    source = "git+%s?rev=%s#%s" % (GIT_URL, commit[:7], commit)
    return {"root": root, "commit": commit, "short": commit[:7], "source": source}


def make_repo(root: str, crate: dict, checkout: dict) -> str:
    """A committed single-package crate shaped like limber (root package, `benches/`)."""
    repo = os.path.join(root, "repo")
    os.makedirs(repo)
    write(os.path.join(repo, "Cargo.toml"),
          '[package]\nname = "%s"\nversion = "0.1.0"\nedition = "2024"\n\n[dependencies]\n'
          '%s = "%s"\n%s = { git = "%s", rev = "%s" }\n\n[dev-dependencies]\n'
          'criterion = { version = "=0.7.0", default-features = false, features = '
          '["cargo_bench_support"] }\n\n[[bench]]\nname = "%s"\nharness = false\ntest = false\n\n'
          '[profile.release]\nlto = true\n\n' % (PACKAGE, CRATE_NAME, CRATE_VERSION, GIT_NAME,
                                                 GIT_URL, checkout["short"], BENCH)
          + PROFILE_BLOCK)
    lock = ('# This file is automatically @generated by Cargo.\nversion = 4\n\n'
            '[[package]]\nname = "%s"\nversion = "0.1.0"\nsource = "%s"\n\n'
            '[[package]]\nname = "%s"\nversion = "0.1.0"\ndependencies = [\n'
            ' "%s",\n "%s",\n]\n\n'
            '[[package]]\nname = "%s"\nversion = "%s"\nsource = "%s"\nchecksum = "%s"\n' %
            (GIT_NAME, checkout["source"], PACKAGE, GIT_NAME, CRATE_NAME, CRATE_NAME,
             CRATE_VERSION, CRATES_IO, crate["checksum"]))
    write(os.path.join(repo, "Cargo.lock"), lock)
    write(os.path.join(repo, "src", "lib.rs"), "pub fn limber() {}\n")
    write(os.path.join(repo, BENCH_SRC), "fn main() {}\n")
    write(os.path.join(repo, ".gitignore"), "ignored.txt\n")
    write(os.path.join(repo, "tool.sh"), "#!/bin/sh\n", 0o755)
    os.symlink("src/lib.rs", os.path.join(repo, "lib-link.rs"))
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixture")
    return repo


def package_id(repo: str) -> str:
    return "path+file://%s#%s@0.1.0" % (repo, PACKAGE)


def make_metadata(repo: str, crate: dict, checkout: dict) -> dict:
    pkg_id = package_id(repo)
    crate_id = "%s#%s@%s" % (CRATES_IO, CRATE_NAME, CRATE_VERSION)
    git_id = "git+%s?rev=%s#%s@0.1.0" % (GIT_URL, checkout["short"], GIT_NAME)

    def target(kind, name, src):
        return {"kind": [kind], "crate_types": ["bin" if kind != "lib" else "lib"], "name": name,
                "src_path": src, "edition": "2024", "doc": True, "doctest": False, "test": True}
    packages = [
        {"name": PACKAGE, "version": "0.1.0", "id": pkg_id, "source": None,
         "dependencies": [
             {"name": CRATE_NAME, "source": CRATES_IO, "req": "^2.1.0", "kind": None,
              "rename": None, "optional": False, "uses_default_features": True, "features": [],
              "target": None, "registry": None},
             {"name": GIT_NAME, "source": "git+%s?rev=%s" % (GIT_URL, checkout["short"]),
              "req": "*", "kind": None, "rename": None, "optional": False,
              "uses_default_features": True, "features": [], "target": None, "registry": None}],
         "targets": [target("lib", PACKAGE, repo + "/src/lib.rs"),
                     target("bench", BENCH, repo + "/" + BENCH_SRC)],
         "features": {},
         "manifest_path": repo + "/Cargo.toml", "metadata": None, "publish": [],
         "authors": [], "categories": [], "keywords": [], "readme": None, "repository": None,
         "homepage": None, "documentation": None, "edition": "2024", "links": None,
         "default_run": None, "rust_version": None},
        {"name": CRATE_NAME, "version": CRATE_VERSION, "id": crate_id, "source": CRATES_IO,
         "dependencies": [], "targets": [target("lib", "zstd_sys", crate["root"] + "/src/lib.rs")],
         "features": {"default": ["legacy", "zdict_builder"], "legacy": [], "zdict_builder": [],
                      "std": [], "pkg-config": []},
         "manifest_path": crate["root"] + "/Cargo.toml", "metadata": None, "publish": None,
         "authors": [], "categories": [], "keywords": [], "readme": None, "repository": None,
         "homepage": None, "documentation": None, "edition": "2018", "links": "zstd",
         "default_run": None, "rust_version": None},
        {"name": GIT_NAME, "version": "0.1.0", "id": git_id, "source": checkout["source"],
         "dependencies": [], "targets": [target("lib", GIT_NAME, checkout["root"] + "/src/lib.rs")],
         "features": {}, "manifest_path": checkout["root"] + "/Cargo.toml", "metadata": None,
         "publish": None, "authors": [], "categories": [], "keywords": [], "readme": None,
         "repository": None, "homepage": None, "documentation": None, "edition": "2021",
         "links": None, "default_run": None, "rust_version": None},
    ]
    nodes = [
        {"id": pkg_id, "dependencies": [crate_id, git_id],
         "deps": [{"name": "zstd_sys", "pkg": crate_id, "dep_kinds": [{"kind": None, "target": None}]},
                  {"name": GIT_NAME, "pkg": git_id, "dep_kinds": [{"kind": None, "target": None}]}],
         "features": []},
        {"id": crate_id, "dependencies": [], "deps": [], "features": ["legacy", "std", "zdict_builder"]},
        {"id": git_id, "dependencies": [], "deps": [], "features": []},
    ]
    return {"packages": packages, "workspace_members": [pkg_id],
            "workspace_default_members": [pkg_id],
            "resolve": {"nodes": nodes, "root": pkg_id}, "target_directory": "<TARGET>",
            "build_directory": "<TARGET>", "version": 1, "workspace_root": repo,
            "metadata": None}


def evidence_control(grammar_tokens: list, repo: str, crate: dict, checkout: dict) -> dict:
    """The stub evidence build: one bench compiler-artifact, build-script lines naming the
    injected CC/AR, the bench rustc Running record (the pinned grammar with real-looking
    ephemeral values), and build-script Running records with CARGO_MANIFEST_DIRs."""
    def hexes(tok):
        return (tok.replace("<ROLE_TARGET>", "<TARGET>").replace("<META>", "7ce1867fa6d06281")
                .replace("<EXTRA>", "62c678ee63335c9b").replace("<HASH>", "3634f7a5d690ea67")
                .replace("<NATIVE_CC>", "<CC>"))
    rustc_args = [hexes(t) for t in grammar_tokens]
    pkg_id = package_id(repo)
    artifact = {"reason": "compiler-artifact", "package_id": pkg_id,
                "manifest_path": repo + "/Cargo.toml",
                "target": {"kind": ["bench"], "crate_types": ["bin"], "name": BENCH,
                           "src_path": repo + "/" + BENCH_SRC, "edition": "2024",
                           "doc": False, "doctest": False, "test": False},
                "profile": {"opt_level": "3", "debuginfo": 0, "debug_assertions": False,
                            "overflow_checks": False, "test": True},
                "features": [], "filenames": ["<TARGET>/" + EXECUTABLE_REL],
                "executable": "<TARGET>/" + EXECUTABLE_REL, "fresh": False}
    lib_artifact = {"reason": "compiler-artifact", "package_id": "%s#%s@%s" % (CRATES_IO, CRATE_NAME, CRATE_VERSION),
                    "manifest_path": crate["root"] + "/Cargo.toml",
                    "target": {"kind": ["lib"], "crate_types": ["lib"], "name": "zstd_sys",
                               "src_path": crate["root"] + "/src/lib.rs", "edition": "2018",
                               "doc": True, "doctest": False, "test": True},
                    "profile": {"opt_level": "3", "debuginfo": 0, "debug_assertions": False,
                                "overflow_checks": False, "test": False},
                    "features": ["legacy", "std", "zdict_builder"],
                    "filenames": ["<TARGET>/release/deps/libzstd_sys-1111111111111111.rlib"],
                    "executable": None, "fresh": False}
    out = "<TARGET>/release/build/zstd-sys-ceb0131b2900a24c/out"
    pkg = "[zstd-sys 2.1.0+zstd.1.5.7] "
    stdout = [
        json.dumps(lib_artifact, separators=(",", ":")),
        pkg + "cargo:rerun-if-env-changed=ZSTD_SYS_USE_PKG_CONFIG",
        pkg + "CC_aarch64-apple-darwin = None",
        pkg + "HOST_CC = None",
        pkg + 'CC = Some("<CC>")',
        pkg + 'running: env -u CC_SHIM_OUT_DIR -u CC_SHIM_OUT_FILES "<CC>" "-E" "%s/1detect_compiler_family.c"' % out,
        pkg + "exit status: 0",
        pkg + 'running: "xcrun" "--show-sdk-version" "--sdk" "macosx"',
        pkg + 'running: cd "%s" && env -u CC_SHIM_OUT_DIR LC_ALL="C" "<CC>" "-O0" "-o" "%s/flag_check" "-c" "%s/flag_check.c"' % (out, out, out),
        pkg + 'running: env -u IPHONEOS_DEPLOYMENT_TARGET LC_ALL="C" "<CC>" "-O3" "-fPIC" "--target=arm64-apple-macosx" "-o" "%s/44ff4c55aa9e5133-debug.o" "-c" "zstd/lib/common/debug.c"' % out,
        pkg + 'AR = Some("<AR>")',
        pkg + 'running: ZERO_AR_DATE="1" "<AR>" "cq" "%s/libzstd.a" "%s/44ff4c55aa9e5133-debug.o"' % (out, out),
        pkg + "cargo:rustc-link-lib=static=zstd",
        pkg + "cargo:rustc-link-search=native=" + out,
        json.dumps({"reason": "build-script-executed",
                    "package_id": "%s#%s@%s" % (CRATES_IO, CRATE_NAME, CRATE_VERSION),
                    "linked_libs": ["static=zstd"], "linked_paths": ["native=" + out],
                    "cfgs": [], "env": [], "out_dir": out}, separators=(",", ":")),
        json.dumps(artifact, separators=(",", ":")),
        json.dumps({"reason": "build-finished", "success": True}, separators=(",", ":")),
    ]
    common_env = {"CARGO": "<SYSROOT>/bin/cargo", "CARGO_PKG_VERSION": "0.1.0",
                  "CARGO_PKG_DESCRIPTION": "A `weird` description\nwith two lines"}
    records = [
        {"env": dict(common_env, CARGO_CRATE_NAME="zstd_sys", CARGO_MANIFEST_DIR=crate["root"]),
         "program": "<RUSTC>", "args": ["--crate-name", "zstd_sys", "--edition=2018",
                                         crate["root"] + "/src/lib.rs"]},
        {"env": dict(common_env, CARGO_MANIFEST_DIR=crate["root"], OUT_DIR=out),
         "program": "<TARGET>/release/build/zstd-sys-1111111111111111/build-script-build",
         "args": []},
        {"env": dict(common_env, CARGO_CRATE_NAME=GIT_NAME, CARGO_MANIFEST_DIR=checkout["root"]),
         "program": "<RUSTC>", "args": ["--crate-name", GIT_NAME, checkout["root"] + "/src/lib.rs"]},
        {"env": dict(common_env, CARGO_CRATE_NAME=PACKAGE, CARGO_MANIFEST_DIR=repo,
                     CARGO_PRIMARY_PACKAGE="1"),
         "program": "<RUSTC>", "args": ["--crate-name", PACKAGE, "--edition=2024",
                                         repo + "/src/lib.rs"]},
        {"env": dict(common_env, CARGO_CRATE_NAME=BENCH,
                     CARGO_MANIFEST_DIR=repo, CARGO_PRIMARY_PACKAGE="1"),
         "program": "<RUSTC>", "args": rustc_args},
    ]
    return {"stdout": stdout, "records": records, "executable": EXECUTABLE_REL,
            "executable_bytes": "fake bench executable\n", "exit": 0,
            "stderr_prefix": ["   Compiling zstd-sys v2.1.0+zstd.1.5.7",
                              "warning: stripping debug info with `rust-objcopy` failed"],
            "stderr_suffix": ["    Finished `bench` profile [optimized] target(s) in 0.01s"]}


def build_fixture(root: str, grammar_tokens: list, python: str | None = None) -> dict:
    python = sys.executable if python is None else python
    home = os.path.join(root, "home")
    cargo_home = os.path.join(root, "cargo-home")
    sysroot = os.path.join(root, "sysroot")
    script = os.path.join(HERE, "fake_build_tools.py")
    for tool in ("cargo", "rustc"):
        _sh_wrapper(os.path.join(sysroot, "bin", tool),
                    'exec %s -B %s %s %s "$@"' % (shlex.quote(python), shlex.quote(script), tool,
                                                  shlex.quote(sysroot)))
    write(os.path.join(sysroot, "lib", "libstd-fake.dylib"), os.urandom(64))
    write(os.path.join(sysroot, "lib", "rustlib", "aarch64-apple-darwin", "lib",
                       "libcore-fake.rlib"), os.urandom(64))
    launchers = os.path.join(home, ".cargo", "bin")
    _sh_wrapper(os.path.join(launchers, "rustup"),
                'exec %s/bin/"$(basename "$0")" "$@"' % shlex.quote(sysroot))
    for tool in ("cargo", "rustc"):
        os.symlink("rustup", os.path.join(launchers, tool))
    crate = make_crate(os.path.join(cargo_home, "registry", "cache"),
                       os.path.join(cargo_home, "registry", "src"))
    checkout = make_git_checkout(cargo_home)
    repo = make_repo(root, crate, checkout)
    control = {"cargo_version": CARGO_VERSION, "rustc_vV": RUSTC_VV,
               "metadata": make_metadata(repo, crate, checkout), "metadata_second": None,
               "evidence": evidence_control(grammar_tokens, repo, crate, checkout)}
    fixture = {"root": root, "home": home, "cargo_home": cargo_home, "sysroot": sysroot,
               "launchers": launchers, "repo": repo, "crate": crate, "checkout": checkout,
               "control": control, "control_path": os.path.join(sysroot, "fake-control.json"),
               "path": "%s:/usr/bin:/bin" % launchers}
    write_control(fixture)
    return fixture


def write_control(fixture: dict) -> None:
    write(fixture["control_path"], json.dumps(fixture["control"], indent=1) + "\n")
    counter = fixture["control_path"] + ".metadata-count"
    if os.path.exists(counter):
        os.unlink(counter)


# ---------------------------------------------------------------------------
# Stub behaviour (script mode).
# ---------------------------------------------------------------------------

def _subst(text: str, env, sysroot: str) -> str:
    table = {"<TARGET>": env.get("CARGO_TARGET_DIR", ""), "<RUSTC>": env.get("RUSTC", ""),
             "<CC>": env.get("CC", ""), "<CXX>": env.get("CXX", ""), "<AR>": env.get("AR", ""),
             "<RANLIB>": env.get("RANLIB", ""), "<CARGO_HOME>": env.get("CARGO_HOME", ""),
             "<SOURCE>": os.getcwd(), "<SYSROOT>": sysroot}
    for key, value in table.items():
        text = text.replace(key, value)
    return text


def _quote_record(rec: dict, env, sysroot: str) -> str:
    parts = ["%s=%s" % (k, shlex.quote(_subst(v, env, sysroot))) for k, v in rec["env"].items()]
    parts.append(shlex.quote(_subst(rec["program"], env, sysroot)))
    parts += [shlex.quote(_subst(a, env, sysroot)) for a in rec["args"]]
    return "     Running `" + " ".join(parts) + "`"


def _stub_cargo(sysroot: str, args: list) -> int:
    with open(os.path.join(sysroot, "fake-control.json"), "rb") as f:
        control = json.loads(f.read().decode("utf-8"))
    env = os.environ
    if args == ["-V"]:
        sys.stdout.write(control["cargo_version"] + "\n")
        return 0
    if args[:1] == ["metadata"]:
        if args[1:6] != ["--locked", "--offline", "--format-version", "1", "--filter-platform"]:
            sys.stderr.write("stub cargo: bad metadata argv %r\n" % (args,))
            return 2
        if len(args) != 7:
            sys.stderr.write("stub cargo: the limber bootstrap takes no features %r\n" % (args,))
            return 2
        counter = os.path.join(sysroot, "fake-control.json.metadata-count")
        count = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
        with open(counter, "w") as f:
            f.write(str(count))
        md = control["metadata"]
        if control.get("metadata_second") is not None:
            md = control["metadata_second"]
        sys.stdout.write(_subst(json.dumps(md, separators=(",", ":")), env, sysroot) + "\n")
        return 0
    if args[:1] == ["bench"] and "--no-run" in args:
        if "-p" in args or "--features" in args or args[args.index("--bench") + 1] != BENCH:
            sys.stderr.write("stub cargo: unexpected evidence argv %r\n" % (args,))
            return 2
        ev = control["evidence"]
        target = env.get("CARGO_TARGET_DIR", "")
        exe = os.path.join(target, *ev["executable"].split("/"))
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        with open(exe, "wb") as f:
            f.write(ev["executable_bytes"].encode())
        os.chmod(exe, 0o755)
        if ev.get("executable_symlink"):
            os.rename(exe, exe + ".real")
            os.symlink(exe + ".real", exe)
        for line in ev["stdout"]:
            sys.stdout.write(_subst(line, env, sysroot) + "\n")
        for line in ev.get("stderr_prefix", []):
            sys.stderr.write(line + "\n")
        for rec in ev["records"]:
            sys.stderr.write(_quote_record(rec, env, sysroot) + "\n")
        for line in ev.get("stderr_suffix", []):
            sys.stderr.write(line + "\n")
        return int(ev.get("exit", 0))
    sys.stderr.write("stub cargo: unsupported %r\n" % (args,))
    return 2


def _stub_rustc(sysroot: str, args: list) -> int:
    with open(os.path.join(sysroot, "fake-control.json"), "rb") as f:
        control = json.loads(f.read().decode("utf-8"))
    if args == ["-vV"]:
        sys.stdout.write(control["rustc_vV"] + "\n")
        return 0
    if args == ["--print", "sysroot"]:
        sys.stdout.write(sysroot + "\n")
        return 0
    sys.stderr.write("stub rustc: unsupported %r\n" % (args,))
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(2)
    tool, sysroot_arg, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    sys.exit(_stub_cargo(sysroot_arg, rest) if tool == "cargo" else _stub_rustc(sysroot_arg, rest))
