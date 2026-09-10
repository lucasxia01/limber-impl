#!/usr/bin/env python3
"""Tests for scripts/poseidon_build_env.py (package C, limber adaptation).

Run with `python3 -B -m unittest scripts.tests.test_poseidon_build_env` from the repository
root. Every test drives the real module against a fixture built by `fake_build_tools`: a fake
sysroot with stub cargo/rustc, rustup-like symlinked launchers, a Cargo home with one registry
crate and one git checkout, and a committed single-package crate shaped like limber.

The shared timing schema pins no rustc argv grammar for the limber bench yet
(`rustc_argv_grammar.roles` has only `zinc`): the evidence build must fail closed against the
pristine schema, and the rest of the evidence checks run against a copy of the schema with a
plausible limber grammar (`LIMBER_GRAMMAR`) injected under `roles.limber`, which is the shape
the integrator will pin from a captured `-vv` build.
"""
import copy
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest

from scripts import poseidon_common as pc
from scripts import poseidon_build_env as be
from scripts.tests import fake_build_tools as fake

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_REPO = os.path.dirname(os.path.dirname(HERE))
TEST_START = time.time()
with open(os.path.join(REAL_REPO, "specs", "poseidon", "timing-schema-v2.json"), "rb") as _f:
    PRISTINE_SCHEMA = pc.load_canonical_json(_f.read(), "timing-schema-v2.json")
# The shared schema now pins `roles.limber`; the fail-closed path is exercised on a
# copy with that role removed (the state before the limber grammar was captured).
ROLELESS_SCHEMA = copy.deepcopy(PRISTINE_SCHEMA)
ROLELESS_SCHEMA["rustc_argv_grammar"]["roles"].pop("limber", None)
# A plausible normalized limber bench grammar (the integrator replaces it by the captured one
# in the shared schema; this copy only pins the shape the checks operate on).
LIMBER_GRAMMAR = [
    "--crate-name", "poseidon_modp", "--edition=2024", "benches/poseidon_modp.rs",
    "--error-format=json", "--json=diagnostic-rendered-ansi,artifacts,future-incompat",
    "--emit=dep-info,link", "-C", "opt-level=3", "-C", "lto=fat", "-C", "codegen-units=1",
    "--cfg", "test", "--check-cfg", "cfg(docsrs,test)",
    "--check-cfg", 'cfg(feature, values("default", "jemalloc"))',
    "-C", "metadata=<META>", "-C", "extra-filename=-<EXTRA>",
    "--out-dir", "<ROLE_TARGET>/release/deps", "-C", "linker=<NATIVE_CC>",
    "-L", "dependency=<ROLE_TARGET>/release/deps",
    "--extern", "criterion=<ROLE_TARGET>/release/deps/libcriterion-<HASH>.rlib",
    "--extern", "fakegit=<ROLE_TARGET>/release/deps/libfakegit-<HASH>.rlib",
    "--extern", "limber=<ROLE_TARGET>/release/deps/liblimber-<HASH>.rlib",
    "--extern", "zstd_sys=<ROLE_TARGET>/release/deps/libzstd_sys-<HASH>.rlib",
    "-C", "target-cpu=native",
    "-L", "native=<ROLE_TARGET>/release/build/zstd-sys-<HASH>/out",
]
SCHEMA = copy.deepcopy(PRISTINE_SCHEMA)
SCHEMA["rustc_argv_grammar"]["roles"][pc.GRAMMAR_ROLE] = {
    "crate_name": pc.BENCH_NAME,
    "features": {"none": {"tokens": LIMBER_GRAMMAR, "token_count": len(LIMBER_GRAMMAR),
                          "source": "test fixture (not a captured build)"}}}
GRAMMAR = LIMBER_GRAMMAR
FAKE_NATIVE = {"cc": "/usr/bin/cc", "cxx": "/usr/bin/c++", "ar": "/usr/bin/ar",
               "ranlib": "/usr/bin/ranlib", "linker": "/usr/bin/cc",
               "linker_var_name": "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER",
               "native_closure_sha256": "0" * 64, "sdk": {"version": "15.4"}}
COMPILED = {"panic_strategy": "unwind", "debug_assertions": False, "overflow_checks": False}


def new_fixture(case) -> dict:
    root = os.path.realpath(tempfile.mkdtemp(prefix="poseidon-be-"))
    case.addCleanup(shutil.rmtree, root, True)
    return fake.build_fixture(root, GRAMMAR)


def fresh_dir(case, root, prefix) -> str:
    path = os.path.realpath(tempfile.mkdtemp(prefix=prefix, dir=root))
    case.addCleanup(shutil.rmtree, path, True)
    return path


class BuildEnvBase(unittest.TestCase):
    """One shared fixture; tests that mutate it restore it through cleanups."""

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.realpath(tempfile.mkdtemp(prefix="poseidon-be-"))
        cls.fx = fake.build_fixture(cls.root, GRAMMAR)
        cls.repo = cls.fx["repo"]
        cls.env0 = {"PATH": cls.fx["path"], "HOME": cls.fx["home"]}
        cls.toolchain = be.resolve_toolchain(cls.env0, SCHEMA, cwd=cls.repo)
        cls.snapshot = be.source_snapshot(cls.repo, False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def closed_env(self, target_dir=None, threads=1):
        target_dir = target_dir or fresh_dir(self, self.root, "target-")
        temp_dir = fresh_dir(self, self.root, "temp-")
        return be.closed_environment({"name": "hyrax", "system": "limber"}, self.toolchain,
                                     FAKE_NATIVE, self.fx["cargo_home"], self.fx["home"],
                                     target_dir, temp_dir, threads, SCHEMA)

    def dependency_audit(self, env=None):
        env = env or self.closed_env()
        meta = fresh_dir(self, self.root, "meta-")
        return be.dependency_source_audit(self.repo, self.snapshot, env, self.toolchain,
                                          self.fx["cargo_home"], meta, [], SCHEMA)

    def set_control(self, mutate):
        control = self.fx["control"]
        saved = json.loads(json.dumps(control))
        mutate(control)
        fake.write_control(self.fx)

        def restore():
            control.clear()
            control.update(saved)
            fake.write_control(self.fx)
        self.addCleanup(restore)

    def add_file(self, path, data=b"x\n", mode=None):
        fake.write(path, data, mode)
        self.addCleanup(lambda: os.path.lexists(path) and os.unlink(path))

    def assertRaisesCode(self, code, fn, *args, **kwargs):
        with self.assertRaises(pc.RunnerError) as ctx:
            fn(*args, **kwargs)
        self.assertEqual(ctx.exception.error_code, code, ctx.exception)
        return ctx.exception


class SourceSnapshotTest(BuildEnvBase):
    def test_clean_snapshot_accepted(self):
        snap = self.snapshot
        self.assertFalse(snap["dirty"])
        self.assertIsNone(snap["dirty_closure_sha256"])
        paths = {e[0]: e for e in snap["closure"]}
        self.assertEqual(paths["lib-link.rs"][1:4], ["symlink", "120000", "src/lib.rs"])
        self.assertEqual(paths["tool.sh"][2], "100755")
        self.assertEqual(paths["Cargo.lock"][4], snap["lock_sha256"])
        self.assertIn("benches/poseidon_modp.rs", paths)
        self.assertEqual(snap["source_snapshot_id"], pc.sha256_hex(pc.canonical_json_bytes(
            {"commit": snap["commit"], "tree": snap["tree"], "lock_blob": snap["lock_blob"],
             "lock_sha256": snap["lock_sha256"], "closure": snap["closure"]})))
        be.recheck_source_snapshot(self.repo, snap)
        self.assertEqual(be.source_snapshot(self.repo, False), snap)

    def test_ignored_extra_file_rejected(self):
        self.add_file(os.path.join(self.repo, "ignored.txt"))
        self.assertRaisesCode("DirtyWorktree", be.source_snapshot, self.repo, False)
        self.assertRaisesCode("SourceChanged", be.recheck_source_snapshot, self.repo,
                              self.snapshot)

    def test_tracked_byte_change_rejected(self):
        path = os.path.join(self.repo, "src", "lib.rs")
        with open(path, "rb") as f:
            original = f.read()
        self.addCleanup(fake.write, path, original)
        fake.write(path, original + b"// changed\n")
        self.assertRaisesCode("DirtyWorktree", be.source_snapshot, self.repo, False)

    def test_git_file_and_nested_repo_rejected(self):
        plain = fresh_dir(self, self.root, "gitfile-")
        fake.write(os.path.join(plain, ".git"), "gitdir: /nowhere\n")
        fake.write(os.path.join(plain, "Cargo.lock"), "version = 4\n")
        self.assertRaisesCode("SourceSnapshotInvalid", be.source_snapshot, plain, False)
        nested = os.path.join(self.repo, "benches", ".git")
        os.makedirs(nested)
        self.addCleanup(shutil.rmtree, nested, True)
        self.assertRaisesCode("SourceSnapshotInvalid", be.source_snapshot, self.repo, True)

    def test_escaping_symlink_and_special_file_rejected(self):
        link = os.path.join(self.repo, "escape")
        os.symlink("../outside", link)
        self.addCleanup(os.unlink, link)
        self.assertRaisesCode("SourceSnapshotInvalid", be.source_snapshot, self.repo, True)
        os.unlink(link)
        os.mkfifo(link)
        self.assertRaisesCode("SourceSnapshotInvalid", be.source_snapshot, self.repo, True)

    def test_dirty_override_closure_changes_detected(self):
        fx = new_fixture(self)
        extra = os.path.join(fx["repo"], "scratch.txt")
        fake.write(extra, b"one\n")
        out = os.path.join(fx["repo"], "out")
        fake.write(os.path.join(out, "result.json"), b"{}\n")
        self.assertRaisesCode("DirtyWorktree", be.source_snapshot, fx["repo"], False)
        snap = be.source_snapshot(fx["repo"], True, external_roots=[out])
        self.assertTrue(snap["dirty"])
        self.assertEqual(snap["dirty_closure_sha256"], snap["closure_sha256"])
        paths = [e[0] for e in snap["closure"]]
        self.assertIn("scratch.txt", paths)
        self.assertNotIn("out/result.json", paths)
        self.assertTrue(pc.is_hex64(snap["head_index_diff_sha256"]))
        be.recheck_source_snapshot(fx["repo"], snap)
        fake.write(extra, b"two\n")
        self.assertRaisesCode("SourceChanged", be.recheck_source_snapshot, fx["repo"], snap)
        fake.write(extra, b"one\n")
        be.recheck_source_snapshot(fx["repo"], snap)
        fake.git(fx["repo"], "add", "scratch.txt")  # index state is part of the identity
        self.assertRaisesCode("SourceChanged", be.recheck_source_snapshot, fx["repo"], snap)


class EnvironmentTest(BuildEnvBase):
    def test_rejected_inherited_names(self):
        for name in ("CARGO_PROFILE_BENCH_LTO", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUSTC",
                     "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO", "CARGO_TARGET_DIR",
                     "CARGO_INCREMENTAL", "CARGO_BUILD_TARGET", "CARGO_BUILD_RUSTC_WRAPPER",
                     "CARGO_BUILD_RUSTFLAGS", "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS",
                     "CARGO_TARGET_X_RUNNER", "CARGO_TARGET_X_LINKER", "PYTHONPYCACHEPREFIX"):
            exc = self.assertRaisesCode("RejectedEnvironment", be.reject_inherited_environment,
                                        {"PATH": "/usr/bin", name: "1"})
            self.assertIn(name, exc.message)
        audit = be.reject_inherited_environment({"PATH": "/usr/bin", "HOME": "/h", "CARGO_HOME": "/c"})
        self.assertEqual(audit["rejected"], [])
        self.assertEqual(audit["checked_names"], list(be.REJECTED_ENV_RULES))
        self.assertEqual(audit["checked_names"],
                         SCHEMA["closed_environment"]["rejected_inherited"])

    def test_toolchain_resolution_with_symlinked_launchers(self):
        tc = self.toolchain
        self.assertEqual(tc["cargo_launcher"]["path"], os.path.join(self.fx["launchers"], "cargo"))
        self.assertEqual(tc["cargo_launcher"]["kind"], "symlink")
        self.assertEqual(tc["cargo_launcher"]["link_chain"], ["rustup"])
        self.assertIsNone(tc["cargo_launcher"]["sha256"])
        self.assertEqual(tc["cargo_launcher"]["resolved_sha256"],
                         pc.sha256_file(os.path.join(self.fx["launchers"], "rustup")))
        self.assertEqual(tc["sysroot"], self.fx["sysroot"])
        self.assertEqual(tc["cargo_path"], os.path.join(self.fx["sysroot"], "bin", "cargo"))
        self.assertEqual(tc["cargo_version"], SCHEMA["toolchain"]["cargo_version"])
        self.assertEqual(tc["host_triple"], "aarch64-apple-darwin")
        self.assertEqual(tc["wrapper_chain"], [])
        names = [e[0] for e in tc["toolchain_closure"]]
        self.assertEqual(names[:2], ["bin/cargo", "bin/rustc"])
        self.assertIn("lib/libstd-fake.dylib", names)
        self.assertIn("lib/rustlib/aarch64-apple-darwin/lib/libcore-fake.rlib", names)
        self.assertEqual(be.recheck_toolchain(tc), tc["toolchain_closure_sha256"])

    def test_toolchain_version_and_closure_mismatch(self):
        self.set_control(lambda c: c.update(cargo_version="cargo 1.0.0 (0 2000-01-01)"))
        self.assertRaisesCode("ToolchainMismatch", be.resolve_toolchain, self.env0, SCHEMA,
                              cwd=self.repo)
        lib = os.path.join(self.fx["sysroot"], "lib", "extra.rlib")
        self.add_file(lib)
        self.assertRaisesCode("ToolchainChanged", be.recheck_toolchain, self.toolchain)

    def test_closed_environment_exact_key_set(self):
        target = fresh_dir(self, self.root, "target-")
        env = self.closed_env(target, threads=1)
        allow = SCHEMA["closed_environment"]["allowlist"]
        expected = {n for n in allow if n != "CARGO_TARGET_<TRIPLE>_LINKER"}
        expected.add("CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER")
        self.assertEqual(set(env), expected)
        self.assertEqual(env["PATH"], os.path.join(self.fx["sysroot"], "bin") + ":/usr/bin:/bin")
        self.assertEqual(env["RUSTC"], self.toolchain["rustc_path"])
        self.assertEqual(env["RUSTFLAGS"], "-C target-cpu=native")
        self.assertEqual(env["CARGO_TARGET_DIR"], target)
        self.assertEqual(env["CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER"], "/usr/bin/cc")
        self.assertEqual(env["RAYON_NUM_THREADS"], "1")
        self.assertEqual(env["CC_ENABLE_DEBUG_OUTPUT"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        # T > 1 only changes RAYON_NUM_THREADS (no feature anywhere).
        self.assertEqual(self.closed_env(threads=8)["RAYON_NUM_THREADS"], "8")
        with self.assertRaises(pc.RunnerError):
            self.closed_env(target, threads=0)

    def test_identity_projection_tokens(self):
        target = fresh_dir(self, self.root, "target-")
        env = self.closed_env(target)
        roots = {"SOURCE": self.repo, "CARGO_HOME": self.fx["cargo_home"],
                 "HOME": self.fx["home"], "SYSROOT": self.fx["sysroot"], "ROLE_TARGET": target,
                 "ROLE_TEMP": env["TMPDIR"], "METADATA_TARGET": None}
        proj = be.identity_projection(env, roots)
        self.assertEqual(proj["PATH"], "<SYSROOT>/bin:/usr/bin:/bin")
        self.assertEqual(proj["HOME"], "<HOME>")
        self.assertEqual(proj["CARGO_HOME"], "<CARGO_HOME>")
        self.assertEqual(proj["CARGO_TARGET_DIR"], "<ROLE_TARGET>")
        self.assertEqual(proj["TMPDIR"], "<ROLE_TEMP>")
        self.assertEqual(proj["RUSTC"], "<SYSROOT>/bin/rustc")
        self.assertEqual(proj["CC"], "/usr/bin/cc")
        be.assert_no_raw_locators(proj, roots, "projection")
        self.assertEqual(be.project_string(fake.package_id(self.repo), roots),
                         "path+file://<SOURCE>#limber@0.1.0")
        self.assertEqual(be.project_string("/x" + self.repo, roots), "/x" + self.repo)
        self.assertRaisesCode("IdentityProjectionInvalid", be.identity_projection, env,
                              {"OTHER": "/tmp"})

    @unittest.skipUnless(platform.system() == "Darwin" and shutil.which("xcrun"),
                         "Darwin native tool resolution")
    def test_resolve_native_tools_darwin(self):
        native = be.resolve_native_tools(self.toolchain, SCHEMA)
        self.assertEqual(native["cc"], "/usr/bin/cc")
        self.assertEqual(native["cxx"], "/usr/bin/c++")
        self.assertEqual(native["ar"], "/usr/bin/ar")
        self.assertEqual(native["ranlib"], "/usr/bin/ranlib")
        self.assertEqual(native["linker_var_name"], "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER")
        self.assertEqual(native["linker"], "/usr/bin/cc")
        self.assertEqual(native["env"]["CC"], "/usr/bin/cc")
        self.assertTrue(native["sdk"]["version"])
        self.assertTrue(os.path.isabs(native["ld"]["path"]))
        self.assertTrue(native["developer_dir"].startswith("/"))
        self.assertTrue(pc.is_hex64(native["native_closure_sha256"]))
        self.assertEqual(be.recheck_native_tools(native), native["native_closure_sha256"])


class CargoConfigTest(BuildEnvBase):
    def audit(self, **kw):
        return be.cargo_config_audit(self.repo, self.fx["cargo_home"], SCHEMA, **kw)

    def test_audit_accepts_absent_and_allowlisted_config(self):
        audit = self.audit()
        self.assertEqual(audit["discovered"], [])
        self.assertEqual(audit["manifest"]["profile_block"],
                         SCHEMA["bench_profile"]["manifest_block"])
        self.assertEqual(audit["candidates"][0]["path"],
                         os.path.join(self.repo, ".cargo", "config.toml"))
        self.assertIn(os.path.join(self.fx["cargo_home"], "config.toml"),
                      [c["path"] for c in audit["candidates"]])
        self.assertEqual(be.recheck_cargo_config(audit), audit["audit_id"])
        self.assertEqual(be.recheck_cargo_config(audit, SCHEMA), audit["audit_id"])
        cfg = os.path.join(self.repo, ".cargo", "config.toml")
        self.add_file(cfg, '[term]\ncolor = "never"\nverbose = false\n')
        self.addCleanup(shutil.rmtree, os.path.dirname(cfg), True)
        self.assertRaisesCode("CargoConfigChanged", be.recheck_cargo_config, audit)
        audit2 = self.audit()
        self.assertEqual(audit2["discovered"], [cfg])
        self.assertEqual(audit2["origins"], {"term.color": cfg, "term.verbose": cfg})
        self.assertNotEqual(audit2["audit_id"], audit["audit_id"])

    def test_rejection_classes(self):
        cfg_dir = os.path.join(self.repo, ".cargo")
        os.makedirs(cfg_dir)
        self.addCleanup(shutil.rmtree, cfg_dir, True)
        cfg = os.path.join(cfg_dir, "config.toml")
        cases = [
            ("symlink", lambda: (fake.write(cfg + ".real", "[term]\n"), os.symlink(cfg + ".real", cfg))),
            ("pair", lambda: (fake.write(cfg, "[term]\n"), fake.write(os.path.join(cfg_dir, "config"), "[term]\n"))),
            ("parse_error", lambda: fake.write(cfg, "[term\n")),
            ("forbidden_table", lambda: fake.write(cfg, "[build]\njobs = 1\n")),
            ("forbidden_profile", lambda: fake.write(cfg, "[profile.bench]\nlto = false\n")),
            ("unknown_key", lambda: fake.write(cfg, "[term]\nfoo = 1\n")),
            ("cargo_home_forbidden", lambda: fake.write(os.path.join(self.fx["cargo_home"], "config.toml"), "[net]\noffline = true\n")),
        ]
        for name, setup in cases:
            setup()
            try:
                exc = self.assertRaisesCode("CargoConfigRejected", self.audit)
                self.assertTrue(exc.message, name)
            finally:
                for p in (cfg, cfg + ".real", os.path.join(cfg_dir, "config"),
                          os.path.join(self.fx["cargo_home"], "config.toml")):
                    if os.path.lexists(p):
                        os.unlink(p)
        self.assertRaisesCode("CargoConfigRejected", self.audit, rustflags="-C target-cpu=native -C panic=abort")
        self.assertRaisesCode("CargoConfigRejected", self.audit, rustflags="-C target-cpu=x86-64")

    def test_profile_block_mismatch_and_package_override(self):
        manifest = os.path.join(self.repo, "Cargo.toml")
        with open(manifest, "rb") as f:
            original = f.read()
        self.addCleanup(fake.write, manifest, original)
        fake.write(manifest, original.replace(b'lto = "fat"', b'lto = "thin"'))
        self.assertRaisesCode("CargoConfigRejected", self.audit)
        fake.write(manifest, original + b'\n[profile.bench.package.zstd-sys]\nopt-level = 1\n')
        self.assertRaisesCode("CargoConfigRejected", self.audit)
        fake.write(manifest, original.replace(b'strip = "none"\n', b''))
        self.assertRaisesCode("CargoConfigRejected", self.audit)
        fake.write(manifest, original)
        self.audit()


class DependencySourceTest(BuildEnvBase):
    def test_audit_document_and_identity(self):
        audit = self.dependency_audit()
        doc = audit["document"]
        self.assertEqual(doc["schema"], "zinc-plus/dependency-sources/v1")
        self.assertEqual(doc["cargo_version"], SCHEMA["toolchain"]["cargo_version"])
        self.assertEqual(doc["metadata_command"][0], "<SYSROOT>/bin/cargo")
        self.assertEqual(doc["metadata_command"][1:], ["metadata", "--locked", "--offline",
                                                       "--format-version", "1",
                                                       "--filter-platform",
                                                       "aarch64-apple-darwin"])
        self.assertEqual(doc["features"], [])
        self.assertEqual(doc["lock_sha256"], self.snapshot["lock_sha256"])
        self.assertEqual(doc["zstd_sys_features"], ["legacy", "std", "zdict_builder"])
        self.assertEqual(sorted(SCHEMA["evidence"]["dependency_sources_fields"]), sorted(doc))
        kinds = {p["name"]: p["source_kind"] for p in doc["packages"]}
        self.assertEqual(kinds, {"limber": "path", "zstd-sys": "registry", "fakegit": "git"})
        ids = [p["id"] for p in doc["packages"]]
        self.assertEqual(ids, sorted(ids, key=lambda s: s.encode()))
        reg = next(p for p in doc["packages"] if p["name"] == "zstd-sys")
        self.assertEqual(reg["lock_checksum"], self.fx["crate"]["checksum"])
        self.assertEqual(reg["registry"]["archive_sha256"], self.fx["crate"]["checksum"])
        self.assertEqual(reg["registry"]["marker"]["sha256"], be.REGISTRY_MARKER_SHA256)
        self.assertEqual(reg["registry"]["unpacked_root"],
                         "<CARGO_HOME>/registry/src/%s/zstd-sys-2.1.0+zstd.1.5.7" % fake.INDEX)
        self.assertEqual(reg["registry"]["directories"],
                         [["scripts", "755", False], ["src", "755", False]])
        entries = {e[0]: e for e in reg["entries"]}
        self.assertEqual(entries["scripts/gen.sh"][2], "100755")
        self.assertEqual(entries["src/alias.rs"][1:4], ["symlink", "120000", "lib.rs"])
        self.assertEqual(entries[".cargo-ok"][3], 7)
        git = next(p for p in doc["packages"] if p["name"] == "fakegit")
        self.assertEqual(git["git"]["commit"], self.fx["checkout"]["commit"])
        self.assertEqual(git["git"]["marker"]["sha256"], be.GIT_MARKER_SHA256)
        self.assertIn(".cargo-ok", [e[0] for e in git["entries"]])
        path_pkg = next(p for p in doc["packages"] if p["name"] == "limber")
        self.assertEqual(path_pkg["manifest_path"], "<SOURCE>/Cargo.toml")
        self.assertEqual(path_pkg["root"], "<SOURCE>")
        self.assertIn("benches/poseidon_modp.rs", [e[0] for e in path_pkg["entries"]])
        raw = audit["bytes"].decode()
        for locator in (self.repo, self.fx["cargo_home"], self.fx["home"], "file:///"):
            self.assertNotIn(locator, raw)
        self.assertEqual(audit["dependency_source_id"], pc.sha256_hex(audit["bytes"]))
        self.assertNotIn(audit["dependency_source_id"], raw)
        self.assertEqual(be.rehash_dependency_closure(audit), audit["closure_sha256"])
        again = self.dependency_audit()
        self.assertEqual(again["dependency_source_id"], audit["dependency_source_id"])
        self.assertEqual(again["metadata_normalized_sha256"], audit["metadata_normalized_sha256"])
        be.replay_metadata(audit, self.closed_env())

    def test_features_are_rejected(self):
        env = self.closed_env()
        meta = fresh_dir(self, self.root, "meta-")
        self.assertRaisesCode("DependencySourceInvalid", be.dependency_source_audit, self.repo,
                              self.snapshot, env, self.toolchain, self.fx["cargo_home"], meta,
                              ["parallel"], SCHEMA)

    def test_registry_archive_checksum_mismatch(self):
        archive = self.fx["crate"]["archive"]
        with open(archive, "rb") as f:
            original = f.read()
        self.addCleanup(fake.write, archive, original)
        fake.write(archive, original + b"\0")
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)

    def test_registry_extra_unpacked_file_and_missing_marker(self):
        extra = os.path.join(self.fx["crate"]["root"], "src", "extra.rs")
        self.add_file(extra)
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)
        os.unlink(extra)
        marker = os.path.join(self.fx["crate"]["root"], ".cargo-ok")
        self.addCleanup(fake.write, marker, b'{"v":1}')
        fake.write(marker, b'{"v":1}\n')
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)

    def test_archive_member_rules(self):
        prefix = "evil-0.1.0"

        def archive_with(members):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                for name, kind, target in members:
                    info = tarfile.TarInfo(name)
                    info.type = kind
                    info.linkname = target
                    if kind == tarfile.REGTYPE:
                        info.size = 1
                        tf.addfile(info, io.BytesIO(b"x"))
                    else:
                        tf.addfile(info)
            path = os.path.join(fresh_dir(self, self.root, "arch-"), prefix + ".crate")
            fake.write(path, buf.getvalue())
            return path
        root = fresh_dir(self, self.root, "unpacked-")
        fake.write(os.path.join(root, "a"), b"x")
        fake.write(os.path.join(root, ".cargo-ok"), b'{"v":1}')
        ok = archive_with([(prefix + "/a", tarfile.REGTYPE, "")])
        self.assertEqual(be._audit_crate_archive(ok, prefix, root)["member_count"], 1)
        for members in ([(prefix + "/a", tarfile.REGTYPE, ""), (prefix + "/link", tarfile.SYMTYPE, "../../etc/passwd")],
                        [(prefix + "/a", tarfile.REGTYPE, ""), (prefix + "/link", tarfile.SYMTYPE, "/etc/passwd")],
                        [(prefix + "/a", tarfile.REGTYPE, ""), (prefix + "/b", tarfile.LNKTYPE, prefix + "/a")],
                        [(prefix + "/a", tarfile.REGTYPE, ""), (prefix + "/a", tarfile.REGTYPE, "")],
                        [("other-0.1.0/a", tarfile.REGTYPE, "")],
                        [(prefix + "/../a", tarfile.REGTYPE, "")],
                        [("/" + prefix + "/a", tarfile.REGTYPE, "")],
                        [(prefix + "/a", tarfile.REGTYPE, ""), (prefix + "/fifo", tarfile.FIFOTYPE, "")]):
            bad = archive_with(members)
            self.assertRaisesCode("DependencySourceInvalid", be._audit_crate_archive, bad,
                                  prefix, root)

    def test_git_head_mismatch_and_untracked_extra(self):
        root = self.fx["checkout"]["root"]
        self.add_file(os.path.join(root, "untracked.txt"))
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)
        os.unlink(os.path.join(root, "untracked.txt"))
        fake.write(os.path.join(root, "new.rs"), b"// new\n")
        fake.git(root, "add", "new.rs")
        fake.git(root, "commit", "-q", "-m", "moved")
        self.addCleanup(fake.git, root, "reset", "-q", "--hard", self.fx["checkout"]["commit"])
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)

    def test_lock_mismatch_and_pkg_config_feature(self):
        def drop_edge(control):
            control["metadata"]["resolve"]["nodes"][0]["dependencies"].append(
                "registry+https://github.com/rust-lang/crates.io-index#pkg-config@0.3.34")
        self.set_control(drop_edge)
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)

        def pkg_config(control):
            control["metadata"]["resolve"]["nodes"][1]["features"].append("pkg-config")
        self.set_control(pkg_config)
        self.assertRaisesCode("DependencySourceInvalid", self.dependency_audit)

    def test_metadata_replay_mismatch(self):
        audit = self.dependency_audit()

        def second(control):
            control["metadata_second"] = json.loads(json.dumps(control["metadata"]))
            control["metadata_second"]["packages"][0]["edition"] = "2021"
        self.set_control(second)
        self.assertRaisesCode("MetadataReplayMismatch", be.replay_metadata, audit,
                              self.closed_env())

    def test_rehash_detects_mutation(self):
        audit = self.dependency_audit()
        path = os.path.join(self.fx["crate"]["root"], "src", "lib.rs")
        with open(path, "rb") as f:
            original = f.read()
        self.addCleanup(fake.write, path, original)
        fake.write(path, original + b"//\n")
        self.assertRaisesCode("DependencyClosureChanged", be.rehash_dependency_closure, audit)
        fake.write(path, original)
        os.chmod(path, 0o755)
        self.assertRaisesCode("DependencyClosureChanged", be.rehash_dependency_closure, audit)
        os.chmod(path, 0o644)
        self.assertEqual(be.rehash_dependency_closure(audit), audit["closure_sha256"])
        marker = os.path.join(self.fx["checkout"]["root"], ".cargo-ok")
        self.addCleanup(fake.write, marker, b"")
        fake.write(marker, b"x")
        self.assertRaisesCode("DependencyClosureChanged", be.rehash_dependency_closure, audit)


class EvidenceBuildTest(BuildEnvBase):
    def build(self, env=None, dep=None, role="hyrax", schema=SCHEMA):
        env = env or self.closed_env()
        dep = dep or self.dependency_audit(env)
        return be.evidence_build(self.repo, env, env["CARGO_TARGET_DIR"], [], role, schema,
                                 dependency_audit=dep), env, dep

    def test_grammar_role_missing_fails_closed(self):
        """A schema without a `roles.limber` grammar makes the evidence build fail closed."""
        env = self.closed_env()
        exc = self.assertRaisesCode("RustcArgvInvalid", be.evidence_build, self.repo, env,
                                    env["CARGO_TARGET_DIR"], [], "hyrax", ROLELESS_SCHEMA)
        self.assertIn("'limber'", exc.message)
        self.assertIn("'hyrax'", exc.message)
        self.assertIn("['zinc']", exc.message)
        self.assertRaisesCode("RustcArgvInvalid", be.rustc_argv_grammar, ROLELESS_SCHEMA,
                              "brakedown")
        # The pinned shared schema carries the captured limber grammar.
        pinned = be.rustc_argv_grammar(PRISTINE_SCHEMA, "hyrax")["features"]["none"]
        self.assertEqual(pinned["tokens"][:2], ["--crate-name", "poseidon_modp"])
        self.assertEqual(pinned["token_count"], len(pinned["tokens"]))
        self.assertEqual(be.rustc_argv_grammar(SCHEMA, "hyrax")["features"]["none"]["tokens"],
                         GRAMMAR)
        self.assertRaisesCode("RustcArgvInvalid", be.rustc_argv_grammar,
                              {"rustc_argv_grammar": {}}, "hyrax")
        # Only the limber roles are accepted.
        self.assertRaisesCode("EvidenceBuildFailed", be.evidence_build, self.repo, env,
                              env["CARGO_TARGET_DIR"], [], "zinc", SCHEMA)

    def test_evidence_build_accepted(self):
        ev, env, dep = self.build()
        self.assertEqual(ev["exit"], 0)
        self.assertEqual(ev["role"], "hyrax")
        self.assertEqual(ev["executable"]["relative_path"], fake.EXECUTABLE_REL)
        self.assertEqual(ev["executable"]["sha256"], pc.sha256_hex(b"fake bench executable\n"))
        self.assertTrue(ev["log_bytes"].startswith(b"--- stdout ---\n"))
        self.assertIn(b"\n--- stderr ---\n", ev["log_bytes"])
        self.assertEqual(ev["log_sha256"], pc.sha256_hex(ev["log_bytes"]))
        self.assertEqual(ev["rustc_argv_normalized"], GRAMMAR)
        self.assertEqual(ev["rustc_program"], "<SYSROOT>/bin/rustc")
        self.assertEqual(ev["argv_normalized"][0], "<SYSROOT>/bin/cargo")
        self.assertEqual(ev["argv"][1:], pc.evidence_command([])[1:])
        self.assertNotIn("-p", ev["argv"])
        self.assertEqual([l["class"] for l in ev["native_lines"]],
                         ["cc", "sdk_probe", "cc", "cc", "ar"])
        self.assertEqual(ev["native_env_reads"], {"AR": ["/usr/bin/ar"], "CC": ["/usr/bin/cc"]})
        self.assertEqual(ev["manifest_dirs"], sorted([
            "<SOURCE>", "<CARGO_HOME>/registry/src/%s/zstd-sys-2.1.0+zstd.1.5.7" % fake.INDEX,
            "<CARGO_HOME>/git/checkouts/fakegit-%s/%s" % (fake.GIT_DIR_HASH, self.fx["checkout"]["short"])]))
        self.assertEqual(ev["zstd_sys"], {"bundled_compile_lines": 1, "linked_libs": ["static=zstd"],
                                          "required": False})
        self.assertEqual(ev["artifact"]["src_path"], "<SOURCE>/benches/poseidon_modp.rs")
        self.assertEqual(ev["artifact"]["package_id"], "path+file://<SOURCE>#limber@0.1.0")
        self.assertEqual(be.rehash_executable(env["CARGO_TARGET_DIR"], ev["executable"]),
                         ev["executable"]["sha256"])
        # The brakedown role uses the same binary and grammar.
        ev2, _, _ = self.build(role="brakedown")
        self.assertEqual(ev2["rustc_argv_normalized"], GRAMMAR)
        self.assertEqual(ev2["role"], "brakedown")
        exe = os.path.join(env["CARGO_TARGET_DIR"], fake.EXECUTABLE_REL)
        fake.write(exe, b"replaced\n", 0o755)
        self.assertRaisesCode("ExecutableChanged", be.rehash_executable, env["CARGO_TARGET_DIR"],
                              ev["executable"])
        os.unlink(exe)
        self.assertRaisesCode("ExecutableChanged", be.rehash_executable, env["CARGO_TARGET_DIR"],
                              ev["executable"])
        self.assertRaisesCode("TempDirInvalid", be.evidence_build, self.repo, env,
                              env["CARGO_TARGET_DIR"], [], "hyrax", SCHEMA)

    def assertBuildRejected(self, code, mutate):
        self.set_control(mutate)
        env = self.closed_env()
        self.assertRaisesCode(code, be.evidence_build, self.repo, env, env["CARGO_TARGET_DIR"],
                              [], "hyrax", SCHEMA)

    def test_fresh_artifact_rejected(self):
        def mutate(c):
            lines = c["evidence"]["stdout"]
            lines[-2] = lines[-2].replace('"fresh":false', '"fresh":true')
        self.assertBuildRejected("EvidenceBuildFailed", mutate)

    def test_two_artifacts_rejected(self):
        def mutate(c):
            lines = c["evidence"]["stdout"]
            lines.insert(-1, lines[-2])
        self.assertBuildRejected("EvidenceBuildFailed", mutate)

    def test_wrong_bench_target_rejected(self):
        def rename(c):
            lines = c["evidence"]["stdout"]
            lines[-2] = lines[-2].replace('"name":"poseidon_modp"', '"name":"poseidon"')
        self.assertBuildRejected("EvidenceBuildFailed", rename)

        def src(c):
            lines = c["evidence"]["stdout"]
            lines[-2] = lines[-2].replace("benches/poseidon_modp.rs", "benches/poseidon.rs")
        self.assertBuildRejected("EvidenceBuildFailed", src)

    def test_argv_grammar_deviations_rejected(self):
        def extra_c(c):
            c["evidence"]["records"][-1]["args"].extend(["-C", "panic=abort"])
        self.assertBuildRejected("RustcArgvInvalid", extra_c)

        def dropped(c):
            args = c["evidence"]["records"][-1]["args"]
            del args[args.index("--cfg"):args.index("--cfg") + 2]
        self.assertBuildRejected("RustcArgvInvalid", dropped)

        def reordered(c):
            args = c["evidence"]["records"][-1]["args"]
            i = args.index("--extern")
            args[i + 1], args[i + 3] = args[i + 3], args[i + 1]
        self.assertBuildRejected("RustcArgvInvalid", reordered)

        def target_cpu(c):
            args = c["evidence"]["records"][-1]["args"]
            args[args.index("target-cpu=native")] = "target-cpu=apple-m1"
        self.assertBuildRejected("RustcArgvInvalid", target_cpu)

        def wrapper(c):
            c["evidence"]["records"][-1]["program"] = "/usr/local/bin/sccache"
            c["evidence"]["records"][-1]["args"].insert(0, "<RUSTC>")
        self.assertBuildRejected("EvidenceBuildFailed", wrapper)

        def foreign_rustc(c):
            c["evidence"]["records"][-1]["program"] = "/usr/local/bin/rustc"
        self.assertBuildRejected("RustcArgvInvalid", foreign_rustc)

        def twice(c):
            c["evidence"]["records"].append(c["evidence"]["records"][-1])
        self.assertBuildRejected("RustcArgvInvalid", twice)

    def test_foreign_manifest_dir_rejected(self):
        def mutate(c):
            c["evidence"]["records"][1]["env"]["CARGO_MANIFEST_DIR"] = "/opt/other/zstd-sys"
        self.assertBuildRejected("EvidenceBuildFailed", mutate)

    def test_native_lines_rejected(self):
        def gcc(c):
            lines = c["evidence"]["stdout"]
            i = next(i for i, l in enumerate(lines) if '"<AR>"' in l)
            lines[i] = lines[i].replace('"<AR>"', '"/usr/bin/gcc-ar"')
        self.assertBuildRejected("NativeToolInvalid", gcc)

        def cc_none(c):
            lines = c["evidence"]["stdout"]
            i = next(i for i, l in enumerate(lines) if l.endswith('CC = Some("<CC>")'))
            lines[i] = lines[i].replace('Some("<CC>")', "None")
        self.assertBuildRejected("NativeToolInvalid", cc_none)

    def test_bundled_zstd_not_required(self):
        """limber has no zstd dependency: the Zinc-only bundled-zstd rule is not applied."""
        def system_zstd(c):
            lines = c["evidence"]["stdout"]
            c["evidence"]["stdout"] = [l for l in lines if "zstd/lib/common/debug.c" not in l]
        self.set_control(system_zstd)
        ev, _, _ = self.build()
        self.assertEqual(ev["zstd_sys"]["bundled_compile_lines"], 0)
        self.assertFalse(ev["zstd_sys"]["required"])

    def test_executable_symlink_rejected(self):
        self.assertBuildRejected("EvidenceBuildFailed",
                                 lambda c: c["evidence"].update(executable_symlink=True))

    def test_build_profile_document(self):
        ev, env, dep = self.build()
        cfg = be.cargo_config_audit(self.repo, self.fx["cargo_home"], SCHEMA)
        roots = {"SOURCE": self.repo, "CARGO_HOME": self.fx["cargo_home"],
                 "HOME": self.fx["home"], "SYSROOT": self.fx["sysroot"],
                 "ROLE_TARGET": env["CARGO_TARGET_DIR"], "ROLE_TEMP": env["TMPDIR"]}
        rejected = be.reject_inherited_environment({"PATH": "/usr/bin"})
        doc = be.build_profile_document(ev, self.snapshot, dep, cfg, self.toolchain, FAKE_NATIVE,
                                        env, roots, rejected, COMPILED, SCHEMA, [], "hyrax")
        self.assertEqual(doc["schema"], "zinc-plus/build-profile/v1")
        self.assertEqual(doc["role"], "hyrax")
        self.assertEqual(sorted(doc), sorted(SCHEMA["evidence"]["build_profile_fields"]))
        self.assertEqual(doc["dependency_source_id"], dep["dependency_source_id"])
        self.assertEqual(doc["dependency_sources_sha256"], dep["dependency_source_id"])
        self.assertEqual(doc["environment"]["raw"], env)
        self.assertEqual(doc["environment"]["projected"]["CARGO_TARGET_DIR"], "<ROLE_TARGET>")
        self.assertEqual(doc["toolchain"]["cargo_path"], "<SYSROOT>/bin/cargo")
        self.assertEqual(doc["launchers"]["cargo"]["path"], "<HOME>/.cargo/bin/cargo")
        self.assertEqual(doc["target"], {"path": "<ROLE_TARGET>", "initially_empty": True,
                                         "symlink": False})
        self.assertEqual(doc["wrapper_chain"], [])
        self.assertEqual(doc["probes"], COMPILED)
        self.assertEqual(doc["recheck_policy"], be.RECHECK_POLICY)
        self.assertEqual(doc["manifest"]["profile_block"], SCHEMA["bench_profile"]["manifest_block"])
        self.assertEqual(doc["command"][1:], pc.evidence_command([])[1:])
        pc.canonical_json_bytes(doc)
        be.assert_no_raw_locators(doc["environment"]["projected"], roots, "projected")
        be.assert_no_raw_locators(doc["toolchain"], roots, "toolchain")
        bad = dict(COMPILED, overflow_checks=True)
        self.assertRaisesCode("EvidenceBuildFailed", be.build_profile_document, ev, self.snapshot,
                              dep, cfg, self.toolchain, FAKE_NATIVE, env, roots, rejected, bad,
                              SCHEMA, [])

    def test_run_cargo_subprocess(self):
        ev, env, dep = self.build()
        cfg = be.cargo_config_audit(self.repo, self.fx["cargo_home"], SCHEMA)
        native = None
        if platform.system() == "Darwin" and shutil.which("xcrun"):
            native = be.resolve_native_tools(self.toolchain, SCHEMA)
        bundle = {"repo": self.repo, "snapshot": self.snapshot, "dependency": dep,
                  "cargo_config": cfg, "toolchain": self.toolchain, "native": native,
                  "executable": ev["executable"], "target_dir": env["CARGO_TARGET_DIR"],
                  "timing_schema": SCHEMA}
        logs = fresh_dir(self, self.root, "logs-")
        sink = {"stdout": os.path.join(logs, "out"), "stderr": os.path.join(logs, "err")}
        rec = be.run_cargo_subprocess(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], env,
                                      self.repo, bundle, sink)
        self.assertEqual((rec["exit"], rec["signal"]), (3, None))
        self.assertEqual(rec["stdout_sha256"], pc.sha256_hex(b"out\n"))
        with open(sink["stderr"], "rb") as f:
            self.assertEqual(f.read(), b"err\n")
        self.assertEqual(rec["pre"], rec["post"])
        self.assertEqual(rec["pre"]["source"], self.snapshot["source_snapshot_id"])
        self.assertEqual(rec["pre"]["dependency"], dep["closure_sha256"])
        self.assertEqual(rec["pre"]["cargo_config"], cfg["audit_id"])
        self.assertEqual(rec["pre"]["toolchain"], self.toolchain["toolchain_closure_sha256"])
        self.assertEqual(rec["pre"]["executable"], ev["executable"]["sha256"])
        self.assertLessEqual(rec["start_mono"], rec["end_mono"])
        self.assertTrue(pc.RFC3339_NS_RE.match(rec["start_utc"]))
        captured = {}
        rec2 = be.run_cargo_subprocess(["/bin/sh", "-c", "kill -9 $$"], env, self.repo, bundle,
                                       lambda name, data: captured.__setitem__(name, data))
        self.assertEqual((rec2["exit"], rec2["signal"]), (None, 9))
        self.assertEqual(captured, {"stdout": b"", "stderr": b""})
        exe = os.path.join(env["CARGO_TARGET_DIR"], fake.EXECUTABLE_REL)
        fake.write(exe, b"mutated\n", 0o755)
        self.assertRaisesCode("ExecutableChanged", be.run_cargo_subprocess, ["/bin/sh", "-c", ""],
                              env, self.repo, bundle, sink)


class BytecodeTest(unittest.TestCase):
    def test_zz_no_bytecode_written_under_repo(self):
        self.assertTrue(sys.dont_write_bytecode, "run the tests with python3 -B")
        fresh = []
        for dirpath, dirnames, filenames in os.walk(REAL_REPO):
            dirnames[:] = [d for d in dirnames if d not in (".git", "target")]
            for name in filenames:
                if name.endswith((".pyc", ".pyo")):
                    full = os.path.join(dirpath, name)
                    if os.stat(full).st_mtime >= TEST_START - 1:
                        fresh.append(full)
        self.assertEqual(fresh, [])
        self.assertFalse(os.path.isdir(os.path.join(REAL_REPO, "scripts", "tests",
                                                    "__pycache__")))
        self.assertFalse(os.path.isdir(os.path.join(REAL_REPO, "scripts", "__pycache__")))


if __name__ == "__main__":
    unittest.main()
