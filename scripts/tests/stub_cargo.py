#!/usr/bin/env python3
"""Stand-in for `<SYSROOT>/bin/cargo` in the unit tests (limber).

Dispatches the exact canonical commands the runner issues:
  `bench ... --bench poseidon_modp --no-run -vv`  -> creates the bench executable under
      `$CARGO_TARGET_DIR/release/deps/` and prints one compiler-artifact JSON message;
  `bench ... --bench poseidon_modp -vv -- <args>` -> runs `stub_bench.py <args> --bench`
      (Cargo appends `--bench` after the user arguments);
  `test ... --lib -vv -- --exact <name>`         -> the one-test summary
      (`STUB_FAIL_GATE=<suffix>` makes that gate fail);
  `metadata --locked --offline ...`               -> a minimal package graph;
  `-V`                                            -> the pinned version string.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXECUTABLE_REL = os.path.join("release", "deps", "poseidon_modp-5tub0000deadbeef")


def main():
    args = sys.argv[1:]
    env = os.environ
    if args == ["-V"]:
        print("cargo 1.98.1 (797e8a9bc 2026-08-05)")
        return 0
    if args[:1] == ["bench"]:
        if "--offline" not in args or "--locked" not in args:
            sys.stderr.write("stub cargo: bench requires --locked --offline\n")
            return 2
        if "--features" in args or "-p" in args:
            sys.stderr.write("stub cargo: the limber bench takes no -p/--features\n")
            return 2
        if args[args.index("--bench") + 1] != "poseidon_modp":
            sys.stderr.write("stub cargo: unknown bench target\n")
            return 2
        target = env.get("CARGO_TARGET_DIR")
        if not target:
            sys.stderr.write("stub cargo: CARGO_TARGET_DIR is required\n")
            return 2
        exe = os.path.join(target, EXECUTABLE_REL)
        if "--no-run" in args:
            os.makedirs(os.path.dirname(exe), exist_ok=True)
            with open(exe, "wb") as f:
                f.write(b"#!stub-bench-executable\nbench=poseidon_modp\n")
            os.chmod(exe, 0o755)
            sys.stderr.write("   Compiling limber v0.1.0 (stub)\n"
                             "     Running `rustc --crate-name poseidon_modp ...`\n"
                             "    Finished `bench` profile [optimized] target(s)\n")
            msg = {"reason": "compiler-artifact", "package_id": "path+file:///stub#limber",
                   "target": {"kind": ["bench"], "name": "poseidon_modp"}, "fresh": False,
                   "executable": exe, "features": []}
            sys.stdout.write(json.dumps(msg) + "\n")
            return 0
        if not os.path.isfile(exe):
            sys.stderr.write("stub cargo: bench executable %s is missing\n" % exe)
            return 101
        if "--" not in args:
            sys.stderr.write("stub cargo: bench without arguments is not used by the runner\n")
            return 2
        bench_args = args[args.index("--") + 1:]
        sys.stderr.write("    Finished `bench` profile [optimized] target(s)\n"
                         "     Running `%s`\n" % exe)
        sys.stderr.flush()
        proc = subprocess.run([sys.executable, "-B", os.path.join(HERE, "stub_bench.py")]
                              + bench_args + ["--bench"], env=env, check=False)
        return proc.returncode
    if args[:1] == ["test"] and "--exact" in args:
        if "--lib" not in args or "-p" in args:
            sys.stderr.write("stub cargo: gates run the crate's --lib tests without -p\n")
            return 2
        name = args[args.index("--exact") + 1]
        if env.get("STUB_FAIL_GATE") and name.endswith(env["STUB_FAIL_GATE"]):
            print("running 1 test\ntest %s ... FAILED\n\ntest result: FAILED. 0 passed; 1 failed;"
                  " 0 ignored; 0 measured; 30 filtered out; finished in 0.01s" % name)
            return 101
        print("running 1 test\ntest %s ... ok\n\ntest result: ok. 1 passed; 0 failed; "
              "0 ignored; 0 measured; 30 filtered out; finished in 0.01s" % name)
        return 0
    if args[:1] == ["metadata"]:
        sys.stdout.write(json.dumps({"packages": [], "workspace_members": [], "version": 1,
                                     "metadata": None}) + "\n")
        return 0
    sys.stderr.write("stub cargo: unsupported %r\n" % (args,))
    return 2


if __name__ == "__main__":
    sys.exit(main())
