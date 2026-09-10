#!/usr/bin/env python3
"""limber runner wrapper for out-of-process tests (the limber counterpart of Zinc's
`stub_zinc_runner.py`).

Runs `scripts/poseidon_runner.py` in this process with the stub build environment injected
(`runner.be = stub_build_env`), which is how tests replace package C. Stub inputs are read
from the wrapper's own `STUB_*` environment, never from the runner's knob environment. The
Zinc orchestrator can be pointed at it with `--limber-runner "python3 -B <this file>"`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from scripts import poseidon_runner as runner  # noqa: E402
from scripts.tests import stub_build_env as sbe  # noqa: E402


def main():
    sbe.settings_from_environ()
    if os.environ.get("STUB_LIMBER_METADATA_JSON"):
        sbe.STUB_SETTINGS["metadata_json"] = os.environ["STUB_LIMBER_METADATA_JSON"]
    runner.be = sbe
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUB_")}
    return runner.main(sys.argv[1:], env)


if __name__ == "__main__":
    sys.exit(main())
