#!/usr/bin/env bash
# Thin wrapper: runs the limber Poseidon2 benchmark runner with the caller's environment
# (Zinc plan v10 section 9 / limber contract section 6; replaces the v9 two-phase script).
#   SWEEP=1 BACKEND=hyrax OUTPUT_PARENT=/abs/out scripts/run_poseidon_bench.sh
#   PSIZE=1 NORMAL_RUN=/abs/session/systems/hyrax TUNING_STORE=/abs/store \
#       OUTPUT_PARENT=/abs/out scripts/run_poseidon_bench.sh
# Normal mode is driven by the Zinc session orchestrator, which invokes
# `python3 -B scripts/poseidon_runner.py --repo <abs> normal <subcommand> ...` directly
# with BACKEND=hyrax|brakedown (see poseidon_runner.py).
#
# Python is always started as `python3 -B` with PYTHONDONTWRITEBYTECODE=1 so that no
# bytecode cache can appear below the source checkout; the Python entry points keep their
# shebang lines but are never relied on (plan v10, "python -B rule").
set -eu
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B "$(dirname "$0")/poseidon_runner.py" "$@"
