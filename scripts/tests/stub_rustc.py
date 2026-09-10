#!/usr/bin/env python3
"""Stand-in for `rustc -vV` in unit tests."""
import sys

if sys.argv[1:] == ["-vV"]:
    print("rustc 1.99.0-stub (0000000 2026-01-01)\nbinary: rustc\ncommit-hash: 0\n"
          "host: stub-host\nrelease: 1.99.0")
    sys.exit(0)
sys.exit(2)
