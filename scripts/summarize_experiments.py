#!/usr/bin/env python3
"""Thin wrapper – delegates to scripts.experiment_manager.summary.

Usage: python scripts/summarize_experiments.py --study <name> [options...]
"""

from __future__ import annotations

import sys

from scripts.experiment_manager.summary import summarize_main

if __name__ == "__main__":
    raise SystemExit(summarize_main())
