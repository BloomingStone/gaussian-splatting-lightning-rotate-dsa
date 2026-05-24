#!/usr/bin/env python3
"""Thin wrapper – delegates to scripts.experiment_manager package.

Usage: python -m scripts.experiment_manager --config-name q1 [overrides...]
"""

from __future__ import annotations

from scripts.experiment_manager.__main__ import main

if __name__ == "__main__":
    main()
