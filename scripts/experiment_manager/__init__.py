"""Experiment manager package.

Provides:
  - __main__.main()       -- Hydra-based experiment pipeline entry point
  - summary.summarize_main()  -- argparse-based study summarization entry point
"""

from __future__ import annotations

from .__main__ import main
from ._logging import Logger
from ._types import RunMeta, RunSpec
from ._utils import _write_json
from .summary import summarize_main

__all__ = [
    "main",
    "summarize_main",
    "Logger",
    "RunSpec",
    "RunMeta",
    "_write_json",
]
