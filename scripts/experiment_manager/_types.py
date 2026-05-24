"""Shared dataclass types used across the experiment_manager package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    """Specification for a single experiment run (one hyperparameter combination)."""

    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class RunMeta:
    """Metadata parsed from a run name and its parameters (used in stats/plots)."""

    run_name: str
    model: str
    sweep_var: str
    sweep_path: str
    sweep_value: float | str
