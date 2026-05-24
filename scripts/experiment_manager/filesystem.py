"""Filesystem operations: checkpoint detection, sweep-match validation."""

from __future__ import annotations

from pathlib import Path

from ._logging import Logger
from ._types import RunSpec


def _has_checkpoint(case_output_dir: Path) -> bool:
    """Check whether a case output directory contains any .ckpt file."""
    ckpt_dir = case_output_dir / "checkpoints"
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.glob("*.ckpt"))


def _discover_summary_run_names(study_root: Path) -> set[str]:
    """Return the set of run names that already have a run_summary.json."""
    return {p.parent.name for p in study_root.glob("*/run_summary.json")}


def _is_summary_sweep_match_exact(
    run_specs: list[RunSpec],
    results_root: Path,
    study_name: str,
    logger: Logger,
) -> bool:
    """Check that expected runs exactly match existing result folders."""
    expected = {spec.name for spec in run_specs}
    study_root = results_root / study_name
    actual = _discover_summary_run_names(study_root)

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if not missing and not extra:
        logger.info(
            "Summary strict-match passed: sweep config and result folders "
            "are exactly matched"
        )
        return True

    logger.warn(
        "Summary strict-match failed: skip summary due to mismatch between "
        "sweep config and result folders"
    )
    if missing:
        preview = ", ".join(missing[:10])
        tail = " ..." if len(missing) > 10 else ""
        logger.warn(
            "Missing run_summary.json for expected runs "
            f"({len(missing)}): {preview}{tail}"
        )
    if extra:
        preview = ", ".join(extra[:10])
        tail = " ..." if len(extra) > 10 else ""
        logger.warn(
            "Extra result runs not in sweep config "
            f"({len(extra)}): {preview}{tail}"
        )
    return False
