"""Study setup: create folder structure and per-case config.yaml before execution.

Decouples "setup" from "run": after setup, each case directory is self-contained
and can be fitted with a simple ``fit --config case_dir/config.yaml`` command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ._logging import Logger
from ._types import RunSpec
from ._utils import _to_plain


def write_case_configs(
    run_specs: list[RunSpec],
    cases: list[Path],
    results_root: Path,
    study_name: str,
    base_config_path: Path,
    static_overrides: dict[str, Any],
    seed: int,
    logger: Logger,
) -> dict[str, Path]:
    """Create folder structure and write resolved config.yaml per (run, case).

    Each case directory receives a self-contained ``config.yaml`` that merges:
      - the base model config (``base_config_path``)
      - ``static_overrides`` (shared across all runs)
      - the run-specific sweep params
      - case-specific values (``data.path``, ``trainer.devices``, seed)

    Returns ``{run_name: run_root}`` mapping for downstream use.
    """
    base_cfg = OmegaConf.load(base_config_path)
    run_roots: dict[str, Path] = {}

    for run_spec in run_specs:
        run_root = results_root / study_name / run_spec.name
        run_root.mkdir(parents=True, exist_ok=True)
        run_roots[run_spec.name] = run_root

        for case in cases:
            case_dir = run_root / "cases" / case.name
            case_dir.mkdir(parents=True, exist_ok=True)

            overrides = {
                **static_overrides,
                **run_spec.params,
                "data.path": str(case),
                "trainer.devices": 1,
                "data.parser.init_args.seed": seed,
            }

            merged = OmegaConf.merge(base_cfg, OmegaConf.create(overrides))
            OmegaConf.save(merged, case_dir / "config.yaml", resolve=True)

    logger.info(
        f"Setup complete: {len(run_specs)} runs × {len(cases)} cases = "
        f"{len(run_specs) * len(cases)} configs written"
    )
    return run_roots


def handle_aliases(
    alias_to_canonical: dict[str, str],
    run_roots: dict[str, Path],
    logger: Logger,
) -> None:
    """Create symlink for alias runs pointing to their canonical counterpart."""
    for alias_name, canonical_name in sorted(alias_to_canonical.items()):
        canonical_root = run_roots.get(canonical_name)
        if canonical_root is None:
            logger.warn(f"Canonical run root not found for alias {alias_name} -> {canonical_name}")
            continue

        alias_root = canonical_root.parent / alias_name
        if alias_root.is_symlink() or alias_root.exists():
            continue

        alias_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            alias_root.symlink_to(canonical_root, target_is_directory=True)
            logger.info(f"Alias: {alias_name} -> {canonical_name}")
        except OSError as e:
            logger.warn(f"Failed to symlink alias {alias_name}: {e}")


def collect_pending_dirs(
    run_specs: list[RunSpec],
    cases: list[Path],
    results_root: Path,
    study_name: str,
) -> list[Path]:
    """Return case directories that need fitting (no checkpoint yet)."""
    from .filesystem import _has_checkpoint

    pending: list[Path] = []
    for run_spec in run_specs:
        for case in cases:
            case_dir = results_root / study_name / run_spec.name / "cases" / case.name
            if not _has_checkpoint(case_dir):
                pending.append(case_dir)
    return pending
