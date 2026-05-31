"""CLI entry point for the experiment manager (Hydra-based).

Two-phase workflow:
  1. SETUP  – create folder structure + per-case config.yaml
  2. RUN    – dispatch ``fit --config <dir>/config.yaml`` across GPUs

Usage: python -m scripts.experiment_manager --config-name q1 [overrides...]
"""

from __future__ import annotations

import shlex
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ._logging import Logger
from ._utils import _to_plain, _write_json
from .config import _deduplicate_run_specs_by_config, _discover_cases, _expand_sweeps
from .filesystem import _is_summary_sweep_match_exact
from .runner import execute_dirs
from .setup import collect_pending_dirs, handle_aliases, write_case_configs
from .summary import generate_study_summary, run_summarize


@hydra.main(
    version_base=None,
    config_path=str(
        Path(__file__).resolve().parent.parent.parent / "configs" / "experiments"
    ),
    config_name="q1",
)
def main(cfg: DictConfig) -> None:
    """Orchestrate the full experiment pipeline: setup → run → summarise."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_root = Path(str(cfg.train.results_root)).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    study_name = str(cfg.study_name)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(
        results_root / "manager_logs" / f"{study_name}_{ts}.log",
        level=str(cfg.train.get("log_level", "INFO")),
    )

    logger.info("Hydra config resolved:")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # ── Discover cases ────────────────────────────────────────────
    data_root = Path(str(cfg.train.data_root)).resolve()
    cases = _discover_cases(
        data_root,
        str(cfg.train.case_filter.include_pattern),
        str(cfg.train.case_filter.exclude_pattern),
        logger,
    )
    if not cases:
        logger.warn("No valid case discovered, exit")
        return

    # ── Expand sweeps + deduplicate ────────────────────────────────
    run_specs = _expand_sweeps(cfg)
    executable_run_specs, alias_to_canonical = _deduplicate_run_specs_by_config(run_specs)
    logger.info(
        f"Study={study_name}, runs={len(run_specs)}, "
        f"executable={len(executable_run_specs)}, "
        f"aliases={len(alias_to_canonical)}, cases={len(cases)}"
    )

    # ── Phase 1: SETUP (create dirs + config.yaml) ────────────────
    base_config_path = Path(str(cfg.train.base_config))
    if not base_config_path.is_absolute():
        base_config_path = repo_root / base_config_path

    run_roots = write_case_configs(
        run_specs=executable_run_specs,
        cases=cases,
        results_root=results_root,
        study_name=study_name,
        base_config_path=base_config_path,
        static_overrides=dict(_to_plain(cfg.train.static_overrides)),
        seed=int(cfg.train.seed),
        logger=logger,
    )
    handle_aliases(alias_to_canonical, run_roots, logger)

    # ── Phase 2: RUN (dispatch fit across GPUs) ───────────────────
    pending = collect_pending_dirs(executable_run_specs, cases, results_root, study_name)
    if not pending:
        logger.info("All cases already completed, nothing to run")
    else:
        gpu_id = str(_to_plain(cfg.train.get("gpu", "0")))
        raw_gpus = _to_plain(cfg.train.get("gpus", []))
        gpus = [str(x) for x in raw_gpus] if isinstance(raw_gpus, list) else []
        if not gpus:
            gpus = [gpu_id] if gpu_id else ["0"]

        runner_cmd = shlex.split(str(cfg.train.runner))
        retries = int(cfg.train.retries)
        dry_run = bool(cfg.train.dry_run)
        skip_existing = bool(cfg.train.skip_existing)

        try:
            execute_dirs(
                dirs=pending,
                gpus=gpus,
                runner_cmd=runner_cmd,
                repo_root=repo_root,
                logger=logger,
                retries=retries,
                dry_run=dry_run,
                skip_existing=skip_existing,
            )
        except KeyboardInterrupt:
            logger.warn("Interrupted by Ctrl+C")
            _write_json(
                results_root / study_name / "study_summary.json",
                {
                    "study": study_name,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "num_runs": len(run_specs),
                    "num_cases": len(cases),
                    "interrupted": True,
                },
            )
            return

    # ── Phase 3: Summarise ────────────────────────────────────────
    generate_study_summary(
        run_specs=run_specs,
        alias_to_canonical=alias_to_canonical,
        results_root=results_root,
        study_name=study_name,
        cases=cases,
        logger=logger,
    )

    strict_match = bool(cfg.train.get("summary_require_exact_sweep_match", False))
    if strict_match and not _is_summary_sweep_match_exact(run_specs, results_root, study_name, logger):
        logger.warn("Skip summarise due to sweep mismatch")
    else:
        plot_metrics = list(_to_plain(cfg.train.get("summary_plot_metrics", [])))
        zero_as_missing = list(_to_plain(cfg.train.get("summary_zero_as_missing_metrics", [])))
        max_outliers = int(_to_plain(cfg.train.get("summary_max_outliers_per_run", 2)))
        run_summarize(
            results_root=results_root,
            study=study_name,
            plot_metrics=plot_metrics if plot_metrics else None,
            zero_as_missing_metrics=zero_as_missing if zero_as_missing else None,
            max_outliers_per_run=max_outliers,
        )

    logger.info(f"Done: study={study_name}, runs={len(run_specs)}, cases={len(cases)}")


if __name__ == "__main__":
    main()
