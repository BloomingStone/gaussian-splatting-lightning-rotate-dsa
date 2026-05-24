"""Configuration loading: sweep expansion, case discovery, deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from ._logging import Logger
from ._types import RunSpec
from ._utils import _hash_params, _to_plain


def _sanitize(text: str) -> str:
    """Replace non-alphanumeric characters with hyphens."""
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-", ".", "="):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)


def _short_grid_key(key: str) -> str:
    """Shorten a dotted config key to its last component (max 3 chars)."""
    tail = str(key).split(".")[-1]
    return _sanitize(tail[:3])


def _short_grid_value(value: Any) -> str:
    """Shorten a grid value for use in run names."""
    if isinstance(value, bool):
        return "T" if value else "F"

    if isinstance(value, (int, float)):
        sci = f"{float(value):.2e}"  # e.g. 1.23e+04
        mantissa, exponent = sci.split("e")
        return _sanitize(f"{mantissa}e{int(exponent)}")

    return _sanitize(str(value)[:3])


def _grid_combo_tag(keys: list[str], row: dict[str, Any]) -> str:
    """Build a compact tag string for a grid combination, e.g. 't_m-1e2_x_m-5e1'."""
    parts: list[str] = []
    for key in keys:
        parts.append(f"{_short_grid_key(key)}-{_short_grid_value(row[key])}")
    return "_".join(parts)


def _discover_cases(
    data_root: Path,
    include_pattern: str,
    exclude_pattern: str,
    logger: Logger,
) -> list[Path]:
    """Scan data_root for valid case directories.

    A valid case must contain:
      - rotate_dsa.json
      - depth_map.npz
      - rotate_dsa/*.png
      - label/*.png
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    cases: list[Path] = []
    for p in sorted([x for x in data_root.iterdir() if x.is_dir()]):
        name = p.name
        if include_pattern and include_pattern not in name:
            continue
        if exclude_pattern and exclude_pattern in name:
            continue

        if not (p / "rotate_dsa.json").is_file():
            logger.error(f"Skip invalid case {name}: missing rotate_dsa.json")
            continue
        if not (p / "depth_map.npz").is_file():
            logger.error(f"Skip invalid case {name}: missing depth_map.npz")
            continue
        if not any((p / "rotate_dsa").glob("*.png")):
            logger.error(f"Skip invalid case {name}: missing rotate_dsa/*.png")
            continue
        if not any((p / "label").glob("*.png")):
            logger.error(f"Skip invalid case {name}: missing label/*.png")
            continue

        cases.append(p)
    return cases


def _expand_sweeps(cfg: DictConfig) -> list[RunSpec]:
    """Expand sweeps configuration into a flat list of RunSpecs.

    Supports two modes:
      - grid: Cartesian product of all grid dimensions.
      - single-variable: one RunSpec per (variable, value) pair.
    """
    sweeps = _to_plain(cfg.sweeps)
    run_specs: list[RunSpec] = []

    for sweep in sweeps:
        sweep_name = str(sweep["name"])
        mode = str(sweep.get("mode", "grid"))
        fixed = dict(sweep.get("fixed", {}))
        grid: dict[str, list[Any]] = {
            k: list(v) for k, v in dict(sweep.get("grid", {})).items()
        }

        if mode == "grid":
            keys = sorted(grid.keys())
            combinations: list[dict[str, Any]] = [dict()]
            for k in keys:
                next_rows: list[dict[str, Any]] = []
                for row in combinations:
                    for v in grid[k]:
                        new_row = dict(row)
                        new_row[k] = v
                        next_rows.append(new_row)
                combinations = next_rows

            for row in combinations:
                params = {**fixed, **row}
                token = _hash_params(params)
                combo_tag = _grid_combo_tag(keys, row)
                run_name = (
                    f"{_sanitize(sweep_name)}__{combo_tag}__{token}"
                    if combo_tag
                    else f"{_sanitize(sweep_name)}__{token}"
                )
                run_specs.append(RunSpec(name=run_name, params=params))
            continue

        if mode == "single-variable":
            for k, values in sorted(grid.items()):
                for v in values:
                    params = dict(fixed)
                    params[k] = v
                    token = _hash_params(params)
                    run_specs.append(
                        RunSpec(
                            name=f"{_sanitize(sweep_name)}__sv-{_sanitize(k)}-{_sanitize(str(v))}__{token}",
                            params=params,
                        )
                    )
            continue

        raise ValueError(f"Unknown sweep mode: {mode}")

    # Guard against duplicate run names.
    seen: set[str] = set()
    dup_names: set[str] = set()
    for spec in run_specs:
        if spec.name in seen:
            dup_names.add(spec.name)
        seen.add(spec.name)
    if dup_names:
        dups = ", ".join(sorted(dup_names))
        raise ValueError(f"Duplicate run_name detected in sweeps: {dups}")

    return run_specs


def _deduplicate_run_specs_by_config(
    run_specs: list[RunSpec],
) -> tuple[list[RunSpec], dict[str, str]]:
    """Remove duplicate RunSpecs that share the same parameter hash.

    Returns (canonical_specs, alias_to_canonical) where alias_to_canonical
    maps duplicated run names to their canonical counterpart.
    """
    token_to_canonical_name: dict[str, str] = {}
    canonical_specs: list[RunSpec] = []
    alias_to_canonical: dict[str, str] = {}

    for spec in run_specs:
        token = _hash_params(spec.params)
        if token not in token_to_canonical_name:
            token_to_canonical_name[token] = spec.name
            canonical_specs.append(spec)
            continue
        alias_to_canonical[spec.name] = token_to_canonical_name[token]

    return canonical_specs, alias_to_canonical
