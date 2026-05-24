"""Pure utility functions shared across the experiment_manager package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _to_plain(obj: Any) -> Any:
    """Convert OmegaConf containers to plain Python objects."""
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _hash_params(data: dict[str, Any]) -> str:
    """Generate a short hash from parameter dict for deduplication."""
    text = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _write_json(path: Path, data: Any) -> None:
    """Write data as JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(v: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _extract_step(path: Path) -> int:
    """Extract the step number from a filename like '...step=12345...'."""
    stem = path.stem
    idx = stem.find("step=")
    if idx < 0:
        return -1
    idx += len("step=")
    digits = []
    while idx < len(stem) and stem[idx].isdigit():
        digits.append(stem[idx])
        idx += 1
    if not digits:
        return -1
    return int("".join(digits))


def _latest_by_step(paths: list[Path]) -> Path | None:
    """Return the path with the largest step number."""
    if not paths:
        return None
    return sorted(paths, key=lambda p: (_extract_step(p), p.name))[-1]



