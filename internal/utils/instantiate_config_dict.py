"""Instantiate jsonargparse-style ``{class_path, init_args}`` configuration trees.

When LightningCLI parses a config it builds the nested dataclass objects itself,
so the training path never has to do this. But the very same configurations are
stored as **plain dicts** in a checkpoint's ``hyper_parameters``::

    hyper_parameters["gaussian"] == {
        "class_path": "internal.models.xray_coronary_gaussian.XrayCoronaryGaussian",
        "init_args": {"optimization": {"means_lr_init": 1e-5, ...}},
    }

Every *offline* consumer -- the web viewer, ``render.py``, the 3D evaluation in
``scripts/experiment_manager`` -- has to rebuild those objects from the
checkpoint, which jsonargparse does not do for them. This module performs the
recursive rebuild.

The repository's configuration classes are plain dataclasses subclassing
:class:`internal.instantiate_config.Instantiable`, and every class involved only
takes keyword arguments, so ``cls(**init_args)`` is the correct construction.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping


def import_from_class_path(class_path: str) -> Any:
    """Import ``"pkg.module.ClassName"`` and return the attribute."""
    module_name, _, attribute = str(class_path).rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Invalid class_path (expected 'module.Class'): {class_path!r}")
    return getattr(importlib.import_module(module_name), attribute)


def instantiate_config_dict(value: Any) -> Any:
    """Recursively rebuild configuration objects from a parsed config tree.

    * a mapping containing ``class_path`` is turned into ``cls(**init_args)``
      (with ``init_args`` rebuilt recursively);
    * any other mapping / list / tuple is rebuilt element-wise so that nested
      configurations (e.g. ``optimization.means_lr_scheduler``) are handled too;
    * anything else (numbers, strings, ``None``, already-built objects) is
      returned unchanged.
    """
    if isinstance(value, Mapping):
        if "class_path" in value:
            cls = import_from_class_path(str(value["class_path"]))
            raw_init_args = value.get("init_args") or {}
            init_args = {key: instantiate_config_dict(item) for key, item in raw_init_args.items()}
            return cls(**init_args)
        return {key: instantiate_config_dict(item) for key, item in value.items()}

    if isinstance(value, list):
        return [instantiate_config_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(instantiate_config_dict(item) for item in value)

    return value
