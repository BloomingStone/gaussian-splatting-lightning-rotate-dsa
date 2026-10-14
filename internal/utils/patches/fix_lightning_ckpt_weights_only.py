"""Make loading this repository's own checkpoints work with modern torch.

``torch.load`` tightened its ``weights_only`` default in torch >= 2.6. This
repository's checkpoints legitimately pickle ``numpy`` scalars into their
hyper-parameters, so the restricted (``weights_only=True``) loader refuses them
with::

    _pickle.UnpicklingError: Weights only load failed ...
    Unsupported global: GLOBAL numpy._core.multiarray.scalar

Two places are affected:

1. :meth:`LightningCLI._parse_ckpt_path` reads the checkpoint with a
   **hardcoded** ``weights_only=True`` to restore the hyper-parameters. It
   therefore raises *before* the run starts, which is why passing
   ``--ckpt_path`` used to be impossible.
2. The ``--weights_only`` argument (auto-generated from
   ``Trainer.fit/validate/test``) defaults to ``None``, i.e. "use torch's
   default" -- the restricted loader -- so the real checkpoint load failed as
   well and ``--weights_only false`` had to be passed explicitly.

Both are switched to the permissive loader here, so the common case just works
again. The behaviour is still overridable: passing ``--weights_only true``
restores the restricted loader.
"""

from typing import Any

import torch
from lightning.pytorch.cli import LightningCLI

_original_parse_ckpt_path = getattr(LightningCLI, "_parse_ckpt_path", None)


def _parse_ckpt_path(self) -> None:
    """``LightningCLI._parse_ckpt_path`` with the permissive loader.

    Rather than duplicating lightning's implementation (which would silently
    diverge if it changes), ``torch.load`` is redirected for the duration of the
    original call only.
    """
    original_torch_load = torch.load

    def _permissive_torch_load(*args: Any, **kwargs: Any):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _permissive_torch_load
    try:
        return _original_parse_ckpt_path(self)
    finally:
        torch.load = original_torch_load


if _original_parse_ckpt_path is not None:
    LightningCLI._parse_ckpt_path = _parse_ckpt_path


_original_prepare_subcommand_kwargs = getattr(
    LightningCLI, "_prepare_subcommand_kwargs", None
)


def _prepare_subcommand_kwargs(self, subcommand: str) -> dict:
    """Default ``weights_only`` to ``False`` for the subcommand that is run.

    Only an unset (``None``) value is replaced, so an explicit
    ``--weights_only true/false`` from the user is always honoured.
    """
    kwargs = _original_prepare_subcommand_kwargs(self, subcommand)
    if "weights_only" in kwargs and kwargs["weights_only"] is None:
        kwargs["weights_only"] = False
    return kwargs


if _original_prepare_subcommand_kwargs is not None:
    LightningCLI._prepare_subcommand_kwargs = _prepare_subcommand_kwargs
