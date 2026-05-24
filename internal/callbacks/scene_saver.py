"""SceneSaver callback — wraps Saver logic as a Lightning callback.

Replaces the old ``SaveGaussian`` callback and the in-step save that was
embedded inside ``GaussianSplatting.training_step``.
"""

from __future__ import annotations

from typing import List, Optional

from jsonargparse.typing import lazy_instance
from lightning.pytorch.callbacks import Callback

from internal.savers.saver import Saver
from internal.savers.vanilla_saver import VanillaSaver


class SceneSaver(Callback):
    """Save scene (Gaussians + checkpoint + optional NIfTI) at configured steps.

    Parameters
    ----------
    save_iterations:
        Global-step milestones at which to trigger a save.
    saver:
        A ``Saver`` configuration (class_path / init_args dict) that will
        be instantiated in :meth:`setup`.
    """

    def __init__(self, save_iterations: Optional[List[int]] = None, saver: Saver = lazy_instance(VanillaSaver)) -> None:
        super().__init__()
        self._save_iterations = save_iterations if save_iterations is not None else [7_000, 30_000]
        self._saver_config = saver
        self._saver_module = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer, pl_module, stage: str) -> None:
        if stage != "fit":
            return
        self._saver_module = self._saver_config.instantiate()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if self._saver_module is None:
            return
        if trainer.global_step not in self._save_iterations:
            return
        # final step is handled by on_train_end
        if trainer.max_steps > 0 and trainer.global_step >= trainer.max_steps:
            return
        # don't re-save when resuming from a checkpoint at this exact step
        if trainer.global_step == getattr(pl_module, "restored_global_step", -1):
            return
        self._saver_module.save(pl_module)

    def on_train_end(self, trainer, pl_module) -> None:
        if self._saver_module is None:
            return
        self._saver_module.save(pl_module)
