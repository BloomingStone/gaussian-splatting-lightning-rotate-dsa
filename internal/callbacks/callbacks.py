"""General Lightning callbacks used across experiments.

These are the callbacks that do NOT deal with validation-image saving
(which lives in ``internal.callbacks.save_image``) or scene saving
(which lives in ``internal.callbacks.scene_saver``).
"""

import math
import os
import sys

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm



# ---------------------------------------------------------------------------
#  KeepRunningIfWebViewerEnabled
# ---------------------------------------------------------------------------

class KeepRunningIfWebViewerEnabled(Callback):
    def on_train_end(self, trainer, pl_module) -> None:
        if pl_module.web_viewer is None:
            return
        print("Training finished! Web viewer is still running. Press `Ctrl+C` to exist.")
        while True:
            pl_module.web_viewer.is_training_paused = True
            pl_module.web_viewer.process_all_render_requests(
                pl_module.gaussian_model, pl_module.renderer, pl_module.background_color,
            )


# ---------------------------------------------------------------------------
#  ProgressBar
# ---------------------------------------------------------------------------

class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position + 1)
        self.on_epoch_metrics = {}

    def get_metrics(self, trainer, model):
        items = trainer._logger_connector.metrics["pbar"]
        return items

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        self.max_epochs = trainer.max_epochs
        if self.max_epochs < 0:
            self.max_epochs = math.ceil(trainer.max_steps / self.total_train_batches)

        self.epoch_progress_bar = Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position) - 1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            total=self.max_epochs,
        )
        self.epoch_progress_bar.update(trainer.current_epoch)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))
        self.epoch_progress_bar.set_postfix(self.on_epoch_metrics)
        self.epoch_progress_bar.update()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self.on_epoch_metrics.update(self.get_metrics(trainer, pl_module))


# ---------------------------------------------------------------------------
#  ValidateOnTrainEnd
# ---------------------------------------------------------------------------

class ValidateOnTrainEnd(Callback):
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.check_val_every_n_epoch is None:
            return
        if trainer.is_last_batch is False or trainer.current_epoch % trainer.check_val_every_n_epoch != 0:
            trainer.validating = True
            trainer._evaluation_loop.run()
            trainer.validating = False


