"""Validation / test image saving callback.

Moved out of ``GaussianSplatting`` so that image-saving parameters can be
configured directly in YAML via ``trainer.callbacks``.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback

import torch
import torchvision
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from internal.utils.image_utils import save_tensor_image


class SaveImage(Callback):
    """Save validation / test images to disk and (optionally) to the logger.

    Parameters
    ----------
    save_val_output:
        Master switch – when ``False`` no images are saved at all.
    max_save_val_output:
        Maximum number of batches to save per epoch (-1 = unlimited).
    max_image_saving_threads:
        Number of background threads used for writing images to disk.
    """

    def __init__(
        self,
        save_val_output: bool = False,
        max_save_val_output: int = -1,
        max_image_saving_threads: int = 16,
    ) -> None:
        super().__init__()
        self.save_val_output = save_val_output
        self.max_save_val_output = max_save_val_output
        self.max_image_saving_threads = max_image_saving_threads
        self.image_queue: queue.Queue = queue.Queue(maxsize=self.max_image_saving_threads)
        self.image_saving_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_image(trainer, tag: str, image_tensor: torch.Tensor) -> None:
        if trainer.logger is None:
            return
        if isinstance(trainer.logger, TensorBoardLogger):
            trainer.logger.experiment.add_image(tag, image_tensor, trainer.global_step)
        elif isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({tag: wandb.Image(image_tensor)}, step=trainer.global_step)
        else:
            raise NotImplementedError(
                "Unsupported logger type for logging images: {}".format(type(trainer.logger))
            )

    def _save_image_item(self, trainer, pl_module, item) -> None:
        image_list = [item["gt_image"]]
        for key in sorted(item["output_images"].keys()):
            image_list.append(item["output_images"][key])

        image = torch.concat(image_list, dim=-1)
        grid = torchvision.utils.make_grid(image)
        self._log_image(
            trainer,
            tag="{}_images/{}".format(item["stage"], item["image_name"].replace("/", "_")),
            image_tensor=grid,
        )

        image_output_path = os.path.join(
            pl_module.hparams["output_path"],
            item["stage"],
            "epoch={}-step={}".format(item["epoch"], item["step"]),
            "{}.jpg".format(item["image_name"].replace("/", "_")),
        )
        os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
        save_tensor_image(image_output_path, image)

    def _save_images(self, trainer, pl_module) -> None:
        while True:
            item = self.image_queue.get()
            if item is None:
                break
            try:
                self._save_image_item(trainer, pl_module, item)
            except:  # noqa: E722
                traceback.print_exc()

    # ------------------------------------------------------------------
    #  Worker lifecycle
    # ------------------------------------------------------------------

    def _start_workers(self, trainer, pl_module) -> None:
        if self.save_val_output is False or len(self.image_saving_threads) > 0:
            return
        for _ in range(self.max_image_saving_threads):
            thread = threading.Thread(target=self._save_images, args=(trainer, pl_module))
            self.image_saving_threads.append(thread)
            thread.start()

    def _stop_workers(self) -> None:
        if len(self.image_saving_threads) == 0:
            return
        for _ in range(len(self.image_saving_threads)):
            self.image_queue.put(None)
        for thread in self.image_saving_threads:
            thread.join()
        self.image_saving_threads = []

    # ------------------------------------------------------------------
    #  Per-batch collection
    # ------------------------------------------------------------------

    def _handle_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, stage: str) -> None:
        if self.save_val_output is False or trainer.global_rank != 0:
            return
        if self.max_save_val_output >= 0 and batch_idx >= self.max_save_val_output:
            return
        if outputs is None:
            return

        _, image_info, _ = batch
        gt_image = image_info[1]
        if gt_image is None:
            return

        output_images = {}
        for key, (img, vis) in outputs.images.items():
            output_images[key] = img.cpu() if vis is None else vis.process(img.cpu())

        self.image_queue.put({
            "output_images": output_images,
            "gt_image": gt_image.cpu(),
            "stage": stage,
            "image_name": image_info[0],
            "epoch": max(trainer.current_epoch, pl_module.restored_epoch),
            "step": max(trainer.global_step, pl_module.restored_global_step),
        })

    # ------------------------------------------------------------------
    #  Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._start_workers(trainer, pl_module)

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._start_workers(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._handle_batch_end(trainer, pl_module, outputs, batch, batch_idx, stage="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        self._handle_batch_end(trainer, pl_module, outputs, batch, batch_idx, stage="test")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._stop_workers()

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._stop_workers()

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._stop_workers()
