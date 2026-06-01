from dataclasses import dataclass
import os
import numpy as np
import torch

from .saver import Saver, SaverModule
from lightning import LightningModule


@dataclass
class VanillaSaver(Saver):
    save_vtp: bool = True

    def instantiate(self, *args, **kwargs) -> "VanillaSaverModule":
        return VanillaSaverModule(self)

class VanillaSaverModule(SaverModule):
    def __init__(self, config: VanillaSaver):
        super().__init__()
        self.config = config
    
    
    def save(self, pl_module: LightningModule):
        if pl_module.trainer.global_rank != 0:
            return

        if self.config.save_vtp is True:
            # save vtp file
            output_dir = os.path.join(pl_module.hparams["output_path"], "point_cloud",
                                      "iteration_{}".format(pl_module.trainer.global_step))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "point_cloud.vtp")
            with torch.no_grad():
                pl_module.gaussian_model.save_to_vtp(output_path)

            print("Gaussians saved to {}".format(output_path))

        # save checkpoint
        checkpoint_path = os.path.join(
            pl_module.hparams["output_path"],
            "checkpoints",
            f"epoch={pl_module.trainer.current_epoch}-step={pl_module.trainer.global_step}.ckpt",
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        pl_module.trainer.save_checkpoint(checkpoint_path)

        # save a simple xyz preview as vtp
        try:
            import pyvista as pv
            with torch.no_grad():
                xyz = pl_module.gaussian_model.get_xyz.detach().cpu().numpy().astype(np.float32)
                pd = pv.PolyData(xyz)
                pd.save(os.path.join(
                    pl_module.hparams["output_path"],
                    "checkpoints",
                    f"epoch={pl_module.trainer.current_epoch}-step={pl_module.trainer.global_step}-xyz.vtp",
                ))
        except Exception:
            pass  # xyz preview is optional

        print("Checkpoint saved to {}".format(checkpoint_path))