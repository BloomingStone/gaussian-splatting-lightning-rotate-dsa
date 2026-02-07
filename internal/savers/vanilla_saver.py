from dataclasses import dataclass
import os
import torch

from . import Saver, SaverModule
from lightning import LightningModule
from ..mp_strategy import MPStrategy
from ..utils.sh_utils import eval_sh
from ..utils.graphics_utils import store_ply
from ..utils.gaussian_utils import GaussianPlyUtils


@dataclass
class VanillaSaver(Saver):
    def instantiate(self, *args, **kwargs) -> "VanillaSaverModule":
        return VanillaSaverModule(self)

class VanillaSaverModule(SaverModule):
    def __init__(self, config: VanillaSaver):
        super().__init__()
        self.config = config
    
    
    def save(self, pl_module: LightningModule):
        is_mp_strategy = isinstance(pl_module.trainer.strategy, MPStrategy)
        if pl_module.trainer.global_rank != 0 and is_mp_strategy is False:
            return

        if pl_module.hparams["save_ply"] is True:
            # save ply file
            filename = "point_cloud.ply"
            # if self.trainer.global_rank != 0:
            #     filename = "point_cloud_{}.ply".format(self.trainer.global_rank)
            with torch.no_grad():
                output_dir = os.path.join(pl_module.hparams["output_path"], "point_cloud",
                                          "iteration_{}".format(pl_module.trainer.global_step))
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, filename)
                GaussianPlyUtils.load_from_model(pl_module.gaussian_model).to_ply_format().save_to_ply(output_path + ".tmp")
                os.rename(output_path + ".tmp", output_path)

            print("Gaussians saved to {}".format(output_path))

        # save checkpoint
        checkpoint_name_suffix = ""
        if is_mp_strategy is True:
            checkpoint_name_suffix = f"-rank={pl_module.global_rank}"

        checkpoint_path = os.path.join(
            pl_module.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}{}.ckpt".format(pl_module.trainer.current_epoch, pl_module.trainer.global_step, checkpoint_name_suffix),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        pl_module.trainer.save_checkpoint(checkpoint_path)
        with torch.no_grad():
            xyz = pl_module.gaussian_model.get_xyz
            rgb = eval_sh(0, pl_module.gaussian_model.get_features[:, :1, :].transpose(1, 2), None)
            store_ply(os.path.join(
                pl_module.hparams["output_path"],
                "checkpoints",
                "epoch={}-step={}{}-xyz_rgb.ply".format(pl_module.trainer.current_epoch, pl_module.trainer.global_step, checkpoint_name_suffix),
            ), xyz.cpu().numpy(), ((rgb + 0.5).clamp(min=0., max=1.) * 255).to(torch.int).cpu().numpy())
        print("Checkpoint saved to {}".format(checkpoint_path))