from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import torch

from . import Saver, SaverModule
from ..gaussian_splatting import GaussianSplatting
from ..mp_strategy import MPStrategy
from ..models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from ..renderers.deformabel_xray_renderer import CoronaryDeformableXrayRenderer
from ..utils.graphics_utils import store_ply


@dataclass
class XRaySaver(Saver):
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)

class XRaySaverModule(SaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__()
        self.config = config
    
    def save(self, pl_module: GaussianSplatting):
        is_mp_strategy = isinstance(pl_module.trainer.strategy, MPStrategy)
        if pl_module.trainer.global_rank != 0 and not is_mp_strategy:
            return

        epoch = pl_module.trainer.current_epoch
        step = pl_module.trainer.global_step
        output_root = Path(pl_module.hparams["output_path"])
        
        ckpt_dir = output_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        
        ckpt_suffix = f"-rank={pl_module.global_rank}" if is_mp_strategy else ""
        ckpt_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ckpt"
        
        pl_module.trainer.save_checkpoint(ckpt_path)
        
        assert isinstance(pl_module.gaussian_model, XrayCoronaryGaussianModel)
        pc = pl_module.gaussian_model
        
        assert isinstance(pl_module.renderer, CoronaryDeformableXrayRenderer)
        deform_model = pl_module.renderer.deform_model
        
        means3D = pc.get_means().clone()
        
        d_xyz, d_scaling, d_rotation, moving_probs = deform_model(
            means3D.detach(), 
            torch.zeros(means3D.shape[0], 1).to(means3D.device)
        )
        
        moving_mask = (moving_probs > 0.05).squeeze()
        if not torch.any(moving_mask):
            return
        
        ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ply"

        gray = torch.exp( - pc.get_density()[moving_mask])
        store_ply(
            path=str(ply_path),
            xyz=pc.get_means()[moving_mask].cpu().numpy(),
            rgb=(gray.clamp(min=0., max=1.)*255).to(torch.int).cpu().numpy()
        )
        