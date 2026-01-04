from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import torch
from gsplat.exporter import export_splats

from . import Saver, SaverModule
from internal.gaussian_splatting import GaussianSplatting
from internal.mp_strategy import MPStrategy
from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel
from internal.renderers.deformabel_xray_renderer import CoronaryDeformableXrayRenderer


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
        
        means3D = pc.get_xyz.clone()
        
        d_xyz, d_scaling, d_rotation, moving_probs = deform_model(
            means3D.detach(), 
            torch.zeros(means3D.shape[0], 1).to(means3D.device)
        )
        
        moving_mask = (moving_probs > 0.5).squeeze()
        if not torch.any(moving_mask):
            return
        
        def model_(key: str) -> torch.Tensor:
            return pc.gaussians[key]
        
        ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}.ply"
        
        gray = model_("gray")[moving_mask]
        sh0 = gray[..., None].repeat(1, 1, 3)
        export_splats(
            means=model_("means")[moving_mask],
            scales=model_("scales")[moving_mask],
            quats=model_("rotations")[moving_mask],
            opacities=model_("opacities")[moving_mask].squeeze(),
            sh0=sh0,
            shN=torch.zeros(gray.shape[0], 0, 3).to(sh0),
            save_to=str(ply_path)
        )
        