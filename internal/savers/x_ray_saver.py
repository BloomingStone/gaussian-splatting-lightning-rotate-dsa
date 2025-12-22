from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import torch
from gsplat.exporter import export_splats

from . import Saver, SaverModule
from internal.gaussian_splatting import GaussianSplatting
from internal.mp_strategy import MPStrategy
from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState
from internal.renderers.deformabel_xray_renderer import DeformModel, CoronaryDeformableXrayRenderer
from internal.utils.gaussian_utils import GaussianTransformUtils


@dataclass
class XRaySaver(Saver):
    save_states: list[Literal["coronary", "background"]] = field(default_factory=lambda: ["coronary"])
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
        model: XrayCoronaryGaussianModel = pl_module.gaussian_model
        renderer = pl_module.renderer
        assert isinstance(renderer, CoronaryDeformableXrayRenderer)
        deform_model: DeformModel = renderer.deform_model
        
        for state in self.config.save_states:
            ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}-{state}.ply"
            model.state = XrayGassianState(state)
            
            if XrayGassianState(state) == XrayGassianState.CORONARY:
                xyz = model.get_xyz.detach()
                phase = torch.zeros(xyz.shape[0], 1).to(xyz)
                d_xyz, d_scaling, d_rotation = deform_model(xyz, phase)
                means3D = model.get_xyz + d_xyz
                rotations: torch.Tensor = GaussianTransformUtils.quat_multiply(model.get_rotation, d_rotation)
                scales = model.get_scaling + d_scaling
            else:
                means3D = model.get_xyz
                rotations = model.get_rotation
                scales = model.get_scaling
            
            
            gray = model.get_features
            sh0 = gray[..., None].repeat(1, 1, 3)
            export_splats(
                means=means3D,
                scales=scales,
                quats=rotations,
                opacities=model.get_opacity.squeeze(),
                sh0=sh0,
                shN=torch.zeros(gray.shape[0], 0, 3).to(sh0),
                save_to=str(ply_path)
            )
        model.state = XrayGassianState.WHOLE
        