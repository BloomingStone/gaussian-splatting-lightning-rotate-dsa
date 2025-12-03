from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

import torch
from gsplat.exporter import export_splats

from . import Saver, SaverModule
from lightning import LightningModule
from internal.mp_strategy import MPStrategy
from internal.models.xray_coronary_gaussian import XrayCoronaryGaussianModel, XrayGassianState


@dataclass
class XRaySaver(Saver):
    save_states: list[Literal["coronary", "background", "whole"]] = field(default_factory=lambda: ["coronary"])
    def instantiate(self, *args, **kwargs) -> "XRaySaverModule":
        return XRaySaverModule(self)

class XRaySaverModule(SaverModule):
    def __init__(self, config: XRaySaver):
        super().__init__()
        self.config = config
    
    def save(self, pl_module: LightningModule):
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
        model = pl_module.gaussian_model
        for state in self.config.save_states:
            ply_path = ckpt_dir / f"epoch={epoch}-step={step}{ckpt_suffix}-{state}.ply"
            model.state = XrayGassianState(state)
            
            # TODO
            opacity = model.opacities.detach()
            gray = torch.sigmoid(opacity).squeeze()
            sh0 = gray[..., None, None].repeat(1, 1, 3)
            export_splats(
                means=model.get_xyz.detach(),
                scales=model.get_scaling.detach(),
                quats=model.get_rotation.detach(),
                opacities=model.get_opacity.squeeze().detach(),
                sh0=sh0,
                shN=torch.zeros(gray.shape[0], 0, 3).to(sh0),
                save_to=str(ply_path)
            )
        model.state = XrayGassianState.WHOLE
        