from dataclasses import dataclass
import lightning

import torch
from typing import Any
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..instantiate_config import Instantiable
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from ..visualizers import Visualizer

@dataclass
class RendererOutputs:
    images: dict[str, tuple[torch.Tensor, Visualizer|None]]
    meta: dict[str, torch.Tensor]


class Renderer(torch.nn.Module):
    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0
    ) -> RendererOutputs:
        raise NotImplementedError

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
    ) -> RendererOutputs:
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
        )

    def before_training_step(
            self,
            step: int,
            module,
    ):
        return

    def after_training_step(
            self,
            step: int,
            module,
    ):
        return

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def training_setup(self, module: lightning.LightningModule) -> tuple[list[Optimizer]|None, list[LRScheduler]|None]:
        return None, None

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        pass


@dataclass
class RendererConfig(Instantiable):
    def instantiate(self, *args, **kwargs) -> Renderer:
        raise NotImplementedError()
