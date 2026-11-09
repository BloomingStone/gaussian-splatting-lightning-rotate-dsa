from dataclasses import dataclass
import lightning

import torch
from typing import Any, Callable, Dict
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..instantiate_config import Instantiable
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from ..visualizers import Visualizer


class RendererOutputTypes:
    """Semantic type of a renderer output.

    Consumers (the Web viewer, ``render.py``) use it to decide which display
    options make sense for an output -- e.g. the depth/intensity clamping and
    colormap controls are only offered for ``GRAY`` outputs.
    """

    RGB: int = 1
    GRAY: int = 2
    NORMAL_MAP: int = 3
    FEATURE_MAP: int = 4
    OTHER: int = 65535  # must provide a visualizer


#: Legacy signature of a visualizer attached to a :class:`RendererOutputInfo`.
#:
#: Prefer the per-output :class:`internal.visualizers.Visualizer` that renderers
#: already put next to each image in :attr:`RendererOutputs.images`; that one is
#: built by the renderer and therefore guaranteed to match its output.
RendererOutputVisualizer = Callable[[torch.Tensor, Dict, "RendererOutputInfo"], torch.Tensor]


@dataclass
class RendererOutputInfo:
    """Describes one entry of :attr:`RendererOutputs.images`."""

    key: str
    """Key used to retrieve the value from the dict returned by ``forward()``."""

    type: int = RendererOutputTypes.RGB
    """One of :class:`RendererOutputTypes`."""

    visualizer: RendererOutputVisualizer | None = None
    """Legacy callable visualizer. The first parameter is the value retrieved
    from the dict returned by ``forward()``, the second is that whole dict and
    the third is this :class:`RendererOutputInfo` instance."""

    other_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.type == RendererOutputTypes.OTHER and self.visualizer is None:
            raise ValueError("`visualizer` must be provided when `type` is `OTHER`")


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

    def initialize_from_checkpoint(self, checkpoint: dict) -> None:
        """
        Hook to restore renderer-owned submodules (e.g. a deform model) from a
        checkpoint produced by a previous training stage.  No-op by default.
        """
        pass

    def get_available_outputs(self) -> dict[str, RendererOutputInfo]:
        """Outputs this renderer can produce, keyed by the name used in
        :attr:`RendererOutputs.images`.

        Concrete renderers should override this so that offline consumers (the
        Web viewer, ``render.py``) know what can be requested. The default
        matches the vanilla renderer's single ``rgb`` output.
        """
        return {"rgb": RendererOutputInfo("render")}

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        pass


@dataclass
class RendererConfig(Instantiable):
    def instantiate(self, *args, **kwargs) -> Renderer:
        raise NotImplementedError()
