from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import viser

from ..models.gaussian import GaussianModel
from ..renderers.renderer import Renderer
from .client import ClientThread
from .viewer_renderer import ViewerRenderer


@dataclass
class NewViewerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    image_format: Literal["jpeg", "png"] = "jpeg"
    up_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    default_camera_position: Tuple[float, float, float] | None = None
    default_camera_look_at: Tuple[float, float, float] | None = None


@dataclass
class _Value:
    value: float


class NewViewer:
    """Minimal web viewer skeleton.

    Only the GaussianModel and Renderer are injected.
    All extra UI and compatibility branches are intentionally left out.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        renderer: Renderer,
        config: NewViewerConfig | None = None,
    ) -> None:
        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.config = config or NewViewerConfig()

        self.device = self._infer_device(gaussian_model)
        self.background_color = torch.tensor(self.config.background_color, dtype=torch.float, device=self.device)

        self.host = self.config.host
        self.port = self.config.port
        self.image_format = self.config.image_format
        self.up_direction = np.asarray(self.config.up_direction, dtype=np.float32)
        self.default_camera_position = self.config.default_camera_position
        self.default_camera_look_at = self.config.default_camera_look_at

        self.camera_transform = torch.eye(4, dtype=torch.float)
        self.camera_center = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

        self.clients: dict[int, ClientThread] = {}

        # These are the only viewer-side values ClientThread needs.
        self.max_res_when_static = _Value(1920)
        self.jpeg_quality_when_static = _Value(100)
        self.max_res_when_moving = _Value(1280)
        self.jpeg_quality_when_moving = _Value(60)
        self.scaling_modifier = _Value(1.0)
        self.time_slider = _Value(0.0)

        self.viewer_renderer = ViewerRenderer(
            gaussian_model=self.gaussian_model,
            renderer=self.renderer,
            background_color=self.background_color,
        )

        if hasattr(self.gaussian_model, "eval"):
            self.gaussian_model.eval()
        if hasattr(self.renderer, "eval"):
            self.renderer.eval()

        # TODO: add renderer-specific controls only when you really need them.
        # TODO: call renderer.setup(...) yourself if your renderer needs stage-specific initialization.
        # TODO: add camera reorientation if your dataset requires it.

    @staticmethod
    def _infer_device(gaussian_model: GaussianModel) -> torch.device:
        try:
            return next(gaussian_model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rerender_for_client(self, client_id: int) -> None:
        client = self.clients.get(client_id)
        if client is None:
            return
        client.state = "low"
        client.render_trigger.set()

    def rerender_for_all_client(self) -> None:
        for client_id in list(self.clients.keys()):
            self.rerender_for_client(client_id)

    def get_appearance_id_value(self) -> tuple[int, float]:
        # TODO: expose appearance selection here if your renderer needs it.
        return (0, 0.0)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        thread = ClientThread(self, self.viewer_renderer, client)
        self.clients[client.client_id] = thread
        thread.start()

    def _handle_client_disconnect(self, client: viser.ClientHandle) -> None:
        thread = self.clients.pop(client.client_id, None)
        if thread is not None:
            thread.stop()

    def start(self, block: bool = True) -> None:
        server = viser.ViserServer(host=self.host, port=self.port)
        self._server = server
        server.scene.set_up_direction(self.up_direction)
        server.gui.configure_theme(control_layout="collapsible", show_logo=False)

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("General"):
            with server.gui.add_folder("Render"):
                self.viewer_renderer.setup_options(self, server)

        # TODO: call renderer.setup_web_viewer_tabs(self, server, tabs) if you want custom renderer UI.

        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        if block is True:
            while True:
                time.sleep(999)
