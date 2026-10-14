from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import viser
import viser.transforms as vtf

from ..cameras import Cameras, Camera
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

    cameras_json: str | None = None
    """Path to a ``cameras.json`` written by a training run. The cameras are
    drawn as frustums and define the time-slider range."""

    show_cameras: bool = True
    """Draw the training cameras as frustums (requires :attr:`cameras_json`)."""

    time_max: float = 1.0
    """Upper bound of the time slider. Camera ``time`` values are normalised to
    ``[0, 1]`` by the camera builders, so ``1.0`` is the right default."""

    output_dir: str | None = None
    """Directory the "Save frame" button writes to. Defaults to ``./outputs``."""

    max_res_when_moving: int = 1280
    max_res_when_static: int = 1920
    jpeg_quality_when_moving: int = 60
    jpeg_quality_when_static: int = 100


@dataclass
class _Value:
    value: float


@dataclass
class _GtCamera:
    """One camera of the training run, as written to ``cameras.json``.

    ``entry`` keeps the raw record so a renderable
    :class:`~internal.cameras.Camera` can be rebuilt from it (see
    :meth:`NewViewer._camera_from_entry`).
    """

    name: str
    position: Tuple[float, float, float]
    wxyz: Tuple[float, float, float, float]
    fov: float
    aspect: float
    time: float
    entry: dict


class NewViewer:
    """Minimal web viewer skeleton.

    Only the GaussianModel and Renderer are injected.
    All extra UI and compatibility branches are intentionally left out.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel | None,
        renderer: Renderer,
        config: NewViewerConfig | None = None,
        viewer_renderer: ViewerRenderer | None = None,
        in_training: bool = False,
    ) -> None:
        """
        Args:
            gaussian_model: model to render. May be ``None`` when
                ``viewer_renderer`` already knows how to reach the model (e.g.
                the in-training viewer, where the training thread owns it).
            viewer_renderer: override the renderer adapter, e.g.
                :class:`~internal.web_viewer.training_viewer.TrainingViewerRenderer`.
            in_training: do not switch the model / renderer to eval mode; they
                belong to the running training loop.
        """
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
        self.max_res_when_static = _Value(self.config.max_res_when_static)
        self.jpeg_quality_when_static = _Value(self.config.jpeg_quality_when_static)
        self.max_res_when_moving = _Value(self.config.max_res_when_moving)
        self.jpeg_quality_when_moving = _Value(self.config.jpeg_quality_when_moving)
        self.scaling_modifier = _Value(1.0)
        self.time_slider = _Value(0.0)

        self.viewer_renderer = viewer_renderer or ViewerRenderer(
            gaussian_model=self.gaussian_model,
            renderer=self.renderer,
            background_color=self.background_color,
        )

        if in_training is False:
            if hasattr(self.gaussian_model, "eval"):
                self.gaussian_model.eval()
            if hasattr(self.renderer, "eval"):
                self.renderer.eval()

        # Cameras of the training run: drawn as frustums and used for the
        # time slider range. Optional (`cameras_json` may be None).
        self.gt_camera_names, self.gt_cameras = self._load_cameras(self.config.cameras_json)
        self.camera_frustum_handles = {}

        # Rendered-frame counter for the "Save frame" button.
        self._save_index = 0
        # Training camera selected in the GUI.
        self._camera_index = 0

    @staticmethod
    def _infer_device(gaussian_model: GaussianModel | None) -> torch.device:
        if gaussian_model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ------------------------------------------------------------------
    #  Training cameras
    # ------------------------------------------------------------------

    def _load_cameras(self, cameras_json: str | None) -> tuple[list[str], list[_GtCamera]]:
        """Load the ``cameras.json`` written by a training run (optional).

        :meth:`internal.datamodule.DataModule.setup` writes, per camera, the
        camera centre (``position``) and the **camera-to-world** rotation
        (``rotation``), together with the intrinsics. Both the frustum pose and
        a renderable camera can be rebuilt from that.
        """
        if cameras_json is None:
            return [], []

        path = Path(cameras_json)
        if not path.is_file():
            print(f"[viewer] cameras json not found, skipping frustums: {path}")
            return [], []

        with path.open("r") as f:
            entries = json.load(f)

        cameras: list[_GtCamera] = []
        names: list[str] = []
        for idx, entry in enumerate(entries):
            name = str(entry.get("img_name", entry.get("id", idx)))
            names.append(name)

            height = float(entry.get("height", 0) or 0)
            width = float(entry.get("width", 0) or 0)
            fy = float(entry.get("fy", entry.get("fx", 0.0)) or 0.0)
            if height <= 0 or width <= 0 or fy <= 0:
                continue

            r_c2w = np.asarray(entry["rotation"], dtype=np.float64)
            cameras.append(
                _GtCamera(
                    name=name,
                    position=tuple(np.asarray(entry["position"], dtype=np.float64).tolist()),
                    wxyz=tuple(vtf.SO3.from_matrix(r_c2w).wxyz.tolist()),
                    fov=2.0 * float(np.arctan(height * 0.5 / fy)),
                    aspect=width / height,
                    time=float(entry.get("time", 0.0)),
                    entry=entry,
                )
            )

        print(f"[viewer] loaded {len(cameras)} cameras from {path}")
        return names, cameras

    @staticmethod
    def _camera_from_entry(entry: dict) -> "Camera":
        """Rebuild a renderable :class:`~internal.cameras.Camera` from a
        ``cameras.json`` entry.

        ``rotation`` is camera-to-world, while the camera builders here expect
        the world-to-camera rotation/tranlation, i.e. ``R = rotationᵀ`` and
        ``T = -R·position``.
        """
        r_c2w = torch.tensor(entry["rotation"], dtype=torch.float)
        position = torch.tensor(entry["position"], dtype=torch.float)
        r_w2c = r_c2w.T
        t_w2c = -(r_w2c @ position)

        return Cameras.build(
            idx=torch.tensor([int(entry.get("id", 0))]),
            R=r_w2c[None],
            T=t_w2c[None],
            fx=torch.tensor([float(entry["fx"])]),
            fy=torch.tensor([float(entry["fy"])]),
            cx=torch.tensor([float(entry["cx"])]),
            cy=torch.tensor([float(entry["cy"])]),
            width=torch.tensor([float(entry["width"])]),
            height=torch.tensor([float(entry["height"])]),
            znear=0.01,
            zfar=1e5,
            time=torch.tensor([float(entry.get("time", 0.0))]),
            phase=torch.tensor([float(entry.get("phase", 0.0))]),
        )[0]

    def _jump_to_camera(self, index: int) -> None:
        """Move every connected client's camera onto a training camera pose.

        The client looks at the iso-centre, which is the centre of the
        reconstruction volume for the camera builders used here.
        """
        if not (0 <= index < len(self.gt_cameras)):
            return
        gt = self.gt_cameras[index]
        for client in self.clients.values():
            client.client.camera.position = np.asarray(gt.position)
            client.client.camera.look_at = np.zeros(3)
        self.rerender_for_all_client()
        print(f"[viewer] jumped to training camera {index} ({gt.name})")

    def add_cameras_to_scene(self, server: viser.ViserServer, visible: bool = True) -> None:
        """Draw the training cameras as frustums."""
        self.camera_frustum_handles = {}
        scale = self._frustum_scale()
        for gt in self.gt_cameras:
            self.camera_frustum_handles[gt.name] = server.scene.add_camera_frustum(
                name=f"/cameras/{gt.name}",
                fov=gt.fov,
                aspect=gt.aspect,
                scale=scale,
                color=(255, 165, 0),
                position=np.asarray(gt.position),
                wxyz=np.asarray(gt.wxyz),
                visible=visible,
            )

    def _frustum_scale(self) -> float:
        """Frustum size relative to the extent of the camera centres."""
        if len(self.gt_cameras) == 0:
            return 1.0
        centers = np.asarray([c.position for c in self.gt_cameras], dtype=np.float64)
        extent = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
        return max(extent * 0.02, 1e-3)

    def _set_cameras_visible(self, visible: bool) -> None:
        for handle in self.camera_frustum_handles.values():
            handle.visible = visible

    # ------------------------------------------------------------------
    #  GUI
    # ------------------------------------------------------------------

    def _setup_render_options(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Render"):
            # Output type + display controls (max clamp / colormap).
            self.viewer_renderer.setup_options(self, server)

            scaling = server.gui.add_slider(
                "Scaling", min=0.01, max=3.0, step=0.01, initial_value=1.0,
            )

            @scaling.on_update
            def _(_) -> None:
                self.scaling_modifier.value = scaling.value
                self.rerender_for_all_client()

    def _setup_resolution_options(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Resolution"):
            res_moving = server.gui.add_slider(
                "Max res (moving)", min=256, max=4096, step=64,
                initial_value=int(self.max_res_when_moving.value),
            )
            res_static = server.gui.add_slider(
                "Max res (static)", min=256, max=4096, step=64,
                initial_value=int(self.max_res_when_static.value),
            )
            quality_moving = server.gui.add_slider(
                "Quality (moving)", min=1, max=100, step=1,
                initial_value=int(self.jpeg_quality_when_moving.value),
            )
            quality_static = server.gui.add_slider(
                "Quality (static)", min=1, max=100, step=1,
                initial_value=int(self.jpeg_quality_when_static.value),
            )

            @res_moving.on_update
            @res_static.on_update
            @quality_moving.on_update
            @quality_static.on_update
            def _(_) -> None:
                self.max_res_when_moving.value = res_moving.value
                self.max_res_when_static.value = res_static.value
                self.jpeg_quality_when_moving.value = quality_moving.value
                self.jpeg_quality_when_static.value = quality_static.value
                self.rerender_for_all_client()

    def _setup_time_options(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Time"):
            time_slider = server.gui.add_slider(
                "Time", min=0.0, max=float(self.config.time_max), step=0.001, initial_value=0.0,
            )

            @time_slider.on_update
            def _(_) -> None:
                self.time_slider.value = time_slider.value
                self.rerender_for_all_client()

    def _setup_scene_options(self, server: viser.ViserServer) -> None:
        with server.gui.add_folder("Scene"):
            background = server.gui.add_rgb("Background", initial_value=(0, 0, 0))
            show_cameras = server.gui.add_checkbox(
                "Cameras", initial_value=bool(self.config.show_cameras and len(self.gt_cameras) > 0),
            )
            up_direction = server.gui.add_dropdown(
                "Up direction", options=("+Z (SI)", "+Y (AP)"), initial_value="+Z (SI)",
            )

            @background.on_update
            def _(_) -> None:
                color = tuple(float(c) for c in background.value)
                if max(color) > 1.0:
                    color = tuple(c / 255.0 for c in color)
                self.background_color = torch.tensor(color, dtype=torch.float, device=self.device)
                self.viewer_renderer.background_color = self.background_color
                self.rerender_for_all_client()

            @show_cameras.on_update
            def _(_) -> None:
                self._set_cameras_visible(bool(show_cameras.value))

            @up_direction.on_update
            def _(_) -> None:
                up = (0.0, 0.0, 1.0) if str(up_direction.value).startswith("+Z") else (0.0, 1.0, 0.0)
                self.up_direction = np.asarray(up, dtype=np.float32)
                server.scene.set_up_direction(self.up_direction)

            save_button = server.gui.add_button("Save frame")

            @save_button.on_click
            def _(_) -> None:
                self.save_frame()

    def _setup_camera_options(self, server: viser.ViserServer) -> None:
        """A control to jump the client camera onto a training pose."""
        if len(self.gt_cameras) == 0:
            return

        with server.gui.add_folder("Training cameras"):
            index_slider = server.gui.add_slider(
                "Index", min=0, max=len(self.gt_cameras) - 1, step=1, initial_value=0,
            )
            jump_button = server.gui.add_button("Jump to")

            @index_slider.on_update
            def _(_) -> None:
                self._camera_index = int(index_slider.value)

            @jump_button.on_click
            def _(_) -> None:
                self._camera_index = int(index_slider.value)
                self._jump_to_camera(self._camera_index)

    def save_frame(self) -> None:
        """Render the current client view (or the selected training camera) to a PNG."""
        usable = [c for c in self.clients.values() if c.last_camera is not None]
        if len(usable) > 0:
            client = usable[0]
            camera = ClientThread.get_camera(
                client.last_camera,
                image_size=int(self.max_res_when_static.value),
                time_value=float(self.time_slider.value),
                camera_transform=self.camera_transform,
            ).to_device(self.device)
        elif len(self.gt_cameras) > 0:
            # No browser attached: fall back to the selected training camera.
            camera = self._camera_from_entry(self.gt_cameras[self._camera_index].entry).to_device(self.device)
        else:
            print("[viewer] no client camera received yet, nothing saved")
            return

        with torch.no_grad():
            image = self.viewer_renderer.get_outputs(camera, scaling_modifier=self.scaling_modifier.value)

        output_dir = Path(self.config.output_dir or "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"viewer_frame_{self._save_index:04d}.png"
        self._save_index += 1

        import torchvision
        torchvision.utils.save_image(image.detach().cpu(), str(path))
        print(f"[viewer] saved {path}")

    # ------------------------------------------------------------------
    #  Server
    # ------------------------------------------------------------------

    def start(self, block: bool = True) -> None:
        server = viser.ViserServer(host=self.host, port=self.port)
        self._server = server
        server.scene.set_up_direction(self.up_direction)
        server.gui.configure_theme(control_layout="collapsible", show_logo=False)

        if len(self.gt_cameras) > 0:
            self.add_cameras_to_scene(server, visible=bool(self.config.show_cameras))

        tabs = server.gui.add_tab_group()
        with tabs.add_tab("General"):
            self._setup_render_options(server)
            self._setup_resolution_options(server)
            self._setup_time_options(server)
            self._setup_camera_options(server)
            self._setup_scene_options(server)

        # Renderer-specific tabs (no-op unless a renderer overrides
        # `setup_web_viewer_tabs`).
        with tabs.add_tab("Renderer"):
            self.viewer_renderer.renderer.setup_web_viewer_tabs(self, server, tabs)

        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        print(f"[viewer] serving on http://{self.host}:{self.port}")

        if block is True:
            while True:
                time.sleep(999)
