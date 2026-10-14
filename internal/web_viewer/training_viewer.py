"""Web viewer that runs *inside* the training loop.

Rendering must happen in the training thread (model state, CUDA context), while
the browser renders from its own thread. The two are connected by two queues,
following the design this repository already sketched::

    browser (ClientThread) --camera request--> request_queue
    training loop          --render-->         output_queue --> browser

Zero overhead when nobody is watching: the training loop only drains the request
queue with a non-blocking ``get_nowait()``.

Enable it with ``pixi run gs-fit <config> ... --viewer`` and open the forwarded
port; see the README (section 3.2) for the SSH port-forwarding recipe.
"""

from __future__ import annotations

import queue
import traceback
from queue import Queue

import numpy as np
import torch

from ..renderers.renderer import Renderer, RendererOutputs
from .new_viewer import NewViewer, NewViewerConfig
from .viewer_renderer import ViewerRenderer


class TrainingViewerRenderer(ViewerRenderer):
    """A :class:`ViewerRenderer` that renders through the training loop.

    Display handling (output selection, max clamp / colormap, visualizers) is
    inherited unchanged; only :meth:`get_outputs` is redirected: the requested
    camera is queued for the training loop and the raw :class:`RendererOutputs`
    come back through a second queue.
    """

    def __init__(
            self,
            renderer: Renderer,
            background_color,
            request_queue: Queue,
            output_queue: Queue,
            request_timeout: float = 15.0,
    ):
        super().__init__(gaussian_model=None, renderer=renderer, background_color=background_color)
        self.request_queue = request_queue
        self.output_queue = output_queue
        self.request_timeout = request_timeout

    def get_outputs(self, camera, scaling_modifier: float = 1.) -> torch.Tensor:
        # TODO: supports a single client at a time; with several connected
        # clients the responses could be matched to the wrong camera pose.
        self.request_queue.put((camera, scaling_modifier))
        try:
            render_outputs = self.output_queue.get(timeout=self.request_timeout)
        except queue.Empty:
            print(
                f"[web_viewer] render request timed out after {self.request_timeout:.0f}s "
                "(training loop busy?); returning an empty frame"
            )
            return torch.zeros(
                (3, max(int(camera.height), 1), max(int(camera.width), 1)),
                dtype=torch.float,
            )
        return self.visualize(self.output_name, render_outputs)


def _drain_render_requests(
        request_queue: Queue,
        output_queue: Queue,
        gaussian_model,
        renderer: Renderer,
        background_color,
        render_types: list | None,
        max_requests: int = 4,
        block_timeout: float = 0.0,
) -> int:
    """Service pending viewer render requests in the training thread.

    Returns the number of requests handled (``0`` when nobody is watching).

    ``block_timeout`` waits up to that long for the *first* request instead of
    returning immediately. The post-training loop in
    :class:`internal.callbacks.callbacks.KeepRunningIfWebViewerEnabled` calls
    this in a tight ``while True``, so it needs the wait to avoid spinning on a
    CPU core; during training the value stays ``0`` (never block the step).
    """
    device = gaussian_model.get_means().device
    handled = 0
    while handled < max_requests:
        try:
            if handled == 0 and block_timeout > 0:
                camera, scaling_modifier = request_queue.get(timeout=block_timeout)
            else:
                camera, scaling_modifier = request_queue.get_nowait()
        except queue.Empty:
            break
        handled += 1
        try:
            with torch.no_grad():
                output_queue.put(
                    renderer(
                        camera.to_device(device),
                        gaussian_model,
                        bg_color=background_color.to(device),
                        scaling_modifier=scaling_modifier,
                        render_types=render_types,
                    )
                )
        except Exception:
            traceback.print_exc()
    return handled


class TrainingViewer(NewViewer):
    """A :class:`NewViewer` driven by the training loop.

    Built by :meth:`internal.gaussian_splatting.GaussianSplatting.on_train_start`
    and serviced from ``on_train_batch_start`` / ``on_validation_batch_start``
    through :meth:`training_step` / :meth:`validation_step`.
    """

    def __init__(
            self,
            renderer: Renderer,
            background_color,
            up_direction: np.ndarray | None = None,
            host: str = "0.0.0.0",
            port: int = 8080,
            cameras_json: str | None = None,
            output_dir: str | None = None,
            show_cameras: bool = True,
            request_timeout: float = 15.0,
    ):
        config = NewViewerConfig(
            host=host,
            port=port,
            background_color=tuple(float(c) for c in background_color.detach().cpu().tolist()),
            up_direction=(
                tuple(float(c) for c in up_direction) if up_direction is not None else (0.0, 0.0, 1.0)
            ),
            cameras_json=cameras_json,
            show_cameras=show_cameras,
            output_dir=output_dir,
            # Keep the browser light while training is competing for the GPU.
            max_res_when_moving=1024,
            max_res_when_static=1600,
            jpeg_quality_when_moving=60,
            jpeg_quality_when_static=95,
        )

        self.request_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        # Set by `KeepRunningIfWebViewerEnabled` once training finished: the
        # render loop then waits for requests instead of busy-spinning.
        self.is_training_paused = False

        super().__init__(
            gaussian_model=None,
            renderer=renderer,
            config=config,
            viewer_renderer=TrainingViewerRenderer(
                renderer=renderer,
                background_color=background_color,
                request_queue=self.request_queue,
                output_queue=self.output_queue,
                request_timeout=request_timeout,
            ),
            in_training=True,
        )

    def process_all_render_requests(
            self, gaussian_model, renderer: Renderer, background_color=None, step: int = 0
    ) -> int:
        """Render every pending viewer request. Returns how many were handled.

        This is the public entry point expected by
        :class:`internal.callbacks.callbacks.KeepRunningIfWebViewerEnabled`,
        which keeps the server alive after training with a tight
        ``while True`` loop -- hence the short wait once
        :attr:`is_training_paused` is set, so it does not spin on a CPU core.
        The viewer's own background colour is used so the background slider
        keeps working.
        """
        return _drain_render_requests(
            request_queue=self.request_queue,
            output_queue=self.output_queue,
            gaussian_model=gaussian_model,
            renderer=renderer,
            background_color=self.viewer_renderer.background_color,
            render_types=[self.viewer_renderer.output_name],
            block_timeout=0.1 if self.is_training_paused else 0.0,
        )

    def training_step(self, gaussian_model, renderer: Renderer, background_color=None, step: int = 0) -> int:
        """Called on every training batch; services pending viewer requests.

        Returns the number of render requests handled (``0`` when nobody is
        watching), which is handy for logging or tests.
        """
        return self.process_all_render_requests(gaussian_model, renderer, background_color, step)

    def validation_step(self, gaussian_model, renderer: Renderer, background_color=None, step: int = 0) -> int:
        """Called on every validation batch; services pending viewer requests."""
        return self.process_all_render_requests(gaussian_model, renderer, background_color, step)


# ---------------------------------------------------------------------------
# Original sketch of this class (from upstream), kept for reference. It stays
# attached to a throw-away class body so the indented comments remain valid.
# ---------------------------------------------------------------------------
class _TrainingViewerSketch:
    """Never instantiated; see the historical sketch below."""

    
    #     self.host = host
    #     self.port = port
    #     self.image_format = "jpeg"
    #     self.enable_transform = False
    #     self.show_cameras = True
    #     self.show_edit_panel = False
    #     self.show_render_panel = False
    #     self.default_camera_position = None
    #     self.default_camera_look_at = None
    #     self.demo_mode = False
    #     self.enable_measurement = False
    #     self.camera_transform = torch.eye(4)
    #     self.device = torch.device("cpu")

    #     self.camera_names = camera_names
    #     self.cameras = cameras
    #     self.up_direction = up_direction
    #     self.camera_center = camera_center

    #     if available_appearance_options is not None:
    #         # convert key type to str
    #         available_appearance_options = {str(i): available_appearance_options[i] for i in available_appearance_options}
    #     self.available_appearance_options = available_appearance_options

    #     self.camera_queue = Queue()
    #     self.renderer_output_queue = Queue()
    #     self.viewer_renderer = TrainingViewerRenderer(self.camera_queue, self.renderer_output_queue)

    #     self.clients = {}

    #     self.is_training_paused = False

    # def add_cameras_to_scene(self, viser_server):
    #     self.camera_handles = []

    #     for idx in range(len(self.cameras)):
    #         camera = self.cameras[idx]
    #         name = self.camera_names[idx]

    #         rotation = camera.world_to_camera[:3, :3].clone()
    #         rotation[:3, 1:3] *= -1
    #         R = vtf.SO3.from_matrix(rotation.cpu().numpy())
    #         R = R @ vtf.SO3.from_x_radians(np.pi)

    #         cx = camera.width.item() // 2
    #         cy = camera.height.item() // 2
    #         fx = camera.fx.item()

    #         camera_handle = viser_server.add_camera_frustum(
    #             name="cameras/{}".format(name),
    #             fov=float(2 * np.arctan(cx / fx)),
    #             scale=0.1,
    #             aspect=float(cx / cy),
    #             wxyz=R.wxyz,
    #             position=camera.camera_center.cpu().numpy(),
    #             color=(205, 25, 0),
    #         )

    #         @camera_handle.on_click
    #         def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
    #             with event.client.atomic():
    #                 event.client.camera.position = event.target.position
    #                 event.client.camera.wxyz = event.target.wxyz

    #         self.camera_handles.append(camera_handle)

    #     self.camera_visible = True

    #     def toggle_camera_visibility(_):
    #         with viser_server.atomic():
    #             self.camera_visible = not self.camera_visible
    #             for i in self.camera_handles:
    #                 i.visible = self.camera_visible

    #     with viser_server.gui.add_folder("Cameras"):
    #         self.toggle_camera_button = viser_server.gui.add_button("Toggle Camera Visibility")
    #     self.toggle_camera_button.on_click(toggle_camera_visibility)

    # def setup_training_panel(self, viewer, server: viser.ViserServer):
    #     self.pause_training_button = server.gui.add_button("Pause Training", icon=viser.Icon.PLAYER_PAUSE_FILLED)

    #     @self.pause_training_button.on_click
    #     def _(_):
    #         self.pause_training_button.visible = False
    #         self.resume_training_button.visible = True
    #         self.is_training_paused = True

    #     self.resume_training_button = server.gui.add_button("Resume Training", icon=viser.Icon.PLAYER_PLAY_FILLED, visible=False)

    #     @self.resume_training_button.on_click
    #     def _(event):
    #         self.pause_training_button.visible = True
    #         self.resume_training_button.visible = False
    #         # mark training resumed
    #         self.is_training_paused = False
    #         # send None to camera queue to wake blocking thread up
    #         self.camera_queue.put((None, None))

    #     self.global_step_label = server.gui.add_markdown(content="Step: 0")

    #     self.render_frequency_slider = server.gui.add_slider("Render Freq", initial_value=10, min=1, max=100, step=1)

    # def start(self):
    #     super().start(False, server_config_fun=self.setup_training_panel, enable_renderer_options=False)
    #     # avoid keeping wasting GPU to render a higher resolution one
    #     self.max_res_when_static.value = self.max_res_when_moving.value
    #     self.jpeg_quality_when_static.value = 100
    #     self.jpeg_quality_when_moving.value = 100

    # def process_all_render_requests(self, gaussian_model, renderer, background_color):
    #     device = gaussian_model.get_xyz.device
    #     while True:
    #         try:
    #             if self.is_training_paused is True:
    #                 client_camera, scaling_modifier = self.camera_queue.get()
    #                 if client_camera is None:
    #                     break
    #             else:
    #                 client_camera, scaling_modifier = self.camera_queue.get_nowait()
    #         except queue.Empty:
    #             break

    #         try:
    #             with torch.no_grad():
    #                 self.renderer_output_queue.put(renderer(
    #                     client_camera.to_device(device),
    #                     gaussian_model,
    #                     bg_color=background_color.to(device),
    #                     scaling_modifier=scaling_modifier,
    #                 )["render"])
    #         except:
    #             traceback.print_exc()

    # def training_step(self, gaussian_model, renderer, background_color, step: int):
    #     self.global_step_label.content = f"Step: {step}"

    #     if self.is_training_paused is False:
    #         if self.camera_queue.empty() is True:
    #             if step % int(self.render_frequency_slider.value) == 0:
    #                 self.rerender_for_all_client()
    #             else:
    #                 return

    #     self.process_all_render_requests(gaussian_model, renderer, background_color)

    # def validation_step(self, gaussian_model, renderer, background_color, step: int):
    #     if self.camera_queue.empty() is True:
    #         return
    #     self.process_all_render_requests(gaussian_model, renderer, background_color)
