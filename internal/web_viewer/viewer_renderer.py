"""Adapt a :class:`~internal.renderers.renderer.Renderer` to the Web viewer.

Current architecture
--------------------

A renderer declares what it can produce through
:meth:`~internal.renderers.renderer.Renderer.get_available_outputs`, and returns,
for every output, the **raw tensor together with the**
:class:`~internal.visualizers.Visualizer` that turns it into a displayable
image::

    RendererOutputs.images == {"gray": (tensor, FloatColormapVisualizer(...))}

So this class does not need the old ``build_viewer_output_visualizers``
machinery (which referenced helper classes that never existed in this
repository). It only has to:

* pick the requested output and apply the visualizer that came with it;
* expose the display controls (max clamp + colormap) for ``GRAY`` outputs, which
  is what both the reconstructed X-ray intensity and depth maps are;
* guarantee the returned tensor is a ``float (3, H, W)`` image in ``[0, 1]``.

For ``GRAY`` outputs the viewer takes over display normalization with
:class:`~internal.visualizers.DepthMapVisualizer` so the max-clamp and colormap
controls are always active. That reproduces the renderer's own
``FloatColormapVisualizer(GRAY)`` output while adding the controls.
"""

from typing import Tuple, Dict
import torch
from torch import Tensor

from ..renderers.renderer import (
    Renderer,
    RendererOutputInfo,
    RendererOutputs,
    RendererOutputTypes,
)
from ..visualizers import ColorMapName, DepthMapVisualizer, IdenticalVisualizer, Visualizer


class ViewerRenderer:
    def __init__(
            self,
            gaussian_model,
            renderer: Renderer,
            background_color,
            difix: bool = False,
    ):
        super().__init__()

        self.gaussian_model = gaussian_model
        self.renderer = renderer
        self.background_color = background_color

        self.max_depth = 0.
        # X-ray projections are grayscale, so default to a gray colormap; the
        # GUI dropdown can switch this to turbo/viridis/... for depth maps.
        self.depth_map_color_map = ColorMapName.GRAY

        # What the renderer can produce, and the currently selected one.
        self.available_outputs: Dict[str, RendererOutputInfo] = dict(self.renderer.get_available_outputs())
        if len(self.available_outputs) == 0:
            raise ValueError(f"{type(self.renderer).__name__}.get_available_outputs() returned nothing")
        first_name = next(iter(self.available_outputs))
        self.output_info: Tuple[str, RendererOutputInfo] = (first_name, self.available_outputs[first_name])

        self.difix = None
        self.difix_enabled = False
        if difix:
            self._setup_difix()

    @property
    def output_name(self) -> str:
        return self.output_info[0]

    @property
    def output_type(self) -> int:
        return self.output_info[1].type

    def set_output_info(self, name: str, renderer_output_info: RendererOutputInfo) -> None:
        self.output_info = (name, renderer_output_info)

    def _setup_depth_map_options(self, viewer, server):
        self.max_depth_gui_number = server.gui.add_number(
            label="Max Clamp",
            initial_value=self.max_depth,
            min=0.,
            step=0.01,
            hint="value=0 means that no max clamping, value will be normalized based on the maximum one",
            visible=False,
        )
        self.depth_map_color_map_dropdown = server.gui.add_dropdown(
            label="Color Map",
            options=[c.value for c in ColorMapName],
            initial_value=str(self.depth_map_color_map),
            visible=False,
        )

        @self.max_depth_gui_number.on_update
        @self.depth_map_color_map_dropdown.on_update
        def _(event):
            with server.atomic():
                self.max_depth = self.max_depth_gui_number.value
                self.depth_map_color_map = ColorMapName(self.depth_map_color_map_dropdown.value)
                viewer.rerender_for_all_client()

    def _set_depth_map_option_visibility(self, visible: bool):
        if getattr(self, "max_depth_gui_number", None) is None:
            return
        self.max_depth_gui_number.visible = visible
        self.depth_map_color_map_dropdown.visible = visible

    def _set_output_type(self, name: str, renderer_output_info: RendererOutputInfo):
        """
        Update properties
        """
        # toggle depth map option, only enable when type is `gray`
        self._set_depth_map_option_visibility(renderer_output_info.type == RendererOutputTypes.GRAY)

        if name not in self.available_outputs:
            raise ValueError(f"Unsupported output name `{name}`")

        # update
        self.set_output_info(name, renderer_output_info)

    def setup_options(self, viewer, server):
        self.available_outputs = dict(self.renderer.get_available_outputs())
        available_outputs = self.available_outputs
        first_type_name = list(available_outputs.keys())[0]

        with server.gui.add_folder("Output"):
            # setup output type dropdown
            output_type_dropdown = server.gui.add_dropdown(
                label="Type",
                options=list(available_outputs.keys()),
                initial_value=first_type_name,
            )
            self.output_type_dropdown = output_type_dropdown

            @output_type_dropdown.on_update
            def _(event):
                if event.client is None:
                    return
                with server.atomic():
                    # whether valid type
                    output_type_info = available_outputs.get(output_type_dropdown.value, None)
                    if output_type_info is None:
                        return

                    self._set_output_type(output_type_dropdown.value, output_type_info)

                    viewer.rerender_for_all_client()

            self._setup_depth_map_options(viewer, server)

        if self.difix is not None:
            # TODO: with reference views
            with server.gui.add_folder("DIFIX"):
                difix_checkbox = server.gui.add_checkbox(
                    "Enable",
                    initial_value=self.difix_enabled,
                )

                @difix_checkbox.on_update
                def _(event):
                    self.difix_enabled = difix_checkbox.value
                    self._set_output_type("rgb", available_outputs["rgb"])
                    viewer.rerender_for_all_client()

        # update default output type to the first one, must be placed after gui setup
        self._set_output_type(name=first_type_name, renderer_output_info=available_outputs[first_type_name])

    def _setup_difix(self):
        from internal.utils.pipeline_difix import DifixPipeline
        # self.difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        self.difix = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
        self.difix.to(self.gaussian_model.get_means().device)

    def get_outputs(self, camera, scaling_modifier: float = 1.) -> Tensor:
        """Render ``camera`` and return a displayable ``(3, H, W)`` image in ``[0, 1]``."""
        return self.visualize(self.output_name, self.get_raw_output(camera, scaling_modifier))

    def get_raw_output(self, camera, scaling_modifier: float = 1.) -> RendererOutputs:
        """Render ``camera`` and return the untouched :class:`RendererOutputs`."""
        return self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
            render_types=[self.output_name],
        )

    def _resolve_visualizer(self, name: str, visualizer: Visualizer | None) -> Visualizer:
        output_info = self.available_outputs.get(name)
        if output_info is not None and output_info.type == RendererOutputTypes.GRAY:
            # Intensity / depth: the viewer drives the display so that the max
            # clamp and colormap controls always take effect.
            return DepthMapVisualizer(
                max_depth=self.max_depth if self.max_depth > 0 else None,
                colormap=self.depth_map_color_map,
            )
        if visualizer is not None:
            # RGB / normal / feature outputs: trust the renderer.
            return visualizer
        return IdenticalVisualizer()

    def visualize(self, name: str, render_outputs: RendererOutputs) -> Tensor:
        """Turn one entry of ``render_outputs.images`` into a ``(3, H, W)`` image."""
        images = render_outputs.images
        if name not in images:
            # The renderer ignored `render_types`; fall back to what it produced.
            name = next(iter(images))

        image, visualizer = images[name]
        image = self._resolve_visualizer(name, visualizer).process(image)

        if image.dim() == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]

        return torch.clamp(image, min=0., max=1.)
