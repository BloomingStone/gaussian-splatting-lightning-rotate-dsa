from typing import Tuple, Dict
import torch
from ..renderers.renderer import Renderer


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
        self.depth_map_color_map = "turbo"

        # TODO: initial value should get from renderer
        self.output_info: Tuple[str, RendererOutputInfo] = (
            "rgb",
            RendererOutputInfo("render"),
        )
        self.output_visualizers: Dict[str, OutputVisualizer] = build_viewer_output_visualizers(
            {"rgb": self.output_info[1]},
            max_depth_provider=lambda: self.max_depth,
            colormap_provider=lambda: self.depth_map_color_map,
            difix_pipeline_provider=lambda: self.difix,
            difix_enabled_provider=lambda: self.difix_enabled,
        )

        self.difix = None
        self.difix_enabled = False
        if difix:
            self._setup_difix()

    def set_output_info(
            self,
            name: str,
            renderer_output_info: RendererOutputInfo,
    ):
        self.output_info = (
            name,
            renderer_output_info,
        )

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
            options=["turbo", "viridis", "magma", "inferno", "cividis", "gray"],
            initial_value=self.depth_map_color_map,
            visible=False,
        )

        @self.max_depth_gui_number.on_update
        @self.depth_map_color_map_dropdown.on_update
        def _(event):
            with server.atomic():
                self.max_depth = self.max_depth_gui_number.value
                self.depth_map_color_map = self.depth_map_color_map_dropdown.value
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

        if name not in self.output_visualizers:
            raise ValueError(f"Unsupported output name `{name}`")

        # update
        self.set_output_info(name, renderer_output_info)

    def setup_options(self, viewer, server):
        available_outputs = self.renderer.get_available_outputs()
        first_type_name = list(available_outputs.keys())[0]
        self.output_visualizers = build_viewer_output_visualizers(
            available_outputs,
            max_depth_provider=lambda: self.max_depth,
            colormap_provider=lambda: self.depth_map_color_map,
            difix_pipeline_provider=lambda: self.difix,
            difix_enabled_provider=lambda: self.difix_enabled,
        )

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

    def get_outputs(self, camera, scaling_modifier: float = 1.):
        render_type, output_info = self.output_info

        render_outputs = self.renderer(
            camera,
            self.gaussian_model,
            self.background_color,
            scaling_modifier=scaling_modifier,
            render_types=[render_type],
        )
        image = self.output_visualizers[render_type](render_outputs)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image
