import argparse
from pathlib import Path

import torch

from internal.utils.gaussian_utils.gaussian_model_loader import GaussianModelLoader
from internal.web_viewer.new_viewer import NewViewer, NewViewerConfig


def _parse_background_color(value: list[str]) -> tuple[float, float, float]:
    """``black`` / ``white`` / three explicit numbers."""
    if len(value) == 1 and isinstance(value[0], str):
        return (1.0, 1.0, 1.0) if value[0] == "white" else (0.0, 0.0, 0.0)
    channels = [float(v) for v in value[:3]]
    while len(channels) < 3:
        channels.append(0.0)
    return (channels[0], channels[1], channels[2])


def _resolve_cameras_json(model_path: str) -> str | None:
    """``cameras.json`` sits next to the ``checkpoints/`` of a training output dir."""
    path = Path(model_path)
    if path.is_file():
        path = path.parent
    candidate = path / "cameras.json"
    return str(candidate) if candidate.is_file() else None


def cli():
    # define arguments
    parser = argparse.ArgumentParser(description="Interactive Web viewer for a trained model")
    parser.add_argument("model_paths", type=str, nargs="+",
                        help="Training output directory or .ckpt file")
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--cameras_json", "--cameras-json", type=str, default=None,
                        help="Path to cameras.json; auto-detected next to the model directory when omitted")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true", default=False,
                        help="Draw the training cameras as frustums")
    parser.add_argument("--time_max", "--time-max", type=float, default=1.0,
                        help="Upper bound of the time slider (camera time is normalised to [0, 1])")
    parser.add_argument("--up", nargs=3, required=False, type=float, default=None,
                        help="Up direction of the scene, e.g. `--up 0 0 1`")
    parser.add_argument("--default_camera_position", "--dcp", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_look_at", "--dcla", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--output_dir", "--output-dir", type=str, default=None,
                        help="Directory the 'Save frame' button writes to (default: ./outputs)")
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    args = parser.parse_args()

    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model(s) and the renderer recorded in the checkpoint.
    models = []
    renderer = None
    for model_path in args.model_paths:
        model, renderer = GaussianModelLoader.search_and_load(model_path, device)
        model.freeze()
        models.append(model)

    if len(models) == 1:
        gaussian_model = models[0]
    else:
        from internal.utils.gaussian_utils.gaussian_model_editor import MultipleGaussianModelEditor
        gaussian_model = MultipleGaussianModelEditor(models, device)

    cameras_json = args.cameras_json or _resolve_cameras_json(args.model_paths[0])
    if cameras_json is None:
        print("[viewer] no cameras.json found: camera frustums are unavailable")

    viewer = NewViewer(
        gaussian_model=gaussian_model,
        renderer=renderer,
        config=NewViewerConfig(
            host=args.host,
            port=args.port,
            background_color=_parse_background_color(args.background_color),
            image_format=args.image_format,
            up_direction=tuple(args.up) if args.up is not None else (0.0, 0.0, 1.0),
            default_camera_position=(
                tuple(args.default_camera_position) if args.default_camera_position is not None else None
            ),
            default_camera_look_at=(
                tuple(args.default_camera_look_at) if args.default_camera_look_at is not None else None
            ),
            cameras_json=cameras_json,
            show_cameras=args.show_cameras,
            time_max=args.time_max,
            output_dir=args.output_dir,
        ),
    )

    # start viewer server
    viewer.start()
