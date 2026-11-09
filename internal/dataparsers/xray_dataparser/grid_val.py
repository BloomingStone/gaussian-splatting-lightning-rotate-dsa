"""Load grid multi-view images as the validation set for XRay DSA datasets.

The ``grid`` directory (e.g. ``data/example_grids/grid``) contains one
subdirectory per cardiac phase (``frame_XXX_phase_Y.YYY``).  Each of those
subdirectories holds several projection views (``dsa/*.png``) taken from
different C-arm angles *at the same time point*, together with a ``meta.json``
describing the camera geometry of every view.

We use those views as a held-out validation set so training reports metrics on
multi-view consistency instead of on held-out frames of the same rotational
sweep.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ...cameras import Cameras
from ..dataparser import DatasetBuilder, Stage
from .meta import XRayMeta
from .cameras_builder import euler_angles_to_matrix
from .datasets import ImagesDataset, ImagesDatasetConfig


def build_cameras_from_angles(
    geom,
    convention: str,
    orientation_type: str,
    alpha_deg: list[float],
    beta_deg: list[float],
    time_norm: list[float],
    phase: list[float],
    idx: list[int],
) -> Cameras:
    """Build a ``Cameras`` object for arbitrary (alpha, beta, time, phase) tuples.

    Mirrors the geometry of ``RotateXRayCamerasBuilder`` so validation cameras
    are exactly consistent with the training cameras.
    """
    n = len(alpha_deg)
    sod = float(geom.sod)

    if orientation_type == "AP":
        R_colmap_orient = euler_angles_to_matrix(
            torch.tensor((torch.pi / 2, torch.pi, 0.0)), "XZY"
        )
    elif orientation_type == "PA":
        R_colmap_orient = euler_angles_to_matrix(
            torch.tensor((-torch.pi / 2, 0.0, 0.0)), "XZY"
        )
    else:
        raise ValueError(f"Invalid orientation_type {orientation_type}.")

    M_colmap_orient = torch.eye(4)
    M_colmap_orient[:3, :3] = R_colmap_orient

    alpha = torch.tensor(alpha_deg, dtype=torch.float32) / 180.0 * np.pi
    beta = torch.tensor(beta_deg, dtype=torch.float32) / 180.0 * np.pi
    angles = torch.stack([alpha, beta, torch.zeros_like(alpha)], dim=-1)
    M_rotation = torch.eye(4)[None].repeat(n, 1, 1)
    M_rotation[:, :3, :3] = euler_angles_to_matrix(angles, convention)

    M_translation = torch.eye(4)
    if orientation_type == "AP":
        M_translation[:3, 3] = torch.tensor([0, sod, 0])
    elif orientation_type == "PA":
        M_translation[:3, 3] = torch.tensor([0, -sod, 0])
    else:
        raise ValueError(f"Invalid orientation_type {orientation_type}.")

    M_c2w = M_rotation @ M_translation @ M_colmap_orient
    M_w2c = torch.linalg.inv(M_c2w)
    R_w2c = M_w2c[:, :3, :3]
    T_w2c = M_w2c[:, :3, 3]

    return Cameras.build(
        idx=torch.tensor(idx, dtype=torch.long),
        R=R_w2c,
        T=T_w2c,
        cx=geom.x0 + geom.width / 2,
        cy=geom.y0 + geom.height / 2,
        fx=geom.sdd / geom.delx,
        fy=geom.sdd / geom.dely,
        height=geom.height,
        width=geom.width,
        time=torch.tensor(time_norm, dtype=torch.float32),
        phase=torch.tensor(phase, dtype=torch.float32),
        znear=0.01,
        zfar=1e5,
    )


@dataclass
class GridValLoader:
    """Collect grid multi-view cameras and image paths into a val dataset."""

    grid_dir_name: str = "grid"
    views_subdir: str = "dsa"
    image_suffix: str = "*.png"
    phase_stride: int = 20
    """Sample every ``phase_stride``-th phase subdirectory (grid has 180)."""

    max_images: int = 0
    """Cap on the total number of validation images (0 = no cap)."""

    def load(self, data_dir: Path, meta: XRayMeta) -> tuple[Cameras, list[Path], list[Path] | None]:
        grid_dir = data_dir / self.grid_dir_name
        if not grid_dir.is_dir():
            raise FileNotFoundError(f"grid dir not found: {grid_dir}")

        phase_dirs = sorted(d for d in grid_dir.iterdir() if d.is_dir())
        phase_dirs = phase_dirs[:: self.phase_stride]

        geom = meta.c_arm_geometry
        convention = meta.rotated_parameters.convention
        orientation = geom.orientation_type

        t_min = float(meta.time_array.min())
        t_max = float(meta.time_array.max())
        t_span = (t_max - t_min) if t_max > t_min else 1.0

        alpha_all: list[float] = []
        beta_all: list[float] = []
        time_all: list[float] = []
        phase_all: list[float] = []
        idx_all: list[int] = []
        image_paths: list[Path] = []
        label_paths: list[Path] = []

        has_label = all((pd / "label").is_dir() for pd in phase_dirs)

        for pd in phase_dirs:
            with open(pd / "meta.json", "r") as f:
                m = json.load(f)
            phase = float(m["phase"])
            source_frame = int(m["source_frame"])

            # The time point is the source frame's time; all views of this
            # phase dir share the same time & phase.
            time_s = float(meta.frames[source_frame].time_s)
            time_norm = (time_s - t_min) / t_span

            views = sorted((pd / self.views_subdir).glob(self.image_suffix))
            n = min(len(m["frames"]), len(views))
            for i in range(n):
                frm = m["frames"][i]
                alpha_all.append(float(frm["alpha_degree"]))
                beta_all.append(float(frm["beta_degree"]))
                time_all.append(time_norm)
                phase_all.append(phase)
                idx_all.append(len(idx_all))
                image_paths.append(views[i])
                if has_label:
                    label_paths.append(pd / "label" / views[i].name)

                if self.max_images > 0 and len(image_paths) >= self.max_images:
                    break
            if self.max_images > 0 and len(image_paths) >= self.max_images:
                break

        cameras = build_cameras_from_angles(
            geom,
            convention,
            orientation,
            alpha_all,
            beta_all,
            time_all,
            phase_all,
            idx_all,
        )
        return cameras, image_paths, (label_paths if has_label else None)


@dataclass
class GridImagesDatasetBuilder(DatasetBuilder[XRayMeta, ImagesDataset]):
    """Build a validation dataset from the grid multi-view directory."""

    grid_val_loader: GridValLoader = field(default_factory=GridValLoader)
    dataset_config: ImagesDatasetConfig = field(default_factory=ImagesDatasetConfig)

    def build_dataset(
        self,
        data_dir: Path,
        cameras: Cameras,
        meta: XRayMeta,
        indices: list[int],
        stage: Stage,
    ) -> ImagesDataset:
        del cameras, indices, stage
        grid_cameras, image_paths, label_paths = self.grid_val_loader.load(data_dir, meta)

        return ImagesDataset(
            cameras=grid_cameras,
            cfg=self.dataset_config,
            image_paths=image_paths,
            masks_paths=label_paths,
            other_data_closure=None,
            source_indices=list(range(len(image_paths))),
            flow_paths=None,
        )
