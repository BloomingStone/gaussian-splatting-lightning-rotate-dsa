from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Optional, Mapping
import torch
import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import lightning
import pyvista as pv

from .gaussian import (
    Gaussian,
    GaussianModel,
    HasNewGetters,
    HasVanillaGetters,
)
from ..utils.general_utils import (
    inverse_sigmoid,
    strip_symmetric,
    build_scaling_rotation,
    knn,
)
from ..optimizers import OptimizerConfig, AdamConfig, SelectiveAdam, SparseGaussianAdam
from ..schedulers import Scheduler, ExponentialDecayScheduler

@dataclass
class VannilaExponentialDecayScheduler(ExponentialDecayScheduler):
    lr_final = 0.0000016
    max_steps = 30_000


@dataclass
class OptimizationConfig:
    means_lr_init: float = 0.00016
    # means_lr_scheduler: Scheduler = field(default_factory=lambda: ExponentialDecayScheduler(
    #     lr_final=0.0000016,
    #     max_steps=30_000,
    # ))
    # only in below format can work with jsonargparse
    means_lr_scheduler: VannilaExponentialDecayScheduler = field(default_factory= VannilaExponentialDecayScheduler)
    spatial_lr_scale: float = -1  # auto calculate from camera poses if <= 0

    shs_dc_lr: float = 0.0025

    shs_rest_lr: float = 0.0025 / 20.

    opacities_lr: float = 0.05

    scales_lr: float = 0.005

    rotations_lr: float = 0.001

    sh_degree_up_interval: int = 1_000

    optimizer: OptimizerConfig = field(default_factory=AdamConfig)


@dataclass
class VanillaGaussian(Gaussian):
    sh_degree: int = 3

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "VanillaGaussianModel":
        return VanillaGaussianModel(self)


class VanillaGaussianModel(
    GaussianModel,
    HasNewGetters,
    HasVanillaGetters,
):
    _active_sh_degree: torch.Tensor
    
    def __init__(self, config: VanillaGaussian) -> None:
        super().__init__()
        self.config = config

        names = [
                    "means",
                    "shs_dc",
                    "shs_rest",
                    "opacities",
                    "scales",
                    "rotations",
                ] + self.get_extra_property_names()
        self._names = tuple(names)

        self.is_pre_activated = False

        # TODO: is it suitable to place `active_sh_degree` in gaussian model?
        self.register_buffer("_active_sh_degree", torch.tensor(0, dtype=torch.uint8), persistent=True)

    def get_extra_property_names(self):
        return []

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Mapping[str, torch.Tensor|nn.Parameter], *args, **kwargs):
        pass


    def setup_from_pcd(self, xyz: Union[torch.Tensor, np.ndarray], rgb: Union[torch.Tensor, np.ndarray], init_scale: float=1.0, *args, **kwargs):
        from ..utils.math.sh_utils import RGB2SH

        if isinstance(xyz, np.ndarray):
            xyz = torch.tensor(xyz)
        if isinstance(rgb, np.ndarray):
            rgb = torch.tensor(rgb)

        fused_point_cloud = xyz.float()
        fused_color = RGB2SH(rgb.float())

        n_gaussians = fused_point_cloud.shape[0]

        # SHs
        shs = torch.zeros((n_gaussians, 3, (self.config.sh_degree + 1) ** 2)).float()
        shs[:, :3, 0] = fused_color
        shs[:, 3:, 1:] = 0.0
        
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(fused_point_cloud, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        # rotations
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        # opacities
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        means = nn.Parameter(fused_point_cloud.requires_grad_(True))
        shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        shs_rest = shs[:, :, 1:].transpose(1, 2).contiguous()
        shs_rest = nn.Parameter(shs_rest.requires_grad_(True))

        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rots.requires_grad_(True))
        opacities = nn.Parameter(opacities.requires_grad_(True))

        property_dict = {
            "means": means,
            "shs_dc": shs_dc,
            "shs_rest": shs_rest,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0

    def before_setup_set_properties_from_number(self, n: int, property_dict: Mapping[str, torch.Tensor], *args, **kwargs):
        pass

    def setup_from_number(self, n: int, *args, **kwargs):
        means = torch.zeros((n, 3))
        shs = torch.zeros((n, 3, (self.max_sh_degree + 1) ** 2))
        shs_dc = shs[:, :, 0:1].transpose(1, 2).contiguous()
        shs_rest = shs[:, :, 1:].transpose(1, 2).contiguous()
        scales = torch.zeros((n, 3))
        rotations = torch.zeros((n, 4))
        opacities = torch.zeros((n, 1))

        means = nn.Parameter(means.requires_grad_(True))
        shs_dc = nn.Parameter(shs_dc.requires_grad_(True))
        shs_rest = nn.Parameter(shs_rest.requires_grad_(True))
        scales = nn.Parameter(scales.requires_grad_(True))
        rotations = nn.Parameter(rotations.requires_grad_(True))
        opacities = nn.Parameter(opacities.requires_grad_(True))

        property_dict = {
            "means": means,
            "shs_dc": shs_dc,
            "shs_rest": shs_rest,
            "scales": scales,
            "rotations": rotations,
            "opacities": opacities,
        }
        self.before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        self.set_properties(property_dict)

        self.active_sh_degree = 0

    def setup_from_tensors(self, tensors: Dict[str, torch.Tensor], active_sh_degree: int = -1, *args, **kwargs):
        """
        Args:
            tensors
            active_sh_degree: -1 means use maximum sh_degree
            *args,
            **kwargs
        """

        # detect sh_degree
        if "shs_rest" in tensors:
            shs_rest_dims = tensors["shs_rest"].shape[1]
            sh_degree = -1
            for i in range(4):
                if shs_rest_dims == (i + 1) ** 2 - 1:
                    sh_degree = i
                    break
            assert sh_degree >= 0, f"can not get sh_degree from `shs_rest`"
        else:
            sh_degree = self.config.sh_degree

        # update sh_degree
        # self.config.sh_degree = sh_degree

        # TODO: may be should enable changing sh_degree
        # validate sh_degree
        assert self.config.sh_degree == sh_degree, "sh_degree not match"

        # initialize by number
        n_gaussians = tensors[list(tensors.keys())[0]].shape[0]
        self.setup_from_number(n_gaussians)

        unused_properties = list(tensors.keys())
        unmet_properties = list(self.property_names)

        # copy from tensor
        property_names = self.property_names

        with torch.no_grad():
            for i in tensors:
                if i not in property_names:
                    continue
                self.get_property(i).copy_(tensors[i])

                unused_properties.remove(i)
                unmet_properties.remove(i)

        if active_sh_degree == -1:
            active_sh_degree = sh_degree
        self.active_sh_degree = min(active_sh_degree, sh_degree)

        return unused_properties, unmet_properties

    def training_setup(self, module: "lightning.LightningModule") -> tuple[list[Optimizer], list[LRScheduler]]:
        spatial_lr_scale = self.config.optimization.spatial_lr_scale
        if spatial_lr_scale <= 0:
            spatial_lr_scale = module.trainer.datamodule.dataparser_outputs.camera_extent   # type: ignore
        assert spatial_lr_scale > 0

        optimization_config = self.config.optimization

        optimizer_factory = self.config.optimization.optimizer

        # the param name and property name must be identical

        # means
        means_lr_init = optimization_config.means_lr_init * spatial_lr_scale
        means_optimizer = optimizer_factory.instantiate(
            [{'params': [self.gaussians["means"]], "name": "means"}],
            lr=means_lr_init,
            eps=1e-15,
        )
        # TODO: other scheduler may not contain `lr_final`, but does not need to change scheduler currently
        optimization_config.means_lr_scheduler.lr_final *= spatial_lr_scale # type: ignore
        means_scheduler = optimization_config.means_lr_scheduler.instantiate().get_scheduler(
            means_optimizer,
            means_lr_init,
        )

        # the params with constant LR
        l = [
            {'params': [self.gaussians["shs_dc"]], 'lr': optimization_config.shs_dc_lr, "name": "shs_dc"},
            {'params': [self.gaussians["shs_rest"]], 'lr': optimization_config.shs_rest_lr, "name": "shs_rest"},
            {'params': [self.gaussians["opacities"]], 'lr': optimization_config.opacities_lr, "name": "opacities"},
            {'params': [self.gaussians["scales"]], 'lr': optimization_config.scales_lr, "name": "scales"},
            {'params': [self.gaussians["rotations"]], 'lr': optimization_config.rotations_lr, "name": "rotations"},
        ]
        constant_lr_optimizer = optimizer_factory.instantiate(l, lr=0.0, eps=1e-15)

        print("spatial_lr_scale={}, learning_rates=".format(spatial_lr_scale))
        print("  means={}->{}".format(means_lr_init, optimization_config.means_lr_scheduler.lr_final))
        for i in l:
            print("  {}={}".format(i["name"], i["lr"]))

        return [means_optimizer, constant_lr_optimizer], [means_scheduler]

    def get_property_names(self) -> Tuple[str, ...]:
        return self._names

    def on_train_batch_end(self, step: int, module: "lightning.LightningModule"):
        # Every `sh_degree_up_interval` iterations increase the levels of SH up to a maximum degree
        if step % self.config.optimization.sh_degree_up_interval != 0:
            return
        if self._active_sh_degree >= self.config.sh_degree:
            return
        self._active_sh_degree += 1

    # define properties by getters and setters

    def get_max_sh_degree(self) -> int:
        return self.config.sh_degree

    @property
    def max_sh_degree(self) -> int:
        return self.config.sh_degree

    def get_active_sh_degree(self) -> int:
        return self._active_sh_degree.item()    # type: ignore

    def set_active_sh_degree(self, v):
        self._active_sh_degree.fill_(v)

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree.item()    # type: ignore

    @active_sh_degree.setter
    def active_sh_degree(self, v: int):
        self._active_sh_degree.fill_(v)

    def opacity_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def opacity_inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return inverse_sigmoid(x)

    def scale_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def scale_inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)

    def rotation_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x)

    def rotation_inverse_activation(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @staticmethod
    def _return_as_is(x):
        return x

    def _get_shs_from_dict(self) -> torch.Tensor:
        return self.gaussians["shs"]

    def pre_activate_all_properties(self):
        self.is_pre_activated = True

        self.scales = self.get_scales()
        self.rotations = self.get_rotations()
        self.opacities = self.get_opacities()

        # concat `shs_dc` and `shs_rest` and store it to dict, then remove `shs_dc` and `shs_rest`
        names = list(self._names)
        ## concat
        self.gaussians["shs"] = self.get_shs()
        names.append("shs")
        ## remove `shs_dc`
        del self.gaussians["shs_dc"]
        names.remove("shs_dc")
        ## remove `shs_rest`
        del self.gaussians["shs_rest"]
        names.remove("shs_rest")
        ## replace `get_shs` interface
        self.get_shs = self._get_shs_from_dict
        ## replace `names`
        self._names = tuple(names)

        self.scale_activation = self._return_as_is
        self.scale_inverse_activation = self._return_as_is
        self.rotation_activation = self._return_as_is
        self.rotation_inverse_activation = self._return_as_is
        self.opacity_activation = self._return_as_is
        self.opacity_inverse_activation = self._return_as_is

    def get_non_pre_activated_properties(self):
        if self.is_pre_activated is True:
            activated_properties = self.properties
            keys = list(activated_properties.keys())
            non_pre_activated_properties = {}
            non_pre_activated_properties["scales"] = torch.log(activated_properties["scales"])
            keys.remove("scales")
            non_pre_activated_properties["opacities"] = inverse_sigmoid(activated_properties["opacities"])
            keys.remove("opacities")
            non_pre_activated_properties["shs_dc"] = activated_properties["shs"][:, :1, :]
            non_pre_activated_properties["shs_rest"] = activated_properties["shs"][:, 1:, :]
            keys.remove("shs")

            for key in keys:
                non_pre_activated_properties[key] = activated_properties[key]

            return non_pre_activated_properties
        else:
            return self.properties

    # below getters are declared for the compatibility purpose

    @property
    def get_scaling(self):
        return self.scale_activation(self.gaussians["scales"])

    @property
    def get_rotation(self):
        return self.rotation_activation(self.gaussians["rotations"])

    @property
    def get_xyz(self):
        return self.gaussians["means"]

    @property
    def get_features(self):
        return self.get_shs()

    @property
    def get_opacity(self):
        return self.opacity_activation(self.gaussians["opacities"])

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def get_covariance(self, scaling_modifier: float = 1.):
        return self.build_covariance_from_scaling_rotation(
            self.get_scales(),
            scaling_modifier,
            self.get_rotations(),
        )

    # --- VTK / PolyData 序列化 ---

    def to_polydata(self) -> pv.PolyData:
        props = self.get_non_pre_activated_properties()

        # means → points
        means_np = props["means"].detach().cpu().numpy().astype(np.float32)

        # build point_data dict from remaining properties
        point_data = {}
        for name in self.property_names:
            if name == "means":
                continue
            val = props[name].detach().cpu().numpy().astype(np.float32)
            # squeeze any trailing dim=1 so arrays like [N, 1] or [N, 1, 3] are clean
            # but don't squeeze the channel dim for shs
            point_data[name] = val

        pd = pv.PolyData(means_np)
        for name, arr in point_data.items():
            pd.point_data[name] = arr

        # store active_sh_degree as field data
        pd.field_data["active_sh_degree"] = np.array([self.active_sh_degree], dtype=np.int32)
        pd.field_data["model_type"] = np.array(["VanillaGaussian"])

        return pd

    def setup_from_polydata(self, polydata: pv.PolyData, *args, **kwargs):
        import numpy as np
        import torch as _torch

        means = _torch.tensor(np.asarray(polydata.points, dtype=np.float32), dtype=_torch.float)

        # detect sh_degree from shs_rest if available
        if "shs_rest" in polydata.point_data:
            shs_rest_dims = polydata.point_data["shs_rest"].shape[1]
            sh_degree = -1
            for i in range(4):
                if shs_rest_dims == (i + 1) ** 2 - 1:
                    sh_degree = i
                    break
            assert sh_degree >= 0, f"cannot determine sh_degree from shs_rest dim={shs_rest_dims}"
        else:
            sh_degree = self.config.sh_degree

        n_gaussians = means.shape[0]

        # build property dict with Parameters
        property_dict: dict = {}
        property_dict["means"] = nn.Parameter(means.requires_grad_(True))

        # required properties
        for name in self.property_names:
            if name == "means":
                continue
            if name in polydata.point_data:
                val = _torch.tensor(np.asarray(polydata.point_data[name], dtype=np.float32), dtype=_torch.float)
            else:
                # fill with zeros as fallback
                shape = list(means.shape)
                if name == "shs_dc":
                    shape = [n_gaussians, 1, 3]
                elif name == "shs_rest":
                    shape = [n_gaussians, (sh_degree + 1) ** 2 - 1, 3]
                elif name == "opacities":
                    shape = [n_gaussians, 1]
                elif name == "scales":
                    shape = [n_gaussians, 3]
                elif name == "rotations":
                    shape = [n_gaussians, 4]
                else:
                    # extra property: try to infer from property_names, use 1D by default
                    shape = [n_gaussians]
                val = _torch.zeros(shape, dtype=_torch.float)
            property_dict[name] = nn.Parameter(val.requires_grad_(True))

        self.set_properties(property_dict)

        # restore active_sh_degree
        if "active_sh_degree" in polydata.field_data:
            self.active_sh_degree = int(polydata.field_data["active_sh_degree"][0])
        else:
            self.active_sh_degree = sh_degree
