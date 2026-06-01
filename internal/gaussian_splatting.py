import os.path
from typing import Tuple, List, Dict, Union, Any, Callable, Optional, cast
from typing_extensions import Self

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import csv
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch.loggers
from jsonargparse.typing import lazy_instance

from .models.gaussian import Gaussian, GaussianModel
from .models.vanilla_gaussian import VanillaGaussian
from .renderers import Renderer, VanillaRenderer, RendererConfig
from .renderers.renderer import RendererOutputs
from .metrics.metric import Metric
from .metrics.vanilla_metrics import VanillaMetrics
from .density_controllers.density_controller import DensityController
from .density_controllers.vanilla_density_controller import VanillaDensityController

from .datamodule import DataModule
from .dataparsers.dataparser import BatchT
from .cameras import Camera, Cameras


class GaussianSplatting(LightningModule):
    def __init__(
            self,
            gaussian: Gaussian = lazy_instance(VanillaGaussian),
            background_color: Tuple[float, float, float] = (0., 0., 0.),
            output_path: str|None = None,
            save_val_metrics: bool = True,  # save metric csv during validation/test  # TODO add it to callbacks
            renderer: Union[Renderer, RendererConfig] = lazy_instance(VanillaRenderer),
            metric: Metric = lazy_instance(VanillaMetrics),
            density: DensityController = lazy_instance(VanillaDensityController),
            web_viewer: bool = False,
            initialize_from: str|None = None,
            drop_optimizer_states: bool = False,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        
        # self.output_path = output_path
        # self.save_val_metrics = save_val_metrics
        # self.with_web_viewer = web_viewer
        # self.initialize_from = initialize_from
        # self.drop_optimizer_states = drop_optimizer_states

        # setup models
        self.gaussian_model = gaussian.instantiate()

        # instantiate renderer
        if isinstance(renderer, RendererConfig):
            renderer = renderer.instantiate()
        self.renderer = renderer

        # instantiate density controller
        self.density_controller = density.instantiate()

        # metrics
        self.metric = metric.instantiate()

        # background color
        self.background_color = torch.tensor(background_color, dtype=torch.float32)

        if web_viewer:
            from .web_viewer.training_viewer import TrainingViewer
            self.web_viewer: TrainingViewer|None = None     # TODO : make it works when web_viewer is True
        else:
            self.web_viewer = None

        self.batch_size = 1
        self.restored_epoch = 0
        self.restored_global_step = 0

        self.val_metrics: List[Tuple[str, Dict]] = []


    def log_metrics(
            self,
            metrics: dict,
            prog_bar: dict,
            prefix: str,
            on_step: bool,
            on_epoch: bool,
            name_prefix: str = "",
    ):
        for name in metrics:
            self.log(
                f"{prefix}/{name_prefix}{name}",
                metrics[name],
                prog_bar=prog_bar[name] if name in prog_bar else False,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=self.batch_size,
            )


    def _initialize_gaussians_from_trained_model(self):
        # assert self.hparams["gaussian"].extra_feature_dims == 0

        from .utils.guassian_utils.gaussian_model_loader import GaussianModelLoader
        load_from = GaussianModelLoader.search_load_file(self.hparams["initialize_from"])

        if load_from.suffix == ".vtp":
            import pyvista as pv
            polydata = cast(pv.PolyData, pv.read(str(load_from)))
            self.gaussian_model.setup_from_polydata(polydata)
            self.gaussian_model.to(self.device)
        else:
            # load from ckpt
            gaussian_model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
                load_from,
                device=self.device,
                eval_mode=False,
                pre_activate=False,
            )
            # replace config
            self.hparams["gaussians"] = gaussian_model.config
            self.gaussian_model = cast(GaussianModel, gaussian_model)

        print(f"initialize from {load_from}")

    def setup(self, stage: str):
        if stage == "fit":
            if self.hparams["initialize_from"] is None:
                self.gaussian_model.setup_from_pcd(xyz=self.get_datamodule().point_cloud.xyz, rgb=self.get_datamodule().point_cloud.rgb / 255.)
            else:
                self._initialize_gaussians_from_trained_model()

        self.renderer.setup(stage=stage, lightning_module=self)
        self.metric.setup(stage=stage, pl_module=self)
        self.density_controller.setup(stage=stage, pl_module=self)

    def on_load_checkpoint(self, checkpoint) -> None:
        if self.hparams["drop_optimizer_states"]:
            checkpoint["optimizer_states"] = []

        # reinitialize parameters based on the gaussian number in the checkpoint
        self.gaussian_model.setup_from_number(checkpoint["state_dict"]["gaussian_model.gaussians.means"].shape[0])

        # get epoch and global_step, which used in the output path of the validation and test images
        self.restored_epoch = checkpoint["epoch"]
        self.restored_global_step = checkpoint["global_step"]

        # call for renderer
        self.renderer.on_load_checkpoint(self, checkpoint)
        # call density controller's hook
        self.density_controller.on_load_checkpoint(self, checkpoint)

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
    
    def transfer_batch_to_device(self, batch: BatchT, device: torch.device, dataloader_idx: int) -> Any:
        camera, image_info, extra_data = batch
        image_name, gt_image, masked_pixels = image_info
        
        camera = camera.to_device(device)

        if extra_data is not None:
            extra_data = super().transfer_batch_to_device(extra_data, device, dataloader_idx)
        if gt_image is not None:
            gt_image = gt_image.to(device)
        if masked_pixels is not None:
            masked_pixels = masked_pixels.to(device)

        return camera, (image_name, gt_image, masked_pixels), extra_data

    def forward(self, camera: Camera) -> RendererOutputs:
        if self.training is True:
            return self.renderer.training_forward(
                self.trainer.global_step,
                self,
                camera,
                self.gaussian_model,
                bg_color=self.background_color.to(camera.R.device),
            )
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self.background_color.to(camera.R.device),
        )

    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer=use_pl_optimizer)

        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):    #type: ignore
            if idx == 0:
                continue
            optimizer._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")   #type: ignore
            optimizer._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step") #type: ignore

        return optimizers

    def lr_schedulers(self) -> None|list[LRScheduler]:
        schedulers = super().lr_schedulers()

        if schedulers is None:
            return []

        if not isinstance(schedulers, list):
            return [schedulers]

        return schedulers

    def is_final_step(self, step: int|None = None):
        if step is None:
            step = self.trainer.global_step
        if self.trainer.max_steps > 0 and step >= self.trainer.max_steps:
            return True
        # TODO: make it works when max_epochs set
        return False

    def on_train_start(self) -> None:
        super().on_train_start()

        if self.hparams["web_viewer"] is True and self.trainer.global_rank == 0:
            from .web_viewer.training_viewer import TrainingViewer
            if self.get_datamodule().hparams["parser"].__class__.__name__.lower() in ["blender", "nsvf", "matrixcity"]:
                up = torch.tensor([0., 0., 1.])
            else:
                c2w = self.get_datamodule().dataparser_outputs.train_set.cameras.world_to_camera[:, :3, :3]
                up = c2w[:, :3, 1].mean(dim=0)
                up = -up / torch.linalg.norm(up)
            self.web_viewer = TrainingViewer(
                camera_names=self.get_datamodule().dataparser_outputs.train_set.image_names,
                cameras=self.get_datamodule().dataparser_outputs.train_set.cameras,
                up_direction=up.cpu().numpy(),
                camera_center=self.get_datamodule().dataparser_outputs.train_set.cameras.camera_center.mean(dim=0).cpu().numpy(),
            )
            self.web_viewer.start()

    def on_train_batch_start(self, batch: BatchT, batch_idx: int):
        if self.web_viewer is not None:
            raise NotImplementedError("The training viewer is currently not implement.")
            # self.web_viewer.training_step(
            #     self.gaussian_model,
            #     self.renderer,
            #     self.background_color,
            #     self.trainer.global_step,
            # )
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch: BatchT, batch_idx: int):
        camera = batch[0]

        global_step = self.trainer.global_step + 1  # must start from 1 to prevent densify at the beginning

        # get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        
        optimizers = cast(list[Optimizer], optimizers)

        # zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # call renderer hook
        self.renderer.before_training_step(global_step, self)

        # forward
        outputs: RendererOutputs = self(camera)

        # metrics
        metrics, prog_bar = self.metric.get_train_metrics(self, self.gaussian_model, global_step, batch, outputs)

        # log learning rate and gaussian count every 100 iterations (without plus one step)
        if self.trainer.global_step % 100 == 0 and self.logger is not None:
            metrics["train/gaussians_count"] = float(self.gaussian_model.n_gaussians)
            for opt_idx, opt in enumerate(optimizers):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]

        self.log_metrics(metrics, prog_bar, prefix="train", on_step=True, on_epoch=False)

        # invoke `before_backward` interface of density controller
        self.density_controller.before_backward(
            outputs=outputs,
            batch=batch,
            gaussian_model=self.gaussian_model,
            optimizers=self.gaussian_optimizers,
            global_step=global_step,
            pl_module=self,
        )
        # backward
        assert "loss" in metrics and isinstance(metrics["loss"], torch.Tensor), "the metric dict returned by `get_train_metrics` must contain the key `loss` with tensor value for backward"
        self.manual_backward(metrics["loss"])

        # optimize
        for optimizer in optimizers:
            optimizer.step()

        # schedule lr
        if schedulers is not None:
            schedulers = cast(list[LRScheduler], schedulers)
            for scheduler in schedulers:
                scheduler.step()
        
        # invoke `after_backward` interface of density controller
        self.density_controller.after_backward(
            outputs=outputs,
            batch=batch,
            gaussian_model=self.gaussian_model,
            optimizers=self.gaussian_optimizers,
            global_step=global_step,
            pl_module=self,
        )

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: BatchT, batch_idx: int) -> None:
        # the value of `trainer.global_step` here
        # is the same as the local variable `global_step` in training_step
        global_step = self.trainer.global_step

        self.gaussian_model.on_train_batch_end(global_step, self)

        self.renderer.after_training_step(self.trainer.global_step, self)

        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: BatchT, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        if self.web_viewer is not None:
            raise NotImplementedError("The training viewer is currently not implement.")
            # self.web_viewer.validation_step(
            #     self.gaussian_model,
            #     self.renderer,
            #     self.background_color,
            #     batch_idx,
            # )

    def validation_step(self, batch: BatchT, batch_idx: int, name: str = "val"):
        camera, image_info, _ = batch
        gt_image = image_info[1]
        assert gt_image is not None, "Ground truth image is required for validation step"

        # forward
        outputs: RendererOutputs = self(camera)
        metrics, prog_bar = self.metric.get_validate_metrics(self, self.gaussian_model, batch, outputs)
        self.log_metrics(metrics, prog_bar, prefix=name, on_step=False, on_epoch=True)
        self.val_metrics.append((image_info[0], metrics))

        # write validation image
        return outputs

    def on_validation_epoch_end(self, name="val") -> None:
        super().on_validation_epoch_end()

        # save metrics
        if self.hparams["save_val_metrics"] is True and self.global_rank == 0 and len(self.val_metrics) > 0:
            metrics_output_dir = os.path.join(self.hparams["output_path"], "metrics")
            os.makedirs(metrics_output_dir, exist_ok=True)
            step = max(self.trainer.global_step, self.restored_global_step)

            metric_list_key_by_name = {}  # [metric_name] = metric_value_list
            metric_fields = list(self.val_metrics[0][1].keys())
            for i in metric_fields:
                metric_list_key_by_name[i] = []

            with open(os.path.join(metrics_output_dir, f"{name}-step={step}.csv"), "w") as f:
                metrics_writer = csv.writer(f)
                metrics_writer.writerow(["name"] + list(metric_fields))

                for image_name, metrics in self.val_metrics:
                    metric_row = [image_name]
                    for metric_name in metric_fields:
                        metric_list_key_by_name[metric_name].append(metrics[metric_name])
                        metric_row.append("{:.8f}".format(metrics[metric_name].item()))
                    metrics_writer.writerow(metric_row)

                # calculate mean metrics
                metrics_writer.writerow([""] + ["" for _ in range(len(metric_fields))])
                mean_metrics = ["MEAN"]
                for i in metric_fields:
                    mean_metrics.append("{:.8f}".format(torch.stack(metric_list_key_by_name[i]).mean(dim=0).item()))
                metrics_writer.writerow(mean_metrics)

        self.val_metrics.clear()

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        return None

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.on_validation_epoch_end(name="test")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        # initialize lists that store optimizers and schedulers
        optimizers = []
        schedulers = []

        def add_optimizers_and_schedulers(new_optimizers, new_schedulers):
            nonlocal optimizers
            nonlocal schedulers

            if new_optimizers is not None:
                if isinstance(new_optimizers, list):
                    optimizers += new_optimizers
                else:
                    optimizers.append(new_optimizers)
            if new_schedulers is not None:
                if isinstance(new_schedulers, list):
                    schedulers += new_schedulers
                else:
                    schedulers.append(new_schedulers)

        # gaussian model optimizer and scheduler setup
        gaussian_optimizers, gaussian_schedulers = self.gaussian_model.training_setup(self)
        self.gaussian_optimizers = gaussian_optimizers
        add_optimizers_and_schedulers(gaussian_optimizers, gaussian_schedulers)

        # renderer optimizer and scheduler setup
        renderer_optimizer, renderer_scheduler = self.renderer.training_setup(self)
        add_optimizers_and_schedulers(renderer_optimizer, renderer_scheduler)

        # metric optimizer and scheduler setup
        metric_optimizer, metric_scheduler = self.metric.training_setup(self)
        add_optimizers_and_schedulers(metric_optimizer, metric_scheduler)

        return optimizers, schedulers

    def set_datamodule_device(self, device):
        # whether trainer exists
        try:
            self.trainer
        except RuntimeError:
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return

    def _on_device_updated(self):
        self.metric.on_parameter_move(device=self.device)
        self.set_datamodule_device(self.device)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)

        self._on_device_updated()

        return self

    def cuda(self, device: Optional[Union[torch.device, int]] = None) -> Self:
        super().cuda(device)

        self._on_device_updated()

        return self

    def cpu(self) -> Self:
        super().cpu()

        self._on_device_updated()

        return self
    
    def get_datamodule(self) -> DataModule:
        return self.trainer.datamodule  #type: ignore

