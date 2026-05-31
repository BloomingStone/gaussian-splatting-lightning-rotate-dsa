from typing import Tuple, Dict, Any, Callable
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from torch import Tensor

from ..dataparsers.dataparser import BatchT
from ..utils.ssim import ssim
from ..models.gaussian import GaussianModel
from ..renderers.renderer import RendererOutputs
from ..instantiate_config import Instantiable


class MetricModule(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str, pl_module):
        pass

    def get_train_metrics(
        self, 
        pl_module: LightningModule, 
        gaussian_model: GaussianModel, 
        step: int, 
        batch: BatchT, 
        outputs: RendererOutputs
    ) -> Tuple[Dict[str, Tensor|float], Dict[str, bool]]:
        """
        :return:
            The first dict: contains the metric values.
                The `backward()` only will be invoked for the one with key `loss`.
                All other values are only for logging.
            The second dict: indicates whether the metric value should be shown on progress bar
        """

        return self.get_validate_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )

    def training_setup(self, pl_module) -> tuple[list[Optimizer]|None, list[LRScheduler]|None]:
        return [], []

    def get_validate_metrics(self, pl_module, gaussian_model, batch: BatchT, outputs) -> Tuple[Dict[str, Tensor|float], Dict[str, bool]]:
        raise NotImplementedError

    def on_parameter_move(self, *args, **kwargs):
        raise NotImplementedError


class MetricImpl(MetricModule):
    pass


class CommonImageMetricImpl(MetricImpl):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.no_state_dict_models: Dict[str, torch.nn.Module] = {}

    @staticmethod
    def _create_fused_ssim_adapter() -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        from fused_ssim import fused_ssim

        def adapter(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
            return fused_ssim(pred, gt)

        return adapter

    def setup(self, stage: str, pl_module):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.no_state_dict_models["lpips"] = LearnedPerceptualImagePatchSimilarity(
            normalize=True,
            net_type=self.config.lpips_net_type,
        )

        self.rgb_diff_loss_fn = self._l1_loss
        if self.config.rgb_diff_loss == "l2":
            print("Use L2 loss")
            self.rgb_diff_loss_fn = self._l2_loss

        self.ssim = ssim
        if self.config.fused_ssim:
            print("Fused SSIM enabled")
            self.ssim = self._create_fused_ssim_adapter()

    @staticmethod
    def _ensure_gray_nchw(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError(f"Unsupported image dim: {image.dim()}")

        if image.shape[1] != 1:
            image = image[:, :1]

        return image

    def add_image_validation_metrics(
        self,
        metrics: Dict[str, Any],
        prog_bar: Dict[str, bool],
        pred_gray: torch.Tensor,
        gt_gray: torch.Tensor,
    ) -> None:
        pred_gray = self._ensure_gray_nchw(pred_gray)
        gt_gray = self._ensure_gray_nchw(gt_gray)

        metrics["psnr"] = self.psnr(pred_gray, gt_gray)
        prog_bar["psnr"] = True

        pred_rgb = pred_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        gt_rgb = gt_gray.clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        metrics["lpips"] = self.no_state_dict_models["lpips"](pred_rgb, gt_rgb)
        prog_bar["lpips"] = True

    def on_parameter_move(self, *args, **kwargs):
        if "lpips" in self.no_state_dict_models:
            self.no_state_dict_models["lpips"] = self.no_state_dict_models["lpips"].to(*args, **kwargs)

    @staticmethod
    def _l1_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    @staticmethod
    def _l2_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)


class Metric(Instantiable):
    def instantiate(self, *args, **kwargs) -> MetricModule:
        raise NotImplementedError
