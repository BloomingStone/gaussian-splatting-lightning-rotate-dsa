from typing import Tuple, Dict, Any
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from lightning import LightningModule
from torch import Tensor

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
        batch: Any, 
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

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Tensor|float], Dict[str, bool]]:
        raise NotImplementedError

    def on_parameter_move(self, *args, **kwargs):
        raise NotImplementedError


class MetricImpl(MetricModule):
    pass


class Metric(Instantiable):
    def instantiate(self, *args, **kwargs) -> MetricModule:
        raise NotImplementedError
