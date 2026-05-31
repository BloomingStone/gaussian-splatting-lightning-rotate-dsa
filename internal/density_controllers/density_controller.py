import torch
from torch.optim.optimizer import Optimizer

from lightning import LightningModule

from ..instantiate_config import Instantiable
from ..renderers.renderer import RendererOutputs
from ..dataparsers.dataparser import BatchT
from ..models.gaussian import GaussianModel


class DensityControllerImpl(torch.nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    def before_backward(
        self, 
        outputs: RendererOutputs, 
        batch: BatchT, 
        gaussian_model: GaussianModel, 
        optimizers: list[Optimizer], 
        global_step: int, 
        pl_module: LightningModule
    ) -> None:
        pass

    def after_backward(
        self, 
        outputs: RendererOutputs, 
        batch: BatchT, 
        gaussian_model: GaussianModel, 
        optimizers: list[Optimizer], 
        global_step: int, 
        pl_module: LightningModule
    ) -> None:
        pass

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        pass

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def after_density_changed(self, gaussian_model: GaussianModel, optimizers: list[Optimizer], pl_module: LightningModule) -> None:
        """
        This interface will be invoked when the density is changed elsewhere
        """
        pass


class DensityController(Instantiable):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        raise NotImplementedError
