from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT
from torch.optim.adam import Adam

from .instantiate_config import Instantiable
 

@dataclass
class OptimizerConfig(Instantiable):
    def instantiate(self, params: ParamsT, lr: float|Tensor, *args, **kwargs) -> Optimizer:
        raise NotImplementedError() 
    


@dataclass
class AdamConfig(OptimizerConfig):
    def instantiate(self, params: ParamsT, lr: float, *args, **kwargs) -> Optimizer:
        return Adam(params, lr, *args, **kwargs)
