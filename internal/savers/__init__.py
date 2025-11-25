from dataclasses import dataclass
from abc import ABC, abstractmethod

from lightning import LightningModule

from internal.configs.instantiate_config import InstantiatableConfig


@dataclass
class Saver(InstantiatableConfig, ABC):
    @abstractmethod
    def instantiate(self, *args, **kwargs) -> "SaverModule":
        pass

class SaverModule(ABC):
    @abstractmethod
    def save(self, pl_module: LightningModule):
        pass