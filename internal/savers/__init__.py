from dataclasses import dataclass
from abc import ABC, abstractmethod

from lightning import LightningModule

from internal.instantiate_config import Instantiable


@dataclass
class Saver(Instantiable, ABC):
    @abstractmethod
    def instantiate(self, *args, **kwargs) -> "SaverModule":
        pass

class SaverModule(ABC):
    @abstractmethod
    def save(self, pl_module: LightningModule):
        pass