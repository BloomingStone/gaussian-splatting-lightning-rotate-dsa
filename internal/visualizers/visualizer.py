from __future__ import annotations

from typing import Protocol
from torch import Tensor

class Visualizer(Protocol):
    def process(self, image: Tensor) -> Tensor:
        raise NotImplementedError()


class IdenticalVisualizer(Visualizer):
    def process(self, image: Tensor) -> Tensor:
        return image