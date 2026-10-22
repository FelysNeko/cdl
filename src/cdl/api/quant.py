import abc

import torch
from torch import nn

from cdl.api.tensor import QuantizedTensor


class Quant(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedTensor: ...
