import abc
from collections.abc import Callable

import torch
from torch import nn


class QTensor(abc.ABC):
    def nvfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def mxfp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def mxfp8(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    @abc.abstractmethod
    def dequantize(self) -> torch.Tensor: ...

    def proxy(self) -> torch.Tensor | None:
        raise NotImplementedError()


class Quant(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> QTensor: ...


type MatMul = Callable[[QTensor, QTensor], torch.Tensor]
