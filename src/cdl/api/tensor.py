import abc

import torch


class QuantizedTensor(abc.ABC):
    @abc.abstractmethod
    def fp4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def fp8(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def dequantize(self) -> torch.Tensor: ...

    def proxy(self) -> torch.Tensor | None:
        return None
