import torch

from cdl.api.quant import QTensor


def matmul(x: QTensor, w: QTensor) -> torch.Tensor:
    fp_x = x.dequantize()
    fp_x_proxy = x.proxy()
    if fp_x_proxy is not None:
        fp_x = fp_x_proxy + (fp_x - fp_x_proxy).detach()

    fp_w = w.dequantize()
    fp_w_proxy = w.proxy()
    if fp_w_proxy is not None:
        fp_w = fp_w_proxy + (fp_w - fp_w_proxy).detach()

    return fp_x @ fp_w.T
