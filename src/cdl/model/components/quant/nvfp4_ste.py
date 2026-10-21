import flashinfer
import torch
import torch.nn.functional as F
from torch import nn


@torch.library.custom_op("cdl::nvfp4_qdq", mutates_args=())
def nvfp4_qdq(x, global_scale: torch.Tensor, global_scale_inv: torch.Tensor):
    packed, sf = flashinfer.fp4_quantize(x, global_scale, is_sf_swizzled_layout=False)
    out = flashinfer.nvfp4_kv_dequantize(
        packed, sf, global_scale_inv, output_dtype=x.dtype
    )
    return out


@nvfp4_qdq.register_fake
def _(x, global_scale: torch.Tensor, global_scale_inv: torch.Tensor):
    return torch.empty_like(x)


def quantize_dequantize(theta: torch.Tensor) -> torch.Tensor:
    k = theta.shape[-1]
    with torch.no_grad():
        x = theta.reshape(-1, k).bfloat16()
        r = -k % 16
        if r:
            x = F.pad(x, (0, r))

        amax = x.float().abs().amax()
        global_scale = torch.where(
            amax > 0,
            amax.new_full((), 448.0 * 6.0) / amax,
            amax.new_full((), torch.finfo(torch.float32).max),
        ).reshape(1)
        out = nvfp4_qdq(x, global_scale, (1.0 / global_scale).float())

    out = out.reshape(*theta.shape[:-1], -1)
    dequantized_theta = out[..., :k].to(theta.dtype)
    return theta + (dequantized_theta - theta).detach()


class Nvfp4SteQuantForActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        flat = input.permute(0, 2, 3, 1).reshape(-1, c)
        out = quantize_dequantize(flat)
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class Nvfp4SteQuantForWeight(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantize_dequantize(input.flatten(1)).view_as(input)
