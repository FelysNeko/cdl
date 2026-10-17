import functools

import flashinfer
import torch


def cdl_topk_cpmf(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = -kappa * ((theta / q).unsqueeze(-1) - a) ** 2
    topk_values, topk_indices = logits.topk(topk)
    topk_pmf = torch.softmax(topk_values, dim=-1)
    return topk_pmf, topk_indices


def cdl_topk_mpmf_entropy(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
    num_slots: int,
    num_quantized_params: int,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    groups = topk_pmf.numel() // (num_quantized_params * topk)
    idx = topk_indices.reshape(groups, num_quantized_params, topk)

    mpmf = topk_pmf.new_zeros(groups, num_slots)
    mpmf.scatter_add_(1, idx.reshape(groups, -1), topk_pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)
    return num_quantized_params * entropies.mean()


@torch.no_grad()
def cdl_topk_q_p(
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    topk = topk_pmf.shape[-1]
    cdf = topk_pmf.cumsum(-1)
    quantile = torch.rand(topk_pmf.shape[:-1], dtype=cdf.dtype, device=cdf.device)
    sub_indices = (cdf <= quantile.unsqueeze(-1)).sum(-1, keepdim=True)
    sub_indices.clamp_(max=topk - 1)
    chosen = topk_indices.gather(dim=-1, index=sub_indices).squeeze(-1)
    return chosen


def cdl_topk_q_d(
    a: torch.Tensor,
    q: torch.Tensor,
    topk_pmf: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    return (topk_pmf * a[topk_indices]).sum(-1) * q


def cdl_topk_forward(
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    kappa: torch.Tensor,
    topk: int,
    num_quantized_params: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_pmf, topk_indices = cdl_topk_cpmf(theta, a, q, kappa, topk)

    entropies = cdl_topk_mpmf_entropy(
        topk_pmf,
        topk_indices,
        a.numel(),
        num_quantized_params,
    )

    proxy = cdl_topk_q_d(a, q, topk_pmf, topk_indices)
    chosen = cdl_topk_q_p(topk_pmf, topk_indices)

    return chosen, proxy, entropies


def cascaded_cdl_topk_forward(
    theta: torch.Tensor,
    theta_lut: torch.Tensor,
    theta_kappa: torch.Tensor,
    theta_topk: int,
    theta_num_quantized_params: int,
    q: torch.Tensor,
    q_lut: torch.Tensor,
    q_kappa: torch.Tensor,
    q_topk: int,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_indices, q_proxy, q_entropies = cdl_topk_forward(
        q,
        q_lut,
        torch.ones(()),
        q_kappa,
        q_topk,
        q.numel(),
    )
    q_hat = q_lut[q_indices]
    proxied_q_hat = q_proxy + (q_hat - q_proxy).detach()
    theta_indices, theta_proxy, theta_entropies = cdl_topk_forward(
        theta,
        theta_lut,
        proxied_q_hat * alpha,
        theta_kappa,
        theta_topk,
        theta_num_quantized_params,
    )

    return theta_indices, q_indices, theta_proxy, q_entropies, theta_entropies


@functools.cache
def get_indices_to_e2m1_mapping(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.uint8,
        device=device,
    )


def pack_e2m1(x: torch.Tensor) -> torch.Tensor:
    mapping = get_indices_to_e2m1_mapping(x.device)
    codes = mapping[x].reshape(x.shape[0], -1)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def pack_e4m3(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(x.shape[0], -1)
        .float()
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


def swizzle_e4m3(sf: torch.Tensor) -> torch.Tensor:
    m, b = sf.shape
    rows, cols = (m + 127) // 128 * 128, (b + 3) // 4 * 4
    out = torch.zeros(rows, cols, dtype=torch.uint8, device=sf.device)
    out[:m, :b] = sf
    return (
        out.view(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )


class NVFP4:
    def __init__(
        self,
        theta: torch.Tensor,
        q: torch.Tensor,
        alpha: torch.Tensor,
    ) -> None:
        self.theta = theta
        self.q = q
        self.alpha = alpha

    @torch.no_grad()
    def pack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = pack_e2m1(self.theta)
        q = pack_e4m3(self.q)
        alpha = self.alpha.float().reshape(1)
        return theta, q, alpha


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: NVFP4,
        x_proxy: torch.Tensor,
        w: NVFP4,
        w_proxy: torch.Tensor,
    ) -> torch.Tensor:
        x_theta, x_q, x_alpha = x.pack()
        w_theta, w_q, w_alpha = w.pack()
        ctx.save_for_backward(x_theta, x_q, x_alpha, w_theta, w_q, w_alpha)

        out = flashinfer.mm_fp4(
            x_theta,
            w_theta.T,
            swizzle_e4m3(x_q),
            swizzle_e4m3(w_q).T,
            x_alpha * w_alpha,
            backend="b12x",
        )
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None, torch.Tensor]:
        saved_tensors: tuple[torch.Tensor, ...] = ctx.saved_tensors
        x_theta, x_q, x_alpha, w_theta, w_q, w_alpha = saved_tensors
        grad_output = grad_output.contiguous()

        w = flashinfer.nvfp4_kv_dequantize(w_theta, w_q, w_alpha)
        dx = grad_output @ w

        x = flashinfer.nvfp4_kv_dequantize(x_theta, x_q, x_alpha)
        dw = grad_output.T @ x

        return None, dx, None, dw
