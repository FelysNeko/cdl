import torch
import triton

from cdl.core.kernel.config import DEFAULT_CONFIG, KernelConfig
from cdl.core.kernel.ecdl.kernel import _ecdl_absmax, _ecdl_bwd, _ecdl_fwd
from cdl.core.kernel.formats.spec import FormatSpec
from cdl.core.kernel.layout import flatten_theta

ABSMAX_CONFIG = KernelConfig(block_chunk=32, num_warps=1)


def _scale_mode_id(scale_mode: str) -> int:
    if scale_mode == "nvidia":
        return 0
    if scale_mode == "ocp":
        return 1
    return 2


def launch_ecdl_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    scale_mode: str = "nvidia",
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    ngb = triton.cdiv(nb, config.block_chunk)
    mode = _scale_mode_id(scale_mode)

    if mode == 0:
        bamax = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
        _ecdl_absmax[(m, triton.cdiv(nb, ABSMAX_CONFIG.block_chunk))](
            flat,
            bamax,
            k,
            nb,
            TPB=tpb,
            BB=ABSMAX_CONFIG.block_chunk,
            num_warps=ABSMAX_CONFIG.num_warps,
        )
        global_absmax = bamax.max().clamp_min(torch.finfo(torch.float32).tiny)
        qmax = fmt.q_max
        alpha_eff = global_absmax / fmt.f_max
    else:
        # OCP (single-pass exponent bits) or unknown fallback: bamax/gmax/qmax are dead.
        bamax = flat
        global_absmax = flat
        qmax = 0.0
        alpha_eff = theta.new_ones((), dtype=torch.float32)

    is_fp4 = fmt.tn == 16
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_theta = torch.empty(
        (m, nb * packed_cols), dtype=torch.uint8, device=theta.device
    )
    packed_scale = torch.empty((m, nb), dtype=torch.uint8, device=theta.device)
    _ecdl_fwd[(m, ngb)](
        flat,
        bamax,
        global_absmax,
        qmax,
        fmt.theta_level,
        fmt.theta_gap,
        fmt.q_level,
        theta_kappa,
        alpha_eff,
        seed,
        packed_theta,
        packed_scale,
        k,
        nb,
        fmt.q_ordinal,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        TPB=tpb,
        TN=fmt.tn,
        QN=fmt.qn,
        BB=config.block_chunk,
        MODE=mode,
        EMAX=fmt.theta_emax,
        num_warps=config.num_warps,
    )

    packed_theta = packed_theta.reshape(*lead, nb, packed_cols)
    if not is_fp4:
        packed_theta = packed_theta.view(torch.float8_e4m3fn)
    packed_scale = packed_scale.reshape(*lead, nb, 1)
    return packed_theta, packed_scale, alpha_eff


def launch_ecdl_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    packed_scale: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    scale_mode: str = "nvidia",
    need_dkappa: bool = True,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, _lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    ps_flat = packed_scale.reshape(m, nb)

    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    ngb = triton.cdiv(nb, config.block_chunk)
    partials = (
        torch.empty((m * ngb, 2), dtype=torch.float32, device=theta.device)
        if need_dkappa
        else alpha
    )
    _ecdl_bwd[(m, ngb)](
        g,
        flat,
        ps_flat,
        fmt.q_level,
        fmt.theta_level,
        fmt.theta_gap,
        kappa_backward,
        alpha,
        dtheta,
        partials,
        k,
        nb,
        k,
        fmt.theta_ordinal,
        TOPK=theta_topk,
        TPB=tpb,
        TN=fmt.tn,
        BB=config.block_chunk,
        MODE=_scale_mode_id(scale_mode),
        COMPUTE_SCALARS=need_dkappa,
        num_warps=config.num_warps,
    )
    scalars = (
        partials.sum(0)
        if need_dkappa
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    return dtheta, scalars
