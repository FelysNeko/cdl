import torch
import triton

from cdl.core.kernel.config import DEFAULT_CONFIG, KernelConfig
from cdl.core.kernel.ecdl.kernel import (
    _ecdl_absmax,
    _ecdl_absmax_2d,
    _ecdl_bwd,
    _ecdl_bwd_2d,
    _ecdl_fwd,
    _ecdl_fwd_2d,
)
from cdl.core.kernel.formats.spec import FormatSpec
from cdl.core.kernel.layout import flatten_theta, flatten_theta_2d

ABSMAX_CONFIG = KernelConfig(block_chunk=32, num_warps=1)


def _scale_mode_id(scale_mode: str) -> int:
    if scale_mode == "nvidia":
        return 0
    if scale_mode == "ocp":
        return 1
    return 2


def _launch_ecdl_fwd_2d(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    scale_mode: str,
    config: KernelConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, lead, m, nb_r, nb_c, tpb = flatten_theta_2d(theta)
    h = nb_r * tpb
    k = nb_c * tpb
    ntiles = m * nb_r * nb_c
    mode = _scale_mode_id(scale_mode)

    if mode == 0:
        bamax = torch.empty((m, nb_r, nb_c), dtype=torch.float32, device=theta.device)
        gmax_u32 = torch.full((), 0x00800000, dtype=torch.uint32, device=theta.device)
        if ntiles > 0:
            _ecdl_absmax_2d[(ntiles,)](
                flat,
                bamax,
                gmax_u32,
                h,
                k,
                nb_r,
                nb_c,
                TPB=tpb,
                num_warps=ABSMAX_CONFIG.num_warps,
            )
        global_absmax = gmax_u32.view(torch.float32)
        qmax = fmt.q_max
        alpha_eff = global_absmax / fmt.f_max
    else:
        bamax = flat
        global_absmax = flat
        qmax = 0.0
        alpha_eff = theta.new_ones((), dtype=torch.float32)

    is_fp4 = fmt.tn == 16
    packed_cols = tpb // 2 if is_fp4 else tpb
    packed_theta = torch.empty(
        (m, h, nb_c * packed_cols), dtype=torch.uint8, device=theta.device
    )
    packed_scale = torch.empty((m, nb_r, nb_c), dtype=torch.uint8, device=theta.device)
    if ntiles > 0:
        _ecdl_fwd_2d[(ntiles,)](
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
            h,
            k,
            nb_r,
            nb_c,
            fmt.q_ordinal,
            fmt.theta_ordinal,
            TOPK=theta_topk,
            TPB=tpb,
            TN=fmt.tn,
            QN=fmt.qn,
            MODE=mode,
            EMAX=fmt.theta_emax,
            SCALE_BIAS=fmt.scale_bias,
            num_warps=config.num_warps,
        )

    packed_theta = packed_theta.reshape(*lead, nb_r, tpb, nb_c, packed_cols)
    if not is_fp4:
        packed_theta = packed_theta.view(torch.float8_e4m3fn)
    packed_scale = packed_scale.reshape(*lead, nb_r, 1, nb_c, 1)
    return packed_theta, packed_scale, alpha_eff


def launch_ecdl_fwd(
    theta: torch.Tensor,
    theta_kappa: torch.Tensor,
    seed: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    block_2d: bool,
    scale_mode: str = "nvidia",
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_2d:
        return _launch_ecdl_fwd_2d(
            theta, theta_kappa, seed, fmt, theta_topk, scale_mode, config
        )

    flat, lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    if config == DEFAULT_CONFIG:
        bm = 16 if tpb == 16 else 8
        bb = 1 if nb <= 1 else (2 if nb == 2 else 4)
        nw = 4
    else:
        bm, bb, nw = 1, config.block_chunk, config.num_warps
    ngb = triton.cdiv(nb, bb)
    nm = triton.cdiv(m, bm)
    mode = _scale_mode_id(scale_mode)

    if mode == 0:
        bamax = torch.empty((m, nb), dtype=torch.float32, device=theta.device)
        gmax_u32 = torch.full((), 0x00800000, dtype=torch.uint32, device=theta.device)
        if m > 0 and nb > 0:
            _ecdl_absmax[(nm, ngb)](
                flat,
                bamax,
                gmax_u32,
                m,
                k,
                nb,
                TPB=tpb,
                BM=bm,
                BB=bb,
                num_warps=nw,
            )
        global_absmax = gmax_u32.view(torch.float32)
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
    if m > 0 and nb > 0:
        _ecdl_fwd[(nm, ngb)](
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
            m,
            k,
            nb,
            fmt.q_ordinal,
            fmt.theta_ordinal,
            TOPK=theta_topk,
            TPB=tpb,
            TN=fmt.tn,
            QN=fmt.qn,
            BM=bm,
            BB=bb,
            MODE=mode,
            EMAX=fmt.theta_emax,
            SCALE_BIAS=fmt.scale_bias,
            num_warps=nw,
        )

    packed_theta = packed_theta.reshape(*lead, nb, packed_cols)
    if not is_fp4:
        packed_theta = packed_theta.view(torch.float8_e4m3fn)
    packed_scale = packed_scale.reshape(*lead, nb, 1)
    return packed_theta, packed_scale, alpha_eff


def _launch_ecdl_bwd_2d(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    packed_scale: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    need_dkappa: bool,
    config: KernelConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, _lead, m, nb_r, nb_c, tpb = flatten_theta_2d(theta)
    h = nb_r * tpb
    k = nb_c * tpb
    g = grad_proxy.reshape(m, h, k).contiguous()
    ps_flat = packed_scale.reshape(m, nb_r, nb_c).contiguous()

    dtheta = theta.new_empty((m, h, k), dtype=theta.dtype)
    ntiles = m * nb_r * nb_c
    partials = (
        torch.empty((ntiles, 2), dtype=torch.float32, device=theta.device)
        if need_dkappa
        else alpha
    )
    if ntiles > 0:
        _ecdl_bwd_2d[(ntiles,)](
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
            h,
            k,
            nb_r,
            nb_c,
            fmt.theta_ordinal,
            TOPK=theta_topk,
            TPB=tpb,
            TN=fmt.tn,
            SCALE_BIAS=fmt.scale_bias,
            COMPUTE_SCALARS=need_dkappa,
            num_warps=config.num_warps,
        )
    scalars = (
        partials.sum(0)
        if need_dkappa
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    return dtheta, scalars


def launch_ecdl_bwd(
    grad_proxy: torch.Tensor,
    theta: torch.Tensor,
    packed_scale: torch.Tensor,
    kappa_backward: torch.Tensor,
    alpha: torch.Tensor,
    fmt: FormatSpec,
    theta_topk: int,
    block_2d: bool,
    scale_mode: str = "nvidia",
    need_dkappa: bool = True,
    config: KernelConfig = DEFAULT_CONFIG,
) -> tuple[torch.Tensor, torch.Tensor]:
    if block_2d:
        return _launch_ecdl_bwd_2d(
            grad_proxy,
            theta,
            packed_scale,
            kappa_backward,
            alpha,
            fmt,
            theta_topk,
            need_dkappa,
            config,
        )

    flat, _lead, m, nb, tpb = flatten_theta(theta)
    k = nb * tpb
    g = grad_proxy.reshape(m, k)
    ps_flat = packed_scale.reshape(m, nb)

    dtheta = theta.new_empty((m, k), dtype=theta.dtype)
    if config == DEFAULT_CONFIG:
        bm = 4
        bb = 1 if nb <= 1 else (2 if nb == 2 else 4)
        nw = 2
    else:
        bm, bb, nw = 1, config.block_chunk, config.num_warps
    ngb = triton.cdiv(nb, bb)
    nm = triton.cdiv(m, bm)
    partials = (
        torch.empty((nm * ngb, 2), dtype=torch.float32, device=theta.device)
        if need_dkappa
        else alpha
    )
    if m > 0 and nb > 0:
        _ecdl_bwd[(nm, ngb)](
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
            m,
            k,
            nb,
            k,
            fmt.theta_ordinal,
            TOPK=theta_topk,
            TPB=tpb,
            TN=fmt.tn,
            BM=bm,
            BB=bb,
            SCALE_BIAS=fmt.scale_bias,
            COMPUTE_SCALARS=need_dkappa,
            num_warps=nw,
        )
    scalars = (
        partials.sum(0)
        if need_dkappa
        else torch.empty(0, dtype=torch.float32, device=theta.device)
    )
    return dtheta, scalars
