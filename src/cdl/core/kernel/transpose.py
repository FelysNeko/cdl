import torch
import triton
import triton.language as tl

_HADAMARD16_CACHE: dict[torch.device, torch.Tensor] = {}


def _hadamard16(device: torch.device) -> torch.Tensor:
    h = _HADAMARD16_CACHE.get(device)
    if h is None:
        idx = torch.arange(16, device=device, dtype=torch.int32)
        bits = idx[:, None] & idx[None, :]
        pop = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1) + ((bits >> 3) & 1)
        h = torch.where(pop % 2 == 0, 0.25, -0.25).to(torch.float32)
        _HADAMARD16_CACHE[device] = h
    return h


@triton.jit
def _fp4_repack_transpose_kernel(
    in_ptr,
    out_ptr,
    N,
    K_HALF,
    N_HALF,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    rn = pid_n * BN + tl.arange(0, BN)
    rk = pid_k * BK + tl.arange(0, BK)
    in_mask = (rn[:, None] < N) & (rk[None, :] < K_HALF)
    raw = tl.load(
        in_ptr + rn[:, None].to(tl.int64) * K_HALF + rk[None, :],
        mask=in_mask,
        other=0,
    )

    lo = raw & 0xF
    hi = (raw >> 4) & 0xF
    nib_k = tl.reshape(tl.join(lo, hi), (BN, 2 * BK))
    nib_t = tl.trans(nib_k)
    p0, p1 = tl.split(tl.reshape(nib_t, (2 * BK, BN // 2, 2)))
    out = (p0 | (p1 << 4)).to(tl.uint8)

    ok = pid_k * (2 * BK) + tl.arange(0, 2 * BK)
    on = pid_n * (BN // 2) + tl.arange(0, BN // 2)
    out_mask = (ok[:, None] < K_HALF * 2) & (on[None, :] < N_HALF)
    tl.store(
        out_ptr + ok[:, None].to(tl.int64) * N_HALF + on[None, :],
        out,
        mask=out_mask,
    )


@triton.jit
def _rht16_transpose_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    M,
    C,
    M_PAD,
    K_TILES: tl.constexpr,
    BC: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    ar16 = tl.arange(0, 16)
    q = tl.load(q_ptr + ar16[:, None] * 16 + ar16[None, :])
    rc = pid_c * BC + tl.arange(0, BC)
    cmask = rc < C

    for t in tl.static_range(K_TILES):
        rm = (pid_m * K_TILES + t) * 16 + ar16
        raw = tl.load(
            x_ptr + rm[:, None].to(tl.int64) * C + rc[None, :],
            mask=(rm[:, None] < M) & cmask[None, :],
            other=0.0,
        )
        if IS_FP32:
            y = tl.dot(q, raw, input_precision="ieee")
        else:
            y = tl.dot(q, raw)

        y_t = tl.trans(y).to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + rc[:, None].to(tl.int64) * M_PAD + rm[None, :],
            y_t,
            mask=cmask[:, None] & (rm[None, :] < M_PAD),
        )


def fp4_repack_transpose(
    packed: torch.Tensor,
    block_n: int = 32,
    block_k: int = 32,
) -> torch.Tensor:
    """Repack FP4 tensor from ``(N, K//2)`` (K-packed) to ``(K, N//2)`` (N-packed)."""
    assert packed.ndim == 2 and packed.dtype == torch.uint8
    packed = packed.contiguous()
    n, k_half = packed.shape
    assert n % 2 == 0
    k = k_half * 2
    n_half = n // 2
    out = packed.new_empty((k, n_half))
    if n == 0 or k_half == 0:
        return out
    grid = (triton.cdiv(n, block_n), triton.cdiv(k_half, block_k))
    _fp4_repack_transpose_kernel[grid](
        packed,
        out,
        n,
        k_half,
        n_half,
        BN=block_n,
        BK=block_k,
        num_warps=4,
    )
    return out


def rht16_transpose(
    x: torch.Tensor,
    signs: torch.Tensor,
    m_multiple: int = 32,
    block_c: int = 64,
) -> torch.Tensor:
    """Apply 16x16 Random Hadamard Transform along M and transpose ``(M, C) -> (C, M_pad)``."""
    assert x.ndim == 2 and signs.shape == (16,)
    assert m_multiple % 16 == 0
    x = x.contiguous()
    m, c = x.shape
    m_pad = triton.cdiv(m, m_multiple) * m_multiple
    out = x.new_empty((c, m_pad))
    if m == 0 or c == 0:
        return out.zero_()
    q = (
        (
            _hadamard16(x.device)
            * signs.to(device=x.device, dtype=torch.float32)[None, :]
        )
        .to(x.dtype)
        .contiguous()
    )
    k_tiles = 2 if m_pad <= 32 else 4
    grid = (triton.cdiv(m_pad, k_tiles * 16), triton.cdiv(c, block_c))
    _rht16_transpose_kernel[grid](
        x,
        q,
        out,
        m,
        c,
        m_pad,
        K_TILES=k_tiles,
        BC=block_c,
        IS_FP32=(x.dtype == torch.float32),
        num_warps=4,
    )
    return out
