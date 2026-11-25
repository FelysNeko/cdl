import torch
import triton
import triton.language as tl


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


def _rht16_configs() -> list[triton.Config]:
    return [
        triton.Config({"K_TILES": 1, "BC": 32}, num_warps=2),
        triton.Config({"K_TILES": 1, "BC": 64}, num_warps=4),
        triton.Config({"K_TILES": 2, "BC": 32}, num_warps=4),
        triton.Config({"K_TILES": 2, "BC": 64}, num_warps=4, num_stages=2),
        triton.Config({"K_TILES": 2, "BC": 64}, num_warps=4, num_stages=3),
        triton.Config({"K_TILES": 4, "BC": 32}, num_warps=4),
        triton.Config({"K_TILES": 4, "BC": 64}, num_warps=4, num_stages=2),
        triton.Config({"K_TILES": 4, "BC": 64}, num_warps=4, num_stages=3),
        triton.Config({"K_TILES": 4, "BC": 64}, num_warps=8),
        triton.Config({"K_TILES": 4, "BC": 128}, num_warps=4),
        triton.Config({"K_TILES": 4, "BC": 128}, num_warps=8, num_stages=2),
        triton.Config({"K_TILES": 8, "BC": 64}, num_warps=4),
        triton.Config({"K_TILES": 8, "BC": 64}, num_warps=8),
        triton.Config({"K_TILES": 8, "BC": 128}, num_warps=8, num_stages=2),
    ]


@triton.jit
def _rht16_transpose_kernel_impl(
    x_ptr,
    seed_ptr,
    out_ptr,
    M,
    C,
    M_PAD,
    M_BUCKET: tl.constexpr,
    C_BUCKET: tl.constexpr,
    K_TILES: tl.constexpr,
    BC: tl.constexpr,
    IS_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    seed = tl.load(seed_ptr)
    ar16 = tl.arange(0, 16)
    bits = ar16[:, None] & ar16[None, :]
    pop = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1) + ((bits >> 3) & 1)
    h16 = tl.where((pop & 1) == 0, 0.25, -0.25)
    s16 = tl.where(((seed >> ar16) & 1) == 1, 1.0, -1.0)
    q = (h16 * s16[None, :]).to(x_ptr.dtype.element_ty)

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


_rht16_transpose_kernel = triton.autotune(
    configs=_rht16_configs(),
    key=["M_BUCKET", "C_BUCKET", "IS_FP32"],
    cache_results=True,
)(_rht16_transpose_kernel_impl)


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
    seed: torch.Tensor,
    m_multiple: int = 32,
    block_c: int | None = None,
) -> torch.Tensor:
    """Apply 16x16 Random Hadamard Transform along M using the low 16 bits of ``seed`` and transpose ``(M, C) -> (C, M_pad)``."""
    assert x.ndim == 2 and seed.shape == () and seed.dtype == torch.int32
    assert seed.device == x.device
    assert m_multiple % 16 == 0
    x = x.contiguous()
    m, c = x.shape
    m_pad = triton.cdiv(m, m_multiple) * m_multiple
    out = x.new_empty((c, m_pad))
    if m == 0 or c == 0:
        return out.zero_()
    m_bucket = (
        0 if m_pad <= 32 else (1 if m_pad <= 512 else (2 if m_pad <= 2048 else 3))
    )
    c_bucket = 0 if c <= 256 else (1 if c <= 1024 else 2)
    if block_c is not None:
        k_tiles = 2 if m_pad <= 32 else 4
        grid = (triton.cdiv(m_pad, k_tiles * 16), triton.cdiv(c, block_c))
        _rht16_transpose_kernel_impl[grid](
            x,
            seed,
            out,
            m,
            c,
            m_pad,
            M_BUCKET=m_bucket,
            C_BUCKET=c_bucket,
            K_TILES=k_tiles,
            BC=block_c,
            IS_FP32=(x.dtype == torch.float32),
            num_warps=4,
        )
    else:
        _rht16_transpose_kernel[
            lambda meta: (
                triton.cdiv(m_pad, meta["K_TILES"] * 16),
                triton.cdiv(c, meta["BC"]),
            )
        ](
            x,
            seed,
            out,
            m,
            c,
            m_pad,
            M_BUCKET=m_bucket,
            C_BUCKET=c_bucket,
            IS_FP32=(x.dtype == torch.float32),
        )
    return out
