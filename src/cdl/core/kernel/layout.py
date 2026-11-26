"""Host-side tensor layout helpers for the quantizer kernels."""

import math

import torch


def flatten_theta(
    theta: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], int, int, int]:
    lead = theta.shape[:-2]
    nb, tpb = theta.shape[-2:]
    m = math.prod(lead)
    return theta.reshape(m, nb, tpb), lead, m, nb, tpb


def flatten_theta_2d(
    theta: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...], int, int, int, int]:
    """Flatten ``(*lead, nb_r, tpb, nb_c, tpb)`` to contiguous ``(m, nb_r*tpb, nb_c*tpb)``."""
    lead = theta.shape[:-4]
    nb_r, tpb_r, nb_c, tpb_c = theta.shape[-4:]
    if tpb_r != tpb_c:
        raise ValueError(
            f"2D block scaling requires square tiles, got ({tpb_r}, {tpb_c})"
        )
    m = math.prod(lead)
    return (
        theta.reshape(m, nb_r * tpb_r, nb_c * tpb_c).contiguous(),
        lead,
        m,
        nb_r,
        nb_c,
        tpb_r,
    )


def flatten_q(
    lead: tuple[int, ...], nb: int, q: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Flatten a (possibly broadcast) q tensor to ``(m, nb)``.

    The second return value is the stride used by the kernel; ``0`` means a
    single q row is shared by every theta row.
    """

    m = math.prod(lead)
    if q.numel() == nb:
        return q.reshape(1, nb), 0
    if q.numel() == m * nb:
        return q.reshape(m, nb), nb
    return (q * q.new_ones(*lead, nb, 1)).reshape(m, nb), nb


def flatten_q_2d(
    lead: tuple[int, ...], nb_r: int, nb_c: int, q: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Flatten a (possibly broadcast) 2D-blocked q tensor to ``(m, nb_r, nb_c)``."""

    m = math.prod(lead)
    ntiles = nb_r * nb_c
    if q.numel() == ntiles:
        return q.reshape(1, nb_r, nb_c).contiguous(), 0
    if q.numel() == m * ntiles:
        return q.reshape(m, nb_r, nb_c).contiguous(), ntiles
    return (q * q.new_ones(*lead, nb_r, 1, nb_c, 1)).reshape(
        m, nb_r, nb_c
    ).contiguous(), ntiles


def q_indices_shape(lead: tuple[int, ...], nb: int, q: torch.Tensor) -> tuple[int, ...]:
    if q.numel() == nb:
        return q.shape
    return (*lead, nb, 1)


def q_indices_shape_2d(
    lead: tuple[int, ...], nb_r: int, nb_c: int, q: torch.Tensor
) -> tuple[int, ...]:
    if q.numel() == nb_r * nb_c:
        return q.shape
    return (*lead, nb_r, 1, nb_c, 1)


def reduce_q_grad(
    dq: torch.Tensor, theta: torch.Tensor, q: torch.Tensor, block_2d: bool
) -> torch.Tensor:
    """Reduce per-block q gradients back to the (possibly broadcast) q shape."""

    if block_2d:
        nb_r, _, nb_c, _ = theta.shape[-4:]
        full = dq.reshape(*theta.shape[:-4], nb_r, 1, nb_c, 1)
        lead_pad = q.ndim - full.ndim
        if lead_pad > 0:
            full = full.reshape((1,) * lead_pad + full.shape)
    else:
        full = dq.reshape(*theta.shape[:-2], -1)
        lead_pad = q.ndim - full.ndim - 1
        full = full.reshape((1,) * max(lead_pad, 0) + full.shape + (1,))
    dims = tuple(
        i for i, size in enumerate(q.shape) if size == 1 and full.shape[i] != 1
    )
    if dims:
        full = full.sum(dim=dims, keepdim=True)
    return full.reshape(q.shape)
