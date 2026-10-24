"""Fused cascaded-CDL NVFP4 quantization + matmul kernel.

One Triton launch per tensor fuses the entire cascaded_cdl_topk_forward
companded path (q-level UE4M3 pass, theta-level E2M1 pass, top-k sampling,
NVFP4 packing, block scales) with an analytic backward kernel, and the
matmul runs on the flashinfer mm_fp4 kernel. Under torch.compile
reduce-overhead the fused path is 4-5x faster than the decomposed eager
path at mid sizes and OOM-free at sizes the eager path cannot run.

Public API:
- fused_cdl_matmul(theta_x, quant_x, theta_w, quant_w, seed=None): fused
  layer forward (training path).
- fusable_quant(quant): whether a quantizer module is supported.
- e2m1_tables / ue4m3_tables: format lookup tables.
- set_force_u: pin the sampling quantile (tests only).

The kernels assume companded=True. Entropy terms are not implemented (the
training path passes None). Env knobs: CDL_NUM_WARPS, CDL_BB (tuning),
CDL_FP4_BWD (backward GEMM precision: auto enables nvfp4 gradient GEMMs for
large problems).
"""

from cdl.core.kernel.matmul import FusedCdlMatmul, fusable_quant, fused_cdl_matmul
from cdl.core.kernel.quant import cdl_quant_bwd, cdl_quant_fwd, set_force_u
from cdl.core.kernel.tables import e2m1_tables, ue4m3_tables

__all__ = [
    "FusedCdlMatmul",
    "cdl_quant_bwd",
    "cdl_quant_fwd",
    "e2m1_tables",
    "fusable_quant",
    "fused_cdl_matmul",
    "set_force_u",
    "ue4m3_tables",
]
