# Fused Cascaded-CDL NVFP4 kernels

Fused Triton kernels + flashinfer fp4 GEMM for training-time cascaded CDL
quantization. One kernel launch per tensor fuses the entire
`cascaded_cdl_topk_forward` companded path (q-level UE4M3 pass, theta-level
E2M1 pass, top-k sampling, NVFP4 packing, block scales) with an analytic
backward kernel; the matmul runs on the flashinfer `mm_fp4` kernel.

## Modules

- `tables.py` — `e2m1_tables` / `ue4m3_tables`: level/gap lookup tables
  (cached per device).
- `quant.py` — the Triton kernels and the `cdl::cdl_quant_fwd` /
  `cdl::cdl_quant_bwd` custom ops (registered at import). `set_force_u`
  pins the sampling quantile for deterministic tests.
- `matmul.py` — `FusedCdlMatmul` autograd Function,
  `fused_cdl_matmul(theta_x, quant_x, theta_w, quant_w)` entry point and
  `fusable_quant` compatibility check.

Import `cdl.core.kernel.matmul` (or `quant`) directly; there is no package
re-export.

## Contract

- Formats are hard-coded: E2M1 theta alphabet, UE4M3ExcludeZero scale
  alphabet, NVFP4 block size 16, `K % 16 == 0`, contiguous inputs, 2-D.
- The companded path only (`companded=True`); entropy terms are not
  implemented (training passes None).
- The seed is an int64 device scalar advanced inside the graph
  (`quant.kernel_seed`), so CUDA-graph replays resample.
- Under `torch.compile`, run one eager step first so flashinfer caches its
  workspace outside the CUDA-graph pool.

Env knobs: `CDL_NUM_WARPS`, `CDL_BB` (blocks per program), `CDL_FP4_BWD`
(backward GEMM precision: `auto` enables nvfp4 gradient GEMMs for
M·N·K >= 5e9).

## Performance envelope

The path wins when GEMM math dominates: at [4096, 7168] x [8192, 7168] it
runs fwd+bwd at ~0.65x the bf16 baseline and OOM-free where the decomposed
eager path cannot fit. At small shapes (ResNet-20/CIFAR: im2col M=65536,
K=144) the step is quantization-overhead-bound — the quant kernels and the
im2col/col2im layout conversions are ~25x the actual GEMM time — so use the
plain fp4 or native backends there.
