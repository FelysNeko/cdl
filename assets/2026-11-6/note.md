# Faster Cascaded CDL II

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C2

## Fused Cascaded CDL GEMM Benchmark

The benchmark results from last week used low-precision matrix multiplications for the backward instead of `bf16`, which is a bug. The memory usage of the torch implementation for `nvfp4` is also overestimated, because the measurement counted the eager warmup, which dominates the peak memory usage and runs out of memory easily.

Here are the new tables.

### Time

Wall-time in ms is the sum of the forward median and the backward median.

| shape (M, K, N)                  |      bf16 | mxfp8 (triton) | mxfp4 (triton) |     nvfp4 (triton) | nvfp4 (torch) |
| -------------------------------- | --------: | -------------: | -------------: | -----------------: | ------------: |
| small (64, 512, 512)             |  **0.51** |   0.79 (1.55x) |   0.71 (1.39x) |       0.63 (1.25x) |  0.77 (1.51x) |
| medium (512, 1024, 1024)         |  **0.52** |   0.76 (1.46x) |   0.75 (1.44x) |       0.70 (1.36x) |  0.86 (1.66x) |
| large (2048, 2048, 2048)         |  **0.79** |   1.95 (2.48x) |   1.66 (2.11x) |       1.20 (1.53x) |  3.94 (5.01x) |
| 7b-attn (4096, 4096, 4096)       |  **4.59** |   8.65 (1.89x) |   7.33 (1.60x) |       5.56 (1.21x) | 16.76 (3.65x) |
| 7b-qkv (4096, 4096, 12288)       | **13.54** |  22.61 (1.67x) |  18.91 (1.40x) |      14.85 (1.10x) | 40.40 (2.98x) |
| 7b-mlp-s4k (4096, 4096, 11008)   | **12.17** |  19.85 (1.63x) |  16.80 (1.38x) |      13.26 (1.09x) | 35.61 (2.93x) |
| 7b-mlp-s8k (8192, 4096, 11008)   |     23.92 |  31.03 (1.30x) |  27.33 (1.14x) |  **22.68 (0.95x)** | 49.62 (2.07x) |
| 7b-mlp-s16k (16384, 4096, 11008) |     47.84 |  53.38 (1.12x) |  48.21 (1.01x) |  **41.79 (0.87x)** | 78.04 (1.63x) |
| 70b-mlp (4096, 8192, 28672)      | **62.43** |  96.14 (1.54x) |  80.73 (1.29x) |      64.57 (1.03x) |             - |
| 70b-mlp-s8k (8192, 8192, 28672)  |    122.24 | 148.16 (1.21x) | 129.38 (1.06x) | **109.82 (0.90x)** |             - |

### Peak Memory

Peak memory in MiB is sampled in-process via `torch.cuda.mem_get_info` throughout the run and reported as the increase over the post-initialization baseline, so it covers the CUDA context, allocator segments and all transient buffers.

| shape (M, K, N)                  |     bf16 | mxfp8 (triton) | mxfp4 (triton) |  nvfp4 (triton) |  nvfp4 (torch) |
| -------------------------------- | -------: | -------------: | -------------: | --------------: | -------------: |
| small (64, 512, 512)             |  **166** |    204 (1.23x) |    206 (1.24x) | **156 (0.94x)** |    184 (1.11x) |
| medium (512, 1024, 1024)         |  **188** |    228 (1.21x) |    248 (1.32x) |     198 (1.05x) |    316 (1.68x) |
| large (2048, 2048, 2048)         |  **226** |    282 (1.25x) |    220 (0.97x) | **204 (0.90x)** |    906 (4.01x) |
| 7b-attn (4096, 4096, 4096)       |  **310** |    390 (1.26x) |    364 (1.17x) |     304 (0.98x) |  3110 (10.03x) |
| 7b-qkv (4096, 4096, 12288)       |  **502** |    792 (1.58x) |    682 (1.36x) |     638 (1.27x) |  8248 (16.43x) |
| 7b-mlp-s4k (4096, 4096, 11008)   |  **472** |    738 (1.56x) |    628 (1.33x) |     586 (1.24x) |  6652 (14.09x) |
| 7b-mlp-s8k (8192, 4096, 11008)   |  **638** |   1002 (1.57x) |    928 (1.45x) |     872 (1.37x) |  7610 (11.93x) |
| 7b-mlp-s16k (16384, 4096, 11008) |  **922** |   1482 (1.61x) |   1338 (1.45x) |    1302 (1.41x) | 11105 (12.04x) |
| 70b-mlp (4096, 8192, 28672)      | **1398** |   2942 (2.10x) |   2592 (1.85x) |    2458 (1.76x) |              - |
| 70b-mlp-s8k (8192, 8192, 28672)  | **1766** |   3226 (1.83x) |   2830 (1.60x) |    3374 (1.91x) |              - |
