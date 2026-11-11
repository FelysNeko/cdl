# The Big Picture

## Family of Formats

`✓` supported · `△` emulated by another datapath · `✗` not supported

|                          | INT8 | INT4 | FP8 | MXFP8 | FP4 | MXFP4 | NVFP4 |
| ------------------------ | :--: | :--: | :-: | :---: | :-: | :---: | :---: |
| SM89                     |  ✓   |  ✓   |  ✓  |   ✗   |  ✗  |   ✗   |   ✗   |
| SM100                    |  ✓   |  △   |  ✓  |   ✓   |  ✓  |   ✓   |   ✓   |
| SM120                    |  ✓   |  △   |  ✓  |   ✓   |  ✓  |   ✓   |   ✓   |
| Block Scaler (Hardware)  |  ✗   |  ✗   |  ✗  |   ✓   |  ✗  |   ✓   |   ✓   |
| Global Scaler (Software) |  ✓   |  ✓   |  ✓  |   ✗   |  ✓  |   ✗   |   ✓   |

Examples of acceleration cards for each architecture:

- **SM89** (Ada): RTX 4070 Super, RTX 6000 Ada, L40S, L4
- **SM100/103** (Blackwell): B200, B300, GB200, GB300
- **SM120/121** (Blackwell): RTX 5070 Ti, RTX Pro 6000 Blackwell, DGX Spark

## Quantization Reasons

|           | Weight                                                                                          | Activation                                                                                          | KV-Cache                                                                                           |
| --------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Bandwidth | 🔥🔥🔥 Reduces weight-loading bandwidth, the bottleneck in small-batch decoding.                | 🔥 Reduces activation read/write bandwidth; mainly matters for large batches or prefill.            | 🔥🔥 Reduces KV-cache read bandwidth, the bottleneck in long-context decoding.                     |
| GEMM      | 🔥🔥 Easy to quantize, but it does not speed up GEMM unless activations are also low precision. | 🔥🔥🔥 Harder to quantize, but combining it with weight quantization unlocks the full GEMM speedup. | 🔥 Low-precision attention kernels are uncommon, and attention is usually sensitive to precision.  |
| Memory    | 🔥🔥🔥 Smaller model footprint for storage and loading.                                         | 🔥 Minor for inference; matters for training, where activations are kept for backward.              | 🔥🔥🔥 Grows $`O(n)`$ with context length, so quantization saves a lot for long-context inference. |

## Challenges

BS stands for block scaler for NVFP4.

- Linear: `N` for batch size, `K` for input features, `M` for output features.
- KV Cache: `B` for batch size, `H` for number of heads, `T` for sequence length, `D` for head dimension.

| Type       | Tensor         | Expected BS       | Batch-independent BS | Learnable BS Counts |
| ---------- | -------------- | ----------------- | -------------------- | ------------------- |
| Activation | `[N, K]`       | `[N, K/16]`       | `[1, K/16]`          | `K/16`              |
| Weight     | `[M, K]`       | `[M, K/16]`       | `[M, K/16]`          | `M*K/16`            |
| KV Cache   | `[B, H, T, D]` | `[B, H, T, D/16]` | `[1, H, 1, D/16]`    | `H*D/16`            |

In other words, static quantization block scalers only work perfectly on weights. For activations, all samples must share the same block scalers, whereas dynamically computed scalers have no such restrictions.

## Case Studies

### DeepSeek v4.1

### Kimi K3

## Preliminary Results

HellaSwag accuracy on SmolLM2-135M. Activations are dynamically quantized and approximated using a straight-through estimator (STE). Weights are quantized using STE or Cascaded CDL. The target modules are all linear layers except for the final output head, and their matrix multiplications are performed using NVFP4 kernels. The entire base model is frozen.

- **Dataset**: 100k documents sampled from the final-stage pre-training corpus of SmolLM2
- **LoRA**: $`r=16`$ and $`\alpha=32`$
- **LR Scheduler**: constant with a 200/1500-step warmup
- **Optimizers**: SGD ($`\eta=0.1`$) for the quantizers and AdamW ($`\eta=10^{-4}`$) for the rest
- **Batch**: batch size of 16 with a sequence length of 1024
- **CDL**: `amax` instead of `mean` for scaler initialization, and $`\kappa=64`$

| Weight Quantization Variants        |  Test PPL |    HS Acc | HS Acc (normalized) | Trainable Params |
| ----------------------------------- | --------: | --------: | ------------------: | ---------------- |
| None (BF16 Base)                    |     15.76 |     34.77 |               41.86 | 0                |
| None (NVFP4 direct conversion)      |     22.67 |     32.15 |               38.58 | 0                |
| STE with LoRA                       |     19.67 |     33.25 |               39.73 | 4.884M           |
| Cascaded CDL only scalers trainable |     21.12 |     32.73 |               39.59 | 6.636M           |
| Cascaded CDL with LoRA              | **19.37** | **33.33** |           **39.87** | 11.520M          |

Full HellaSwag, 10,042 docs. Test perplexity is on the same 64-block test split.
