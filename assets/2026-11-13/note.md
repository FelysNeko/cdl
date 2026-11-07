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
