# Hardware Accelerated CDL

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C2

## Quantization Walk-through

**Shape**: `[64, 32, 3, 3]`

**MXFP4**: 32 E2M1 floating-point numbers scaled by one E8M0 factor per block.

### Parameters

Reshape the weight `[64, 32, 3, 3]` into `[64, 288]` and pad it into `[64, 288]` (no padding required for this case). Then reshape it into `[64, 9, 32]`, where `64` is the output feature dimension and `32` is the block size. This implies `[64, 9, 1]` scalers.

Thus, the trainable parameters are:

- Weight: `[64, 9, 32]` stored in FP32, targeted FP4E2M1
- Scaler: `[64, 9, 1]` stored in FP32, targeted FP8E8M0

Todo: decide whether the sharpness factor should be block-wise or tensor-wise.

### Forward Pass

![](assets/diagram.png)

The critical part is the MXFP4 GEMM, which takes the weight in MXFP4 and the activation in any format and outputs a high-precision activation, e.g., BF16. We may choose to quantize this activation back to MXFP4 to achieve W4A4.

### Backward Pass

Given that the forward pass is fully differentiable, the backward pass does not need any STE proxy. Gradients are stored in FP32, but we can use stochastic rounding to quantize them and apply MXFP4 GEMM to them and to the saved tensor from the forward pass.

## Other Formats

MXFP8 uses the E4M3 layout for the weights and E8M0 for the scaler, which is the same scaler as in MXFP4.

NVFP4 is a proprietary format designed by Nvidia that has a higher precision than MXFP4. The block size is 16 instead of 32, and the scaler uses the E4M3 layout. Additionally, each tensor has an FP32 global scaler.
