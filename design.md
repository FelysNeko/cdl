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

Note: this implementation does not train `alpha`.

### Forward Pass

Define the CDL function signature as follows:

```python
# note: `a` is omitted since it remains unchanged
def cdl(theta: Tensor, q: Tensor) -> Tensor: ...
```

We can compute `s = cdl(scaler, 1)`, getting a tensor in shape `[64, 9, 1]` stored in `uint8` or an equivalent format. Note that `q` is simply 1 because the target is a fixed specification. There is no need to scale it. Then compute `quantized = cdl(weight, s)`, getting a tensor in shape `[64, 9, 32]` stored in E2M1 format or an equivalent format. The MXFP4 quantized weight is ready.

The critical part is the MXFP4 GEMM, which takes the weight in MXFP4 and the activation in any format and outputs a high-precision activation.

We may choose to quantize this activation back to MXFP4.

### Backward Pass

Given that the forward pass is fully differentiable, the backward pass does not need any STE proxy. Gradients are stored in FP32, but we can use stochastic rounding to quantize them and apply MXFP4 GEMM to them and to the saved tensor from the forward pass.

## Other Formats

MXFP8 uses the E4M3 layout for the weights and E8M0 for the scaler, which is the same scaler as in MXFP4.

NVFP4 is a proprietary format designed by Nvidia that has a higher precision than MXFP4. The block size is 16 instead of 32, and the scaler uses the E4M3 layout. Additionally, each tensor has an FP32 global scaler.
