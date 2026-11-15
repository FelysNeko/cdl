# Activation Quantization

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C2

## Preliminary Result

The same fine-tuning experiments as last week are run on SmolLM2-135M. This time, quantizers are applied to both weights and activations. Only BoolQ is reported, since the remaining metrics seem to reveal only noise.

- Only LoRA is trainable, so the quantizers add no learnable parameters
- AdamW optimizer for everything, with $`\eta = 3 \cdot 10^{-4}`$, training for 4096 steps, i.e., 67M tokens
- $`\kappa_{forward}=16`$ and $`\kappa_{backward}=2`$ for activation quantizers, and $`\kappa=64`$ for weight quantizers

| Variants              | BoolQ | Parameters |
| --------------------- | :---: | :--------: |
| Original Weights      | 60.55 |     0      |
| BF16 w/o Quantization | 59.88 |   4.88M    |
| NVFP4 STE             | 52.54 |   4.88M    |
| NVFP4 Extended CDL    | 58.72 |   4.88M    |

The full model shrinks from 270 MB to 117 MB, but inference is ~4x slower. GEMM speedup only pays off on models large enough.
