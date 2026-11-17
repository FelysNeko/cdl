# Activation Quantization

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C2

## Preliminary Result

The same fine-tuning experiments as last week are run on SmolLM2-135M. This time, quantizers are applied to both weights and activations. WikiText-2, C4, BoolQ, ARC-Easy, and PIQA are reported, as the remaining benchmarks (WinoGrande, OpenBookQA, ARC-Challenge, SIQA) fall largely within noise at 135M scale.

- Only LoRA is trainable (4.88M parameters), so the quantizers add no learnable parameters
- AdamW optimizer for everything, with $`\eta = 3 \cdot 10^{-4}`$, training for 4096 steps, i.e., 67M tokens
- $`\kappa_{forward}=16`$ and $`\kappa_{backward}=2`$ for activation quantizers, and $`\kappa=64`$ for weight quantizers

| Variants              | WikiText-2 (↓) | C4 (↓) | BoolQ (↑) | ARC-Easy (↑) | PIQA (↑) |
| --------------------- | :------------: | :----: | :-------: | :----------: | :------: |
| Original Weights      |     15.43      | 21.62  |   60.55   |    64.44     |  68.28   |
| BF16 w/o Quantization |     15.76      | 21.67  |   59.88   |    63.22     |  68.44   |
| NVFP4 STE             |     19.64      | 25.91  |   52.54   |    56.44     |  65.61   |
| NVFP4 Extended CDL    |     19.09      | 25.49  |   58.72   |    59.09     |  64.64   |
| MXFP4 STE             |     25.77      | 32.65  |   49.60   |    55.26     |  61.48   |
| MXFP4 Extended CDL    |     23.07      | 29.90  |   58.56   |    56.86     |  62.73   |
| MXFP8 STE             |     16.08      | 22.04  |   58.84   |    62.37     |  68.12   |
| MXFP8 Extended CDL    |     16.21      | 22.19  |   59.08   |    62.58     |  68.12   |
