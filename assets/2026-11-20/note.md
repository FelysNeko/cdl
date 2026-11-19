# Activation Quantization

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C2

## Preliminary Result

The same fine-tuning experiments from last week. This time, quantizers are applied to both weights and activations. WikiText-2, C4, HellaSwag, ARC-Easy, and PIQA are reported, as the remaining benchmarks (WinoGrande, OpenBookQA, ARC-Challenge, SIQA) fall largely within noise.

- Only LoRA is trainable, so the quantizers add no learnable parameters
- AdamW optimizer for everything, with $`\eta = 3 \cdot 10^{-4}`$, training for 4000 steps, i.e., 65.5M tokens
- $`\kappa_{forward}=16`$ and $`\kappa_{backward}=2`$ for activation quantizers, and $`\kappa=64`$ for weight quantizers

### SmolLM2-135M

Trainable parameters: 4.88M

| Variants              | WikiText-2 (↓) | C4 (↓) | HellaSwag (↑) | ARC-Easy (↑) | PIQA (↑) |
| --------------------- | :------------: | :----: | :-----------: | :----------: | :------: |
| Original Weights      |     15.43      | 21.62  |     42.04     |    64.44     |  68.28   |
| BF16 w/o Quantization |     15.77      | 21.68  |     41.98     |    63.05     |  68.34   |
| NVFP4 STE             |     19.61      | 26.01  |     39.73     |    56.78     |  63.60   |
| NVFP4 Extended CDL    |     19.08      | 25.52  |     39.89     |    59.60     |  64.64   |
| MXFP4 STE             |     25.72      | 32.69  |     36.74     |    54.84     |  62.30   |
| MXFP4 Extended CDL    |     23.14      | 30.00  |     37.54     |    56.27     |  62.73   |
| MXFP8 STE             |     16.11      | 22.08  |     41.67     |    62.12     |  67.79   |
| MXFP8 Extended CDL    |     16.19      | 22.24  |     41.66     |    63.43     |  67.79   |

### SmolLM2-360M

Trainable parameters: 8.68M

| Variants              | WikiText-2 (↓) | C4 (↓) | HellaSwag (↑) | ARC-Easy (↑) | PIQA (↑) |
| --------------------- | :------------: | :----: | :-----------: | :----------: | :------: |
| Original Weights      |     11.47      | 16.98  |     54.47     |    70.33     |  71.87   |
| BF16 w/o Quantization |     11.61      | 16.85  |     54.76     |    70.66     |  71.65   |
| NVFP4 STE             |     13.57      | 19.16  |     51.63     |    67.93     |  69.91   |
| NVFP4 Extended CDL    |     13.29      | 18.87  |     51.99     |    67.55     |  69.31   |
| MXFP4 STE             |     17.17      | 23.09  |     46.69     |    63.51     |  66.38   |
| MXFP4 Extended CDL    |     15.61      | 21.42  |     48.28     |    65.66     |  67.95   |
| MXFP8 STE             |     11.79      | 17.07  |     54.20     |    70.12     |  72.25   |
| MXFP8 Extended CDL    |     11.86      | 17.20  |     54.17     |    70.83     |  71.55   |
