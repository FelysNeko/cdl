---
marp: true
math: mathjax
paginate: true
---

# Activation Quantization

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C1 & C2

---

## Preliminary Result

The same fine-tuning experiments from last week. This time, quantizers are applied to both weights and activations. WikiText-2, C4, HellaSwag, ARC-Easy, and PIQA are reported, as the remaining benchmarks (WinoGrande, OpenBookQA, ARC-Challenge, SIQA) fall largely within noise.

- Only LoRA is trainable, so the quantizers add no learnable parameters
- AdamW optimizer with $\eta = 3 \cdot 10^{-4}$, trained for 4000 steps, i.e., 65.5M tokens
- $\kappa_{fwd}=16$ and $\kappa_{bwd}=2$ for activation quantizers, and $\kappa=64$ for weight quantizers

---

### SmolLM2-135M

Trainable parameters: 4.88M

| Variants              | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| --------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Original Weights      |        15.43        |      21.62      |        42.04         |        64.44        |      68.28      |
| BF16 w/o Quantization |        15.77        |      21.68      |        41.98         |        63.05        |      68.34      |
| NVFP4 STE             |        19.82        |      26.02      |        39.77         |        56.99        |      64.91      |
| NVFP4 Extended CDL    |        19.08        |      25.52      |        39.89         |        59.60        |      64.64      |
| MXFP4 STE             |        24.37        |      31.12      |        37.48         |        55.72        |      62.89      |
| MXFP4 Extended CDL    |        23.14        |      30.00      |        37.54         |        56.27        |      62.73      |

---

### SmolLM2-360M

Trainable parameters: 8.68M

| Variants              | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| --------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Original Weights      |        11.47        |      16.98      |        54.47         |        70.33        |      71.87      |
| BF16 w/o Quantization |        11.61        |      16.85      |        54.76         |        70.66        |      71.65      |
| NVFP4 STE             |          -          |        -        |          -           |          -          |        -        |
| NVFP4 Extended CDL    |        13.29        |      18.87      |        51.99         |        67.55        |      69.31      |
| MXFP4 STE             |          -          |        -        |          -           |          -          |        -        |
| MXFP4 Extended CDL    |        15.61        |      21.42      |        48.28         |        65.66        |      67.95      |

---

### SmolLM2-1.7B

Trainable parameters: 18.09M

| Variants              | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| --------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Original Weights      |        8.12         |      12.60      |        69.39         |        77.82        |      76.82      |
| BF16 w/o Quantization |        8.18         |      12.65      |        69.22         |        77.53        |      76.88      |
| NVFP4 STE             |        9.17         |      13.85      |        66.81         |        75.29        |      75.46      |
| NVFP4 Extended CDL    |        9.14         |      13.76      |        67.24         |        75.55        |      75.68      |
| MXFP4 STE             |          -          |        -        |          -           |          -          |        -        |
| MXFP4 Extended CDL    |          -          |        -        |          -           |          -          |        -        |

---

### Wall-time

Pure training wall-time for 4000 steps on an RTX 5070 Ti with `torch.compile` enabled.

| Variants              | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B |
| --------------------- | :----------: | :----------: | :----------: |
| BF16 w/o Quantization |   14m 20s    |   34m 56s    |  2h 30m 02s  |
| NVFP4 STE             |   14m 04s    |      -       |  2h 05m 24s  |
| NVFP4 Extended CDL    |   19m 49s    |   43m 46s    |  3h 04m 23s  |
| MXFP4 STE             |   14m 40s    |      -       |      -       |
| MXFP4 Extended CDL    |   18m 25s    |   39m 10s    |      -       |

Memory only fits the 135M model with the full 16 batches; 360M and 1.7B need batch sizes of 8 and 2, with gradient accumulation to keep tokens per update equivalent. The GEMM size grows less, and the speedup is limited.

---

## More Questions

Some minor but interesting questions:

- Cascaded CDL for weights and Extended CDL for activations?
- Low-precision backward on Transformers?
- NVFP4 attention (surprisingly, this kernel is available)?
- Performance when Shannon entropy term is enabled?

A bigger question: how can we take advantage of a low-entropy model?

---

## Next Move

For each potential selling point (can be parallelized):

- Training speed: GEMM must be the bottleneck, so LoRA training for a ~30B dense model or larger is a good task. Precision doesn't matter. An RTX PRO 6000 is required.
- Inference speed: KV-cache bandwidth and/or GEMM must be the bottleneck. Test pre-fill and decode performance on vLLM or SGLang. Current lab infrastructure (RTX 6000 Ada) is sufficient if we discard the GEMM speedup. A larger model is preferred.
- High precision: post-training quantization via supervised fine-tuning (current work) or distillation. Eventually, we apply the method to SOTA small models (e.g., Qwen3-32B) and verify it on math and coding benchmarks.

We avoid mixture-of-experts and linear attention models at the moment.

---

## Faster Low-precision Pre-training with High Precision

Pre-train two SmolLM2-135M models on an RTX PRO 6000: BF16 as the benchmark vs. FP4 CDL variants with everything quantized. Both consume 5B tokens and would take around 1 day each. We compare the loss curves and run some simple benchmarks.

Note: the original SmolLM2 paper uses 2T tokens, which is not practical for us, so we have to train the baseline ourselves for a fair comparison.
