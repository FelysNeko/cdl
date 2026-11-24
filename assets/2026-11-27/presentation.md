---
marp: true
math: mathjax
paginate: true
---

# One Step Back

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C1 & C2

---

## KV-Cache Quantization

The same fine-tuning tasks as in the last two weeks, except that

- All the KV-caches are quantized, similar to the activations
- The learning rate scheduler is changed to cosine
- Only NVFP4 on SmolLM2-360M is reported

---

## Metrics

The same common benchmarks as before (all accuracies length-normalized):

| Variants | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| -------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Base     |        11.47        |      16.98      |        54.47         |        69.49        |      71.82      |
| STE      |        13.96        |      19.65      |        51.28         |        63.93        |      71.44      |
| ECDL     |        13.79        |      19.46      |        51.31         |        65.61        |      71.16      |

Test perplexity, evaluated every 500 steps:

| Step |  500  | 1000  | 1500  | 2000  | 2500  | 3000  | 3500  | 4000  |
| ---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| STE  | 15.24 | 15.19 | 15.20 | 15.16 | 15.04 | 15.01 | 14.96 | 14.99 |
| ECDL | 15.24 | 15.01 | 14.98 | 14.95 | 14.93 | 14.88 | 14.91 | 14.90 |

---

## The Four Bits Era: W4-A4-G4-KV4

Following the official NVFP4 [recipe](https://arxiv.org/abs/2509.25149) from NVIDIA, we additionally apply:

- Random Hadamard transforms (RHT) to bound outliers for backward gradients
- Stochastic rounding for unbiased gradient estimation
- 2D ($16 \times 16$) block scales for weights instead of 1D ($1 \times 16$)

_Note: it only stabilizes backward propagation, and is orthogonal to Extended CDL._

---

## NVFP4 Backward Ablation

Test perplexity, evaluated every 500 steps:

| Step     |  500  | 1000  | 1500  | 2000  | 2500  | 3000  | 3500  | 4000  |
| -------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| STE-G16  | 15.99 | 15.99 | 15.86 | 15.79 | 15.63 | 15.52 | 15.45 | 15.45 |
| STE-G4   | 16.08 | 15.84 | 15.82 | 15.81 | 15.74 | 15.61 | 15.50 | 15.47 |
| ECDL-G16 | 15.80 | 15.60 | 15.55 | 15.41 | 15.36 | 15.35 | 15.39 | 15.35 |
| ECDL-G4  | 15.90 | 15.61 | 15.53 | 15.50 | 15.43 | 15.39 | 15.39 | 15.34 |

NVFP4 backward precision works well for this fine-tuning task. As expected, Extended CDL consistently outperforms STE on test perplexity.

---

## Progressive Quantization

Ordered from easiest to hardest.

| Variants         | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| ---------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Base             |        11.47        |      16.98      |        54.47         |        69.49        |      71.82      |
| W16-A16-G16-KV16 |                     |                 |                      |                     |                 |
| W4-A16-G16-KV16  |                     |                 |                      |                     |                 |
| W4-A4-G16-KV16   |                     |                 |                      |                     |                 |
| W4-A4-G16-KV4    |        13.79        |      19.46      |        51.31         |        65.61        |      71.16      |
| W4-A4-G4-KV4     |        14.35        |      20.01      |        50.31         |        63.68        |      69.86      |

---

## Caveat

Neither the QAT nor the QAD experiment shows a significant difference. We cannot conclude anything at the moment until we have finished pre-training, which is more sensitive to the accumulated error during training.
