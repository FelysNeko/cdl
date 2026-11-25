---
marp: true
math: mathjax
paginate: true
---

# One Step Forward

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
| STE      |                     |                 |                      |                     |                 |
| ECDL     |                     |                 |                      |                     |                 |

Test perplexity, evaluated every 500 steps:

| Step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
| ---- | :-: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| STE  |     |      |      |      |      |      |      |      |
| ECDL |     |      |      |      |      |      |      |      |

---

## The Four Bits Era: W4-A4-G4-KV4

Following the official NVFP4 [recipe](https://arxiv.org/abs/2509.25149) from NVIDIA, we additionally apply:

- $\uparrow$ Random Hadamard transforms (RHT) to bound outliers for backward gradients
- $\uparrow$ Stochastic rounding for unbiased gradient estimation
- $\downarrow$ 2D ($16 \times 16$) block scales for weights instead of 1D ($1 \times 16$), unfortunately

_Note: it only stabilizes backward propagation, and is orthogonal to Extended CDL._

---

![](recipe.png)

---

## NVFP4 Backward Ablation

Test perplexity, evaluated every 500 steps:

| Step     |  500  | 1000  | 1500  | 2000  | 2500  | 3000  | 3500  | 4000  |
| -------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| STE-G16  | 15.99 | 15.99 | 15.86 | 15.79 | 15.63 | 15.52 | 15.45 | 15.45 |
| STE-G4   |       |       |       |       |       |       |       |       |
| ECDL-G16 | 15.80 | 15.60 | 15.55 | 15.41 | 15.36 | 15.35 | 15.39 | 15.35 |
| ECDL-G4  |       |       |       |       |       |       |       |       |

NVFP4 backward precision works well for this fine-tuning task. As expected, Extended CDL consistently outperforms STE on test perplexity.

---

## Progressive Quantization

Ordered from easiest to hardest, at the finest possible granularity.

| Variants            | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| ------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Base                |        11.47        |      16.98      |        54.47         |        69.49        |      71.82      |
| W16-A16-G16-KV16    |        11.56        |      16.82      |        54.76         |        68.43        |      72.52      |
| W4(1D)-A16-G16-KV16 |        12.16        |      17.53      |        53.51         |        67.13        |      71.55      |
| W4(1D)-A4-G16-KV16  |                     |                 |                      |                     |                 |
| W4(1D)-A4-G16-KV4   |                     |                 |                      |                     |                 |
| W4(2D)-A4-G4-KV4    |                     |                 |                      |                     |                 |

---

## Weight Block Scaling

2D means each $16 \times 16$ block shares one scale, versus $1 \times 16$ for 1D.

| Variants            | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| ------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| W4(1D)-A16-G16-KV16 |        12.16        |      17.53      |        53.51         |        67.13        |      71.55      |
| W4(2D)-A16-G16-KV16 |        12.62        |      18.04      |        52.82         |        67.00        |      71.06      |
| W4(1D)-A4-G16-KV16  |                     |                 |                      |                     |                 |
| W4(2D)-A4-G16-KV16  |                     |                 |                      |                     |                 |
| W4(1D)-A4-G16-KV4   |                     |                 |                      |                     |                 |
| W4(2D)-A4-G16-KV4   |        14.33        |      20.00      |        50.29         |        63.93        |      69.70      |

---

## Caveat

Due to the nature of LoRA and limited training steps, we cannot conclude much things at the moment. Pre-training is more sensitive to the accumulated error during training, but also has more flexibility adjust the weights.
