---
marp: true
math: mathjax
paginate: true
---

# One Step Forward

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C1 & C2

---

## The Four Bits Era: W4-A4-G4-KV4

Following the official NVFP4 [recipe](https://arxiv.org/abs/2509.25149) from NVIDIA, we additionally apply:

- $\uparrow$ Random Hadamard transforms (RHT) to bound outliers for backward gradients
- $\uparrow$ Stochastic rounding for unbiased gradient estimation
- $\downarrow$ 2D ($16 \times 16$) block scales for weights instead of 1D ($1 \times 16$), unfortunately

_Note: it only stabilizes backward propagation, and is orthogonal to CDL variants._

---

![](recipe.png)

_Figure 5 from [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) (NVIDIA et al., 2025)._

---

## Experiments

The same fine-tuning tasks as in the last two weeks, except that

- All the KV-caches are quantized, similar to the activations
- The learning rate scheduler is changed from constant to cosine decay
- Full fine-tuning instead LoRA which is more similar to pre-training

Only NVFP4 on SmolLM2-360M is reported for this week unless specified.

---

## Weight Block Scaling

2D means each $16 \times 16$ block shares one scale, versus $1 \times 16$ for 1D. Acitvation and KV caches are proxied by STE for simplicity. We focus on the weights only.

| Variants | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| -------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| STE-1D   |
| STE-2D   |
| ECDL-1D  |
| ECDL-2D  |
| CCDL-1D  |
| CCDL-2D  |

---

## NVFP4 Backward Ablation

Same, acitvation and KV caches are proxied by STE for simplicity.

| Step     | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 |
| -------- | :-: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| STE-G16  |
| STE-G4   |
| ECDL-G16 |
| ECDL-G4  |
| CCDL-G16 |
| CCDL-G4  |

---

## Short Conlusion

The full-stack NVFP4 recipe seems to work fine for all. Then we shall make decisions.

---

## Deisgn Choices

Weights, activations, KV caches, and gradients are all in NVFP4.

| Variants (W-A) | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| -------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| STE-STE        |
| STE-ECDL       |
| ECDL-STE       |
| ECDL-ECDL      |
| CCDL-STE       |
| CCDL-ECDL      |

---

## Progressive Quantization

Ordered from easiest to hardest, at the finest possible granularity.

| Variants            | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| ------------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Base                |
| W16-A16-G16-KV16    |
| W4(1D)-A16-G16-KV16 |
| W4(1D)-A4-G16-KV16  |
| W4(1D)-A4-G16-KV4   |
| W4(2D)-A4-G4-KV4    |

---

## Different Model Sizes

Given that CCDL-STE is the best variant, we compare it againt STE-STE baseline only.

| Variants      | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| ------------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| 135M-STE-STE  |
| 135M-CCDL-STE |
| 360M-STE-STE  |
| 360M-CCDL-STE |
| 1.7B-STE-STE  |
| 1.7B-CCDL-STE |

---

## Caveat

We cannot conclude much things at the moment. Pre-training is more sensitive to the accumulated error during training, but also has more flexibility adjust the weights.
