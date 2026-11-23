---
marp: true
math: mathjax
paginate: true
---

# One Step Back

[Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) - C1 & C2

---

## Does LSQ Still Fit LLMs?

_Challenge: Transformers exhibit severe, dynamically shifting outliers._

The industry has introduced the fine-grained format to address this problem. If LSQ is still applied, the number of trainable parameters increases by roughly 5%, potentially moving closer to the global optimum. However, it fails to address activation outliers (see the previous analysis of batch-independent block scales), the major problem in Transformers.

---

## Industry Choices

It seems that dynamically computed scales are the trend.

- NVIDIA [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) is an application of Dual-LSQ on LLMs, but it is only designed for post-training fine-tuning instead of pre-training. (Cascaded CDL)
- [BitNet](https://arxiv.org/abs/2402.17764) from Microsoft pre-trained an LLM using 1.5-bit weight quantization, but the per-tensor scale is still derived from statistics. (Extended CDL)
- The official full-stack NVFP4 pre-training [recipe](https://arxiv.org/abs/2509.25149) from NVIDIA still uses STE as the gradient proxy and computes scales from statistics. (Extended CDL)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) was trained on FP8 with a customized block-scale design that has coarser granularity than MXFP8, with the scales also computed from statistics. (Extended CDL)

---

## Decision

It's time to move from Cascaded CDL to Extended CDL. Current QAT and QAD experiments either show that the former does not outperform the latter, or the benchmark results are mostly noise.

---

## KV-Cache Quantization

The same fine-tuning tasks as in the last two weeks, except that

- All the KV-caches are quantized, similar to the activations
- The learning rate scheduler is changed to cosine
- Only NVFP4 on SmolLM2-360M is reported

---

## Metrics

The same common benchmarks as before:

| Variants | $\downarrow$ Wiki-2 | $\downarrow$ C4 | $\uparrow$ HellaSwag | $\uparrow$ ARC-Easy | $\uparrow$ PIQA |
| -------- | :-----------------: | :-------------: | :------------------: | :-----------------: | :-------------: |
| Base     |        11.47        |      16.98      |        54.47         |        70.33        |      71.87      |
| STE      |        13.96        |      19.65      |        51.28         |        67.17        |      70.18      |
| ECDL     |        13.79        |      19.46      |        51.31         |        68.01        |      70.40      |

Test perplexity, evaluated every 500 steps:

| Step |  500  | 1000  | 1500  | 2000  | 2500  | 3000  | 3500  | 4000  |
| ---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| STE  | 15.24 | 15.19 | 15.20 | 15.16 | 15.04 | 15.01 | 14.96 | 14.99 |
| ECDL | 15.24 | 15.01 | 14.98 | 14.95 | 14.93 | 14.88 | 14.91 | 14.90 |

---

## Caveat

Neither the QAT nor the QAD experiment shows a significant difference. We cannot conclude anything at the moment until we have finished pre-training, which is more sensitive to the accumulated error during training.
