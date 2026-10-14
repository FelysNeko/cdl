# Coded Deep Learning

This is an unofficial implementation of [Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) with additional FP4 and FP8 support. However, I found the paper difficult to reproduce. Therefore, I changed the model to train `kappa` instead of `alpha`, where `kappa` is equivalent to `alpha * step ** 2`. This prevents `alpha`'s gradient from being scaled by `step`. However, simply setting `kappa` to `1` and excluding it from training leads to better performance. Batch normalization re-estimation is required since the quantization is softer. Additionally, I normalized the entropy term.

## Implementation

Below is an unoptimized and faithful implementation of the paper:

```python
import torch
from torch.distributions import Categorical


def cdl_topk_forward(
    relaxed: bool,
    num_quantized_params: int,
    topk: int,
    theta: torch.Tensor,
    a: torch.Tensor,
    q: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = -alpha * (theta.unsqueeze(-1) - a * q) ** 2
    topk_values, topk_indices = logits.topk(topk)
    pmf = torch.softmax(topk_values, dim=-1)

    groups = pmf.numel() // (num_quantized_params * topk)
    idx = topk_indices.reshape(groups, num_quantized_params, topk)

    mpmf = pmf.new_zeros(groups, a.numel())
    mpmf.scatter_add_(1, idx.reshape(groups, -1), pmf.reshape(groups, -1))
    mpmf = mpmf / num_quantized_params

    tiny = torch.finfo(mpmf.dtype).tiny
    entropies = -(mpmf * mpmf.clamp_min(tiny).log2()).sum(-1)
    entropies = num_quantized_params * entropies.mean()

    q_d = (pmf * a[topk_indices]).sum(-1)

    if relaxed:
        return q_d, entropies

    with torch.no_grad():
        sub_indices = Categorical(probs=pmf).sample().unsqueeze(-1)
        chosen = topk_indices.gather(dim=-1, index=sub_indices).squeeze(-1)
        q_p = a[chosen]

    return q_d + (q_p - q_d).detach(), entropies
```

## References

The following papers helped me a lot. Also, ask LLMs.

- [Learnable Companding Quantization for Accurate Low-bit Neural Networks](https://arxiv.org/abs/2103.07156)
- [Learned JPEG Compression for DNN Vision](https://arxiv.org/abs/2606.16185)
- [JPEG Inspired Deep Learning](https://arxiv.org/abs/2410.07081)

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
