# Coded Deep Learning

This is an unofficial implementation of [Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) with additional FP4 and FP8 support. However, I found the paper difficult to reproduce. Therefore, I changed the model to train `kappa` instead of `alpha`, where `kappa` is equivalent to `alpha * step ** 2`. This prevents `alpha`'s gradient from being scaled by `step`. However, simply setting `kappa` to `1` and excluding it from training leads to better performance. Batch normalization re-estimation is requires since the quantization is softer. Additionally, I normalized the entropy term.

## References

The following papers helped me a lot. Also, ask LLMs.

- [Learnable Companding Quantization for Accurate Low-bit Neural Networks](https://arxiv.org/abs/2103.07156)
- [Learned JPEG Compression for DNN Vision](https://arxiv.org/abs/2606.16185)
- [JPEG Inspired Deep Learning](https://arxiv.org/abs/2410.07081)

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
