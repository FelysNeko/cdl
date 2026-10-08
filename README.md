# Coded Deep Learning

This is an unofficial implementation of [Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849) with additional FP4 and FP8 support. However, I found the paper difficult to reproduce. I changed the model to train `kappa` instead of `alpha`, where `kappa` is equivalent to `alpha * step ** 2`. This removes the `step ** 2` scaling from `alpha`'s gradient.

Refer to [Learned JPEG Compression for DNN Vision](https://arxiv.org/abs/2606.16185) and [JPEG Inspired Deep Learning](https://arxiv.org/abs/2410.07081) to see how others patch it, e.g., by fixing `alpha` or `kappa`.

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
