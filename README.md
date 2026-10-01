# Coded Deep Learning

Reading [Coded Deep Learning: Framework and Algorithm](https://arxiv.org/abs/2501.09849).

## Sep 30, 2026

I suspect that the official implementation differs slightly from the paper. Setting the initial alpha of the activation quantizer to `500.0` leads to gradient explosion at the very beginning. The entropy term is not normalized and dominates the training. I cannot reproduce their results.

## License

Distributed under the terms of the [LICENSE](LICENSE).

## Copyright

© All rights reserved by FelysNeko
