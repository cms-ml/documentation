# Direct inference with (Q)ONNX Runtime

[![ReadTheDocs](https://readthedocs.org/projects/qonnx/badge/?version=latest&style=plastic)](http://qonnx.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/qonnx.svg)](https://badge.fury.io/py/qonnx)
[![arxiv](https://img.shields.io/badge/arXiv-2206.07527-b31b1b.svg)](https://arxiv.org/abs/2206.07527)

Text taken and adopted from the QONNX [README.md](https://github.com/fastmachinelearning/qonnx/blob/main/README.md).

<img align="left" src="https://xilinx.github.io/finn/img/TFC_1W2A.onnx.png" alt="QONNX example" style="margin-right: 20px" width="200"/>

QONNX (Quantized ONNX) introduces three new custom operators -- `Quant`, `BipolarQuant`, and `Trunc` -- in order to represent arbitrary-precision uniform quantization in [ONNX](onnx.md). This enables:

* Representation of binary, ternary, 3-bit, 4-bit, 6-bit or any other quantization.
* Quantization is an operator itself, and can be applied to any parameter or layer input.
* Flexible choices for scaling factor and zero-point granularity.
* Quantized values are carried using standard `float` datatypes to remain [ONNX](onnx.md) protobuf-compatible.

This repository contains a set of Python utilities to work with QONNX models, including but not limited to:

* executing QONNX models for (slow) functional verification
* shape inference, constant folding and other basic optimizations
* summarizing the inference cost of a QONNX model in terms of mixed-precision MACs, parameter and activation volume
* Python infrastructure for writing transformations and defining executable, shape-inferencable custom ops
* (experimental) data layout conversion from standard ONNX NCHW to custom QONNX NHWC ops
