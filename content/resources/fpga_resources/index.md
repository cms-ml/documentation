# CMS-ML FPGA Resource
## Introduction
Welcome to FPGA Resource tab! Our tab is designed to provide accurate, up-to-date, and relevant information about tooling used in CMS for ML on FPGAs.

## FPGA Basics

Field-Programmable Gate Array (FPGA) are  reconfigurable hardware for creating custom digital circuits. FPGAs are build out of Configurable Logic Blocks (CLBs), Programmable Interconnects, and I/O Blocks. They are used in areas where high parallel processing and high performance is needed.

To programm an FPGA one has two main options. The first one is to use a Hardware Design Languages (HDL) while the second one is to use High-Level Design Tools (HLS). In the HDL case one has the control about everything but the design flow can be kind of tricky. Mostly two languages are used VHDL, which is verbose and structured, and Verilog, which is more compact and looks more like a c-style language. In CMS most of the FPGAs are programmed using VHDL.

When it comes to vendors there are two big onces: Xilinx (part of AMD) and Altera (part of Intel). Both vendors provide there own tooling to simulate, synthesis and debug the design. Xilinx FPGAs are used for CMS and they have Vivado for design, simulation, synthesis, and debugging tasks and Vitis for software development for Xilinx FPGAs and SoCs. Intel FPGAs are programmed using Quartus Prime. For HLS tools they come with Vivado HLS (Xilinx) and HLS Compiler (Intel).

To simplify the pipeline from a trained model to an implementation on the FPGA CMS is supporting different tools, which will be explained in the flowing in more detail.

## 1. hls4ml

![image](https://github.com/fastmachinelearning/fastmachinelearning.github.io/raw/master/images/hls4ml_logo.svg)

[![Documentation Status](https://github.com/fastmachinelearning/hls4ml/actions/workflows/build-sphinx.yml/badge.svg)](https://fastmachinelearning.org/hls4ml)
[![PyPI version](https://badge.fury.io/py/hls4ml.svg)](https://badge.fury.io/py/hls4ml)

#### Description
hls4ml is a Python library designed to bring machine learning inference to FPGAs by leveraging high-level synthesis (HLS). The idea is to convert trained machine learning models from popular open-source frameworks (such as PyTorch, Tensorflow, Keras etc.) into FPGA-compatible firmware, tailored to specific needs.

As the project is actively evolving the hls4ml team is always looking for people trying there tools.

#### Current tools supported

| ML framework/HLS backend | (Q)Keras  | PyTorch | (Q)ONNX        | Vivado HLS | Intel HLS | Vitis HLS    |
|--------------------------|-----------|---------|----------------|------------|-----------|--------------|
| MLP                      | supported | limited | in development | supported  | supported | experimental |
| CNN                      | supported | limited | in development | supported  | supported | experimental |
| RNN (LSTM)               | supported | N/A     | in development | supported  | supported | N/A          |
| GNN (GarNet)             | supported | N/A     | N/A            | N/A        | N/A       | N/A          |


### Compile an example model

```python
import hls4ml

# Fetch a keras model from our example repository
# This will download our example model to your working directory and return an example configuration file
config = hls4ml.utils.fetch_example_model('KERAS_3layer.json')

# You can print the configuration to see some default parameters
print(config)

# Convert it to a hls project
hls_model = hls4ml.converters.keras_to_hls(config)

# Print full list of example models if you want to explore more
hls4ml.utils.fetch_example_list()

# Use Vivado HLS to synthesize the model
# This might take several minutes
hls_model.build()

# Print out the report if you want
hls4ml.report.read_vivado_report('my-hls-test')

```

## 2. Conifer

<img src="https://github.com/thesps/conifer/raw/master/conifer_v1.png" width="250" alt="conifer">

Conifer converts from popular BDT training frameworks, and can emit code projects in different FPGA languages.

Available converters:

- scikit-learn
- xgboost
- ONNX - giving access to other training libraries such as lightGBM and CatBoost with ONNXMLTools
- TMVA
- Tensorflow Decision Forest (tf_df)

Available backends:

- Xilinx HLS - for best results use latest Vitis HLS, but Vivado HLS is also supported (conifer uses whichever is on your `$PATH`)
- VHDL - a direct-to-VHDL implementation, deeply pipelined for high clock frequencies
- FPU - Forest Processing Unit reusable IP core for flexible BDT inference
- C++ - intended for bit-accurate emulation on CPU with a single include header file
- Python - intended for validation of model conversion and to allow inspection of a model without a configuration

### Usage

```python
from sklearn.ensemble import GradientBoostingClassifier
# Train a BDT
clf = GradientBoostingClassifier().fit(X_train, y_train)

# Create a conifer config dictionary
cfg = conifer.backends.xilinxhls.auto_config()
# Change the bit precision (print the config to see everything modifiable)
cfg['Precision'] = 'ap_fixed<12,4>' 

# Convert the sklearn model to a conifer model
model = conifer.converters.convert_from_sklearn(clf, cfg)
# Write the HLS project and compile the C++-Python bridge                      
model.compile()

# Run bit-accurate prediction on the CPU
y_hls = model.decision_function(X)
y_skl = clf.decision_function(X)

# Synthesize the model for the target FPGA
model.build()
```

## 3. (Q)ONNX

[![ReadTheDocs](https://readthedocs.org/projects/qonnx/badge/?version=latest&style=plastic)](http://qonnx.readthedocs.io/)
[![PyPI version](https://badge.fury.io/py/qonnx.svg)](https://badge.fury.io/py/qonnx)
[![arxiv](https://img.shields.io/badge/arXiv-2206.07527-b31b1b.svg)](https://arxiv.org/abs/2206.07527)

<img align="left" src="https://xilinx.github.io/finn/img/TFC_1W2A.onnx.png" alt="QONNX example" style="margin-right: 20px" width="200"/>


QONNX (Quantized ONNX) introduces three new custom operators -- [`Quant`](docs/qonnx-custom-ops/quant_op.md), [`BipolarQuant`](docs/qonnx-custom-ops/bipolar_quant_op.md), and [`Trunc`](docs/qonnx-custom-ops/trunc_op.md) -- in order to represent arbitrary-precision uniform quantization in ONNX. This enables:

* Representation of binary, ternary, 3-bit, 4-bit, 6-bit or any other quantization.
* Quantization is an operator itself, and can be applied to any parameter or layer input.
* Flexible choices for scaling factor and zero-point granularity.
* Quantized values are carried using standard `float` datatypes to remain ONNX protobuf-compatible.

This repository contains a set of Python utilities to work with QONNX models, including but not limited to:

* executing QONNX models for (slow) functional verification
* shape inference, constant folding and other basic optimizations
* summarizing the inference cost of a QONNX model in terms of mixed-precision MACs, parameter and activation volume
* Python infrastructure for writing transformations and defining executable, shape-inferencable custom ops
* (experimental) data layout conversion from standard ONNX NCHW to custom QONNX NHWC ops

## 4. High Granularity Quantization (HGQ)

[![docu](https://github.com/calad0i/HGQ/actions/workflows/sphinx-build.yml/badge.svg)](https://calad0i.github.io/HGQ/)
[![pypi](https://badge.fury.io/py/hgq.svg)](https://badge.fury.io/py/hgq)
[![arxiv](https://img.shields.io/badge/arXiv-2405.00645-b31b1b.svg)](https://arxiv.org/abs/2405.00645)


[High Granularity Quantization (HGQ)](https://github.com/calad0i/HGQ/) is a library that performs gradient-based automatic bitwidth optimization and quantization-aware training algorithm for neural networks to be deployed on FPGAs. By laveraging gradients, it allows for bitwidth optimization at arbitrary granularity, up to per-weight and per-activation level.

![image](https://calad0i.github.io/HGQ/_images/overview.svg)

Conversion of models made with HGQ library is fully supported. The HGQ models are first converted to proxy model format, which can then be parsed by hls4ml bit-accurately. Below is an example of how to create a model with HGQ and convert it to hls4ml model.

```python
   import keras
   from HGQ.layers import HDense, HDenseBatchNorm, HQuantize
   from HGQ import ResetMinMax, FreeBOPs

   model = keras.models.Sequential([
      HQuantize(beta=1.e-5),
      HDenseBatchNorm(32, beta=1.e-5, activation='relu'),
      HDenseBatchNorm(32, beta=1.e-5, activation='relu'),
      HDense(10, beta=1.e-5),
   ])

    opt = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    callbacks = [ResetMinMax(), FreeBOPs()]

    model.fit(..., callbacks=callbacks)

    from HGQ import trace_minmax, to_proxy_model
    from hls4ml.converters import convert_from_keras_model

    trace_minmax(model, x_train, cover_factor=1.0)
    proxy = to_proxy_model(model, aggressive=True)

    model_hls = convert_from_keras_model(
        proxy,
        backend='vivado',
        output_dir=...,
        part=...
    )
```

An interactive example of HGQ can be found in the [kaggle notebook](https://www.kaggle.com/code/calad0i/small-jet-tagger-with-hgq-1). Full documentation can be found at [calad0i.github.io/HGQ](https://calad0i.github.io/HGQ/>).
