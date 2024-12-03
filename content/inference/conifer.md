# Direct inference with conifer

<p align="center">
  <img src="https://raw.githubusercontent.com/thesps/conifer/master/conifer_v1.png" alt="drawing" width="300" />
</p>

## Introduction

[conifer](https://ssummers.web.cern.ch/conifer/) is a Python package developed by the [Fast Machine Learning Lab](https://fastmachinelearning.org/) for the deployment of Boosted Decision Trees in FPGAs for Level 1 Trigger applications. Documentation, examples, and tutorials are available from the conifer [website](https://ssummers.web.cern.ch/conifer/), [GitHub](https://github.com/thesps/conifer), and the [hls4ml tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial/blob/master/part5_bdt.ipynb) respectively. conifer is on the Python Package Index and can be installed like `pip install conifer`. Targeting FPGAs requires Xilinx's [Vivado/Vitis](https://www.xilinx.com/products/design-tools/vivado.html) suite of software. Here's a brief summary of features:

- conversion from common BDT training frameworks: scikit-learn, XGBoost, Tensorflow Decision Forests (TF DF), TMVA, and ONNX
- conversion to FPGA firmware with backends: HLS (C++ for FPGA), VHDL, C++ (for CPU)
- utilities for bit- and cycle-accurate firmware simulation, and interface to FPGA synthesis tools for evaluation and deployment from Python

## Emulation in CMSSW

All L1T algorithms require bit-exact emulation for performance studies and validation of the hardware system. For conifer this is provided with a single header file at `L1Trigger/Phase2L1ParticleFlow/interface/conifer.h`. The user must also provide the BDT JSON file exported from the conifer Python tool for their model. JSON loading in CMSSW uses the `nlohmann/json` external.

Both the conifer FPGA firmware and C++ emulation use Xilinx's arbitrary precision types for fixed-point arithmetic (`hls` external of CMSSW). This is cheaper and faster in the FPGA fabric than floating-point types. An important part of the model preparation process is choosing the proper fixed-point data types to avoid loss of performance compared to the trained model. Input preprocessing, in particular scaling, can help constrain the input variables to a smaller numerical range, but may also have a hardware cost to implement. In C++ the arbitrary precision types are specified like: `ap_fixed<width, integer, rounding mode, saturation mode>`. 

Minimal preparation from Python:
```python
import conifer
model = conifer. ... # convert or load a conifer model
# e.g. model = conifer.converters.convert_from_xgboost(xgboost_model)
model.save('my_bdt.json')
```

CMSSW C++ user code:
```c++
// include the conifer emulation header file
#include "L1Trigger/Phase2L1ParticleFlow/interface/conifer.h"

... model setup
// define the input/threshold and score types
// important: this needs to match the firmware settings for bit-exactness!
// note: can use native types like float/double for development/debugging
typedef ap_fixed<18,8> input_t;
typedef ap_fixed<12,3,AP_RND_CONV,AP_SAT> score_t;

// create a conifer BDT instance
// 'true' to use balanced add-tree score aggregation (needed for bit-exactness)
bdt = conifer::BDT<input_t, score_t, true>("my_bdt.json");

... inference
// prepare the inputs, vector length same as model n_features
std::vector<input_t> inputs = ... 
// run inference, scores vector length same as model n_classes (or 1 for binary classification/regression)
std::vector<score_t> scores = bdt.decision_function(inputs);
```

conifer does not compute class probabilities from the raw predictions for the avoidance of extra resource and latency cost in the L1T deployment. Cuts or working points should therefore be applied on the raw predictions.