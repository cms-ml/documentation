# Direct inference with ONNX Runtime

[ONNX](https://onnx.ai) is an open format built to represent machine learning models. It is designed to improve ==interoperability== across a variety of frameworks and platforms in the AI tools community—most deep learning frameworks (e.g. XGBoost, TensorFlow, PyTorch which are frequently used in CMS) support converting their model into the ONNX format or loading a model from an ONNX format.

<figure>
<img src="../images/inference/onnx/onnx-interoperability.jpeg"  width="70%"/>
<figcaption>The figure showing the ONNX interoperability. (Source from <a href="https://towardsdatascience.com/onnx-preventing-framework-lock-in-9a798fb34c92">website</a>.)</figcaption>
</figure>

[ONNX Runtime](https://onnxruntime.ai/) is a tool aiming for the ==acceleration of machine learning inferencing== across a variety of deployment platforms. It allows to "run any ONNX model using a single set of inference APIs that provide access to the best hardware acceleration available". It includes "built-in optimization features that trim and consolidate nodes without impacting model accuracy."

The CMSSW interface to ONNX Runtime is avaiable since CMSSW\_11\_1\_X ([cmssw#28112](https://github.com/cms-sw/cmssw/pull/28112), [cmsdist#5020](https://github.com/cms-sw/cmsdist/pull/5020)). Its functionality is improved in CMSSW\_11\_2\_X.
The final implementation is also backported to CMSSW\_10\_6\_X to facilitate Run 2 UL data reprocessing. The inference of a number of deep learning tagger models (e.g. DeepJet, DeepTauID, ParticleNet, DeepDoubleX, etc.) has been made with ONNX Runtime in the routine of UL processing and has gained substantial speedup.

On this page, we will ==use a simple example to show how to use ONNX Runtime for deep learning model inference in the CMSSW framework==, both in C++ (e.g. to process the MiniAOD file) and in Python (e.g. using NanoAOD-tools to process the NanoAODs). This may help readers who will deploy an ONNX model into their analyses or in the CMSSW framework.


## Software Setup

We use CMSSW\_11\_2\_5\_patch2 to show the simple example for ONNX Runtime inference. The example can also work under the new 12 releases (note that inference with C++ can also run on CMSSW\_10\_6\_X)


```shell linenums="1"
export SCRAM_ARCH="slc7_amd64_gcc900"
export CMSSW_VERSION="CMSSW_11_2_5_patch2"

source /cvmfs/cms.cern.ch/cmsset_default.sh

cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"

cmsenv
scram b
```

## Converting model to ONNX

The model deployed into CMSSW or our analysis needs to be converted to ONNX from the original framework format where it is trained. Please see [here](https://github.com/onnx/tutorials#converting-to-onnx-format) for a nice deck of tutorials on converting models from different mainstream frameworks into ONNX.

Here we take PyTorch as an example. A PyTorch model can be converted by `torch.onnx.export(...)`. As a simple illustration, we convert a randomly initialized feed-forward network implemented in PyTorch, with 10 input nodes and 2 output nodes, and two hidden layers with 64 nodes each. The conversion code is presented below. The output model `model.onnx` will be deployed under the CMSSW framework in our following tutorial.

??? hint "Click to expand"

    ```python linenums="1"
    import torch
    import torch.nn as nn
    torch.manual_seed(42)

    class SimpleMLP(nn.Module):

        def __init__(self, **kwargs):
            super(SimpleMLP, self).__init__(**kwargs)
            self.mlp = nn.Sequential(
                nn.Linear(10, 64), nn.BatchNorm1d(64), nn.ReLU(), 
                nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), 
                nn.Linear(64, 2), nn.ReLU(), 
                )
        def forward(self, x):
            # input x: (batch_size, feature_dim=10)
            x = self.mlp(x)
            return torch.softmax(x, dim=1)

    model = SimpleMLP()

    # create dummy input for the model
    dummy_input = torch.ones(1, 10, requires_grad=True) # batch size = 1

    # export model to ONNX
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=['my_input'], output_names=['my_output'])

    ```
## Inference in CMSSW (C++)

We will introduce how to write a module to run inference on the ONNX model under the CMSSW framework. CMSSW is known for its multi-threaded ability. In a threaded framework, multiple threads are served for processing events in the event loop. The logic is straightforward: a new event is assigned to idled threads following the first-come-first-serve princlple.

In most cases, each thread is able to process events individually as the majority of event processing workflow can be accomplished only by seeing the information of that event. Thus, the `stream` modules (`stream` `EDAnalyzer` and `stream` `EDFilter`) are used frequently as each thread holds an individual copy of the module instance—they do not need to communicate with each other. It is however also possible to share a global cache object between all threads in case sharing information across threads is necessary. In all, such CMSSW EDAnalyzer modules are declared by `#!cpp class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>>` (similar for `EDFilter`). Details can be found in documentation on the [C++ interface of `stream` modules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface).

Let's then think about what would happen when interfacing CMSSW with ONNX for model inference. When ONNX Runtime accepts a model, it converts the model into an in-memory representation, and performance a variety of optimizations depending on the operators in the model. The procedure is done when an ONNX Runtime `Session` is created with an inputting model. The economic method will then be to hold only one `Session` for all threads—this may save memory to a large extent, as the model has only one copy in memory. Upon request from multiple threads to do inference with their input data, the `Session` accepts those requests and serializes them, then produces the output data. ONNX Runtime has by design accepted that multithread threads invoke the `Run()` method on the same inference `Session` object. Therefore, what has left us to do is to

1. create a `Session` as a global object in our CMSSW module and share it among all threads;
2. in each thread, we process the input data and then call the `Run()` method from that global `Session`.

That's the main logic for implementing ONNX inference in CMSSW. For details of high-level designs of ONNX Runtime, please see [documentation here](https://onnxruntime.ai/docs/reference/high-level-design.html).

With this concept, let's build the module.

##### 1. includes

```cpp linenums="1"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
// further framework includes
...
```

We include `stream/EDAnalyzer.h` to build the `stream` CMSSW module.

##### 2. Global cache object

In CMSSW there exists a class [`ONNXRuntime`](https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/ONNXRuntime/src/ONNXRuntime.cc) which can be used directly as the global cache object. Upon initialization from a given model, it holds the ONNX Runtime `Session` object and provides the handle to invoke the `Run()` for model inference.

We put the `ONNXRuntime` class in the `#!cpp edm::GlobalCache` template argument:

```cpp linenums="1" hl_lines="1"
class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<ONNXRuntime>> {
  ...
};
```

##### 3. Initiate objects

In the `stream` `EDAnlyzer` module, it provides a hook `initializeGlobalCache()` to initiate the global object. We simply do

```cpp linenums="1"
std::unique_ptr<ONNXRuntime> MyPlugin::initializeGlobalCache(const edm::ParameterSet &iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}
```

to initiate the `ONNXRuntime` object upon a given model path.

##### 4. Inference

We know the event processing step is implemented in the `void EDAnalyzer::analyze` method. When an event is assigned to a valid thread, the content will be processed in that thread. This can go in parallel with other threads processing other events.

We need to first construct the input data dedicated to the event. Here we create a dummy input: a sequence of consecutive integers of length 10. The input is set by replacing the values of our pre-booked vector, `data_`. This member variable has `#!cpp vector<vector<float>>` format and is initialised as `#!cpp { {0, 0, ..., 0} }` (contains only one element, which is a vector of 10 zeros). In processing of each event, the input `data_` is modified:

```cpp linenums="1"
std::vector<float> &group_data = data_[0];
for (size_t i = 0; i < 10; i++){
  group_data[i] = float(iEvent.id().event() % 100 + i);
}
```

Then, we send `data_` to the inference engine and get the model output:

```cpp linenums="1"
std::vector<float> outputs = globalCache()->run(input_names_, data_, input_shapes_)[0];
```

We clarify a few details here.

First, we use `#!cpp globalCache()` which is a class method in our `stream` CMSSW module to access the global object shared across all threads. In our case it is the `ONNXRuntime` instance.

The `run()` method is a wrapper to call `Run()` on the ONNX `Session`. Definations on the method arguments are (code from [link](https://github.com/cms-sw/cmssw/blob/CMSSW_11_2_5_patch2/PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h#L32-L44)):
```cpp linenums="1"
// Run inference and get outputs
// input_names: list of the names of the input nodes.
// input_values: list of input arrays for each input node. The order of `input_values` must match `input_names`.
// input_shapes: list of `int64_t` arrays specifying the shape of each input node. Can leave empty if the model does not have dynamic axes.
// output_names: names of the output nodes to get outputs from. Empty list means all output nodes.
// batch_size: number of samples in the batch. Each array in `input_values` must have a shape layout of (batch_size, ...).
// Returns: a std::vector<std::vector<float>>, with the order matched to `output_names`.
// When `output_names` is empty, will return all outputs ordered as in `getOutputNames()`.
FloatArrays run(const std::vector<std::string>& input_names,
                FloatArrays& input_values,
                const std::vector<std::vector<int64_t>>& input_shapes = {},
                const std::vector<std::string>& output_names = {},
                int64_t batch_size = 1) const;
```
where we have
```cpp linenums="1"
typedef std::vector<std::vector<float>> FloatArrays;
```

In our case, `#!cpp input_names` is set to `#!cpp {"my_input"}` which corresponds to the names upon model creation. `#!cpp input_values` is a length-1 vector, and `#!cpp input_values[0]` is a vector of float of length 10, which are inputs to the 10 nodes. `#!cpp input_shapes` can be set empty here and will be necessary for advanced usage, when our input has dynamic lengths (e.g., in boosed jet tagging, we use different numbers of particle-flow candidates and secondary vertices as input).

For the usual model design, we have only one vector of output. In such a case, the output is simply a length-1 vector, and we use `#!cpp [0]` to get the vector of two float numbers—the output of the model.

##### Full example

Let's construct the full example.

??? hint "Click to expand"

    The example assumes the following directory structure:

    ```
    MySubsystem/MyModule/
    │
    ├── plugins/
    │   ├── MyPlugin.cpp
    │   └── BuildFile.xml
    │
    ├── test/
    │   └── my_plugin_cfg.py
    │
    └── data/
        └── model.onnx
    ```

    === "plugins/MyPlugin.cpp"

        ```cpp linenums="1" hl_lines="2"
        --8<-- "content/inference/code/onnx/ort_plugin.cpp"
        ```

    === "plugins/BuildFile.xml"

        ```xml linenums="1"
        --8<-- "content/inference/code/onnx/ort_buildfile.xml"
        ```

    === "test/my_plugin_cfg.py"

        ```python linenums="1"
        --8<-- "content/inference/code/onnx/ort_cfg.py"
        ```

    === "data/model.onnx"

        The model is produced by code in the section ["Converting model to ONNX"](#converting-model-to-onnx) and can be downloaded [here](https://github.com/cms-ml/documentation/raw/master/content/inference/code/onnx/model.onnx).

##### Test our module

Under `MySubsystem/MyModule/test`, run `#!bash cmsRun my_plugin_cfg.py` to launch our module. You may see the following from the output, which include the input and output vectors in the inference process.

??? hint "Click to see the output"
    ```
    ...
    19-Jul-2022 10:50:41 CEST  Successfully opened file root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/4C8619B2-D0C0-4647-B946-B33754F4ED16.root
    Begin processing the 1st record. Run 1, Event 27074045, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.494 CEST
    input data -> 45 46 47 48 49 50 51 52 53 54
    output data -> 0.995657 0.00434343
    Begin processing the 2nd record. Run 1, Event 27074048, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.495 CEST
    input data -> 48 49 50 51 52 53 54 55 56 57
    output data -> 0.996884 0.00311563
    Begin processing the 3rd record. Run 1, Event 27074059, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.495 CEST
    input data -> 59 60 61 62 63 64 65 66 67 68
    output data -> 0.999081 0.000919373
    Begin processing the 4th record. Run 1, Event 27074061, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.495 CEST
    input data -> 61 62 63 64 65 66 67 68 69 70
    output data -> 0.999264 0.000736247
    Begin processing the 5th record. Run 1, Event 27074046, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 46 47 48 49 50 51 52 53 54 55
    output data -> 0.996112 0.00388828
    Begin processing the 6th record. Run 1, Event 27074047, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 47 48 49 50 51 52 53 54 55 56
    output data -> 0.996519 0.00348065
    Begin processing the 7th record. Run 1, Event 27074064, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 64 65 66 67 68 69 70 71 72 73
    output data -> 0.999472 0.000527586
    Begin processing the 8th record. Run 1, Event 27074074, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 74 75 76 77 78 79 80 81 82 83
    output data -> 0.999826 0.000173664
    Begin processing the 9th record. Run 1, Event 27074050, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 50 51 52 53 54 55 56 57 58 59
    output data -> 0.997504 0.00249614
    Begin processing the 10th record. Run 1, Event 27074060, LumiSection 10021 on stream 0 at 19-Jul-2022 10:50:43.496 CEST
    input data -> 60 61 62 63 64 65 66 67 68 69
    output data -> 0.999177 0.000822734
    19-Jul-2022 10:50:43 CEST  Closed file root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/4C8619B2-D0C0-4647-B946-B33754F4ED16.root
    ```

Also we could try launching the script with more threads. Change the corresponding line in `my_plugin_cfg.py` as follows to activate the multi-threaded mode with 4 threads.

```python linenums="31"
process.options.numberOfThreads=cms.untracked.uint32(4)
```

Launch the script again, and one could see the same results, but with the inference processed concurrently on 4 threads.


## Inference in CMSSW (Python)

Doing ONNX Runtime inference with python is possible as well. For those releases that have the ONNX Runtime C++ package installed, the `onnxruntime` python package is also installed in ==`python3`== (except for CMSSW\_10\_6\_X). We still use CMSSW\_11\_2\_5\_patch2 to run our examples. We could quickly check if `onnxruntime` is available by:

```python linenums="1"
python3 -c "import onnxruntime; print('onnxruntime available')"
```

The python code is simple to construct: following the quick examples ["Get started with ORT for Python"](https://onnxruntime.ai/docs/get-started/with-python.html), we create the file `MySubsystem/MyModule/test/my_standalone_test.py` as follows:

```python linenums="1"
import onnxruntime as ort
import numpy as np

# create input data in the float format (32 bit)
data = np.arange(45, 55).astype(np.float32)

# create inference session using ort.InferenceSession from a given model
ort_sess = ort.InferenceSession('../data/model.onnx')

# run inference
outputs = ort_sess.run(None, {'my_input': np.array([data])})[0]

# print input and output
print('input ->', data)
print('output ->', outputs)
```

Under the directory `MySubsystem/MyModule/test`, run the example with `python3 my_standalone_test.py`. Then we see the output:

```
input -> [45. 46. 47. 48. 49. 50. 51. 52. 53. 54.]
output -> [[0.9956566  0.00434343]]
```

Using ONNX Runtime on NanoAOD-tools follows the same logic. Here we create the ONNX `Session` in the beginning stage and run inference in the event loop. Note that NanoAOD-tools runs the event loop in the single-thread mode.

Please find details in the following block.

??? hint "Click to see the NanoAOD-tools example"

    We run the NanoAOD-tools example following the above CMSSW\_11\_2\_5\_patch2 environment. According to the setup instruction in [NanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools), do

    ```bash
    cd $CMSSW_BASE/src
    git clone https://github.com/cms-nanoAOD/nanoAOD-tools.git PhysicsTools/NanoAODTools
    cd PhysicsTools/NanoAODTools
    cmsenv
    scram b
    ```

    Now we add our custom module to run ONNX Runtime inference. Create a file `PhysicsTools/NanoAODTools/python/postprocessing/examples/exampleOrtModule.py` with the content:

    ```python linenums="1"  hl_lines="16 34"
    --8<-- "content/inference/code/onnx/ort_nanoaod_tools_module.py"
    ```

    Please notice the highlighted lines for the creation of ONNX Runtime `Session` and launching the inference.
    
    Finally, following the test command from NanoAOD-tools, we run our custom module in `python3` by
    ```bash
    python3 scripts/nano_postproc.py outDir /eos/cms/store/user/andrey/f.root -I PhysicsTools.NanoAODTools.postprocessing.examples.exampleOrtModule exampleOrtModuleConstr -N 10
    ```

    We should see the output as follows
    ```
    processing.examples.exampleOrtModule exampleOrtModuleConstr -N 10
    Loading exampleOrtModuleConstr from PhysicsTools.NanoAODTools.postprocessing.examples.exampleOrtModule
    Will write selected trees to outDir
    Pre-select 10 entries out of 10 (100.00%)
    input -> [11. 12. 13. 14. 15. 16. 17. 18. 19. 20.]
    output -> [[0.83919346 0.16080655]]
    input -> [ 7.  8.  9. 10. 11. 12. 13. 14. 15. 16.]
    output -> [[0.76994413 0.2300559 ]]
    input -> [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]
    output -> [[0.7116992 0.2883008]]
    input -> [ 2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
    output -> [[0.66414535 0.33585465]]
    input -> [ 9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]
    output -> [[0.80617136 0.19382869]]
    input -> [ 6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]
    output -> [[0.75187963 0.2481204 ]]
    input -> [16. 17. 18. 19. 20. 21. 22. 23. 24. 25.]
    output -> [[0.9014619  0.09853811]]
    input -> [18. 19. 20. 21. 22. 23. 24. 25. 26. 27.]
    output -> [[0.9202239  0.07977609]]
    input -> [ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]
    output -> [[0.7330253  0.26697478]]
    input -> [10. 11. 12. 13. 14. 15. 16. 17. 18. 19.]
    output -> [[0.82333535 0.17666471]]
    Processed 10 preselected entries from /eos/cms/store/user/andrey/f.root (10 entries). Finally selected 10 entries
    Done outDir/f_Skim.root
    Total time 1.1 sec. to process 10 events. Rate = 9.3 Hz.
    ```


## Links and further reading

- ONNX/ONNX Runtime
    - [Tutorials on converting models to ONNX format](https://github.com/onnx/tutorials#converting-to-onnx-format)
    - [ONNX Runtime C++ example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx)
    - [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/index.html)
    - [ONNX Runtime python example](https://onnxruntime.ai/docs/get-started/with-python.html)
    - [ONNX Runtime python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
    - [ONNX Runtime in CMSSW (talk)](https://indico.cern.ch/event/1127774/contributions/4733524/attachments/2394910/4094695/ONNXRuntime_20220221.pdf)

---

Developers: [Huilin Qu](mailto:huilin.qu@cern.ch)

Authors: [Congqiao Li](mailto:congqiao.li@cern.ch)
