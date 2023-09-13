# Direct inference with TensorFlow 2
---

TensorFlow 2 is available since CMSSW\_11\_1\_X ([cmssw#28711](https://github.com/cms-sw/cmssw/pull/28711), [cmsdist#5525](https://github.com/cms-sw/cmsdist/pull/5525)).
The integration into the software stack can be found in [cmsdist/tensorflow.spec](https://github.com/cms-sw/cmsdist/blob/latest/tensorflow.spec) and the interface is located in [cmssw/PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow).


## Available versions

=== "Python 3 on el8"

    | TensorFlow | el8\_amd64\_gcc10 | el8\_amd64\_gcc11 |
    | :--------: | ----------------- | ----------------- |
    |   v2.6.0   | ≥ CMSSW\_12\_3\_4 | -                 |
    |   v2.6.4   | ≥ CMSSW\_12\_5\_0 | ≥ CMSSW\_12\_5\_0 |

=== "Python 3 on slc7"

    | TensorFlow | slc7\_amd64\_gcc900 | slc7\_amd64\_gcc10 | slc7\_amd64\_gcc11 |
    | :--------: | :------------------ | ------------------ | ------------------ |
    |   v2.1.0   | ≥ CMSSW\_11\_1\_0   | -                  | -                  |
    |   v2.3.1   | ≥ CMSSW\_11\_2\_0   | -                  | -                  |
    |   v2.4.1   | ≥ CMSSW\_11\_3\_0   | -                  | -                  |
    |   v2.5.0   | ≥ CMSSW\_12\_0\_0   | ≥ CMSSW\_12\_0\_0  | -                  |
    |   v2.6.0   | ≥ CMSSW\_12\_1\_0   | ≥ CMSSW\_12\_1\_0  | ≥ CMSSW\_12\_3\_0  |
    |   v2.6.4   | -                   | ≥ CMSSW\_12\_5\_0  | ≥ CMSSW\_13\_0\_0  |

=== "Python 2 on slc7"

    | TensorFlow | slc7\_amd64\_gcc900 |
    | :--------: | ------------------- |
    |   v2.1.0   | ≥ CMSSW\_11\_1\_0   |
    |   v2.3.1   | ≥ CMSSW\_11\_2\_0   |


At this time, only CPU support is provided.
While GPU support is generally possible, it is currently disabled due to some interference with production workflows but will be enabled once they are resolved.


## Software setup

To run the examples shown below, create a mininmal inference setup with the following snippet.
==Adapt== the `SCRAM_ARCH` according to your operating system and desired compiler.

```shell linenums="1"
export SCRAM_ARCH="el8_amd64_gcc11"
export CMSSW_VERSION="CMSSW_12_6_0"

source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""

cmsrel "${CMSSW_VERSION}"
cd "${CMSSW_VERSION}/src"

cmsenv
scram b
```

Below, the [`cmsml`](https://github.com/cms-ml/cmsml) Python package is used to convert models from TensorFlow objects (`tf.function`'s or Keras models) to protobuf graph files ([documentation](https://cmsml.readthedocs.io)).
It should be available after executing the commands above.
You can check its version via

```shell
python -c "import cmsml; print(cmsml.__version__)"
```

and compare to the [released tags](https://github.com/cms-ml/cmsml/tags).
If you want to install a newer version from either the [master branch of the cmsml repository](https://github.com/cms-ml/cmsml) or the [Python package index (PyPI)](https://pypi.org/project/cmsml), you can simply do that via pip.

=== "master"

    ```shell
    # into your user directory (usually ~/.local)
    pip install --upgrade --user git+https://github.com/cms-ml/cmsml

    # _or_

    # into a custom directory
    pip install --upgrade --prefix "CUSTOM_DIRECTORY" git+https://github.com/cms-ml/cmsml
    ```

=== "PyPI"

    ```shell
    # into your user directory (usually ~/.local)
    pip install --upgrade --user cmsml

    # _or_

    # into a custom directory
    pip install --upgrade --prefix "CUSTOM_DIRECTORY" cmsml
    ```


## Saving your model

After successfully training, you should save your model in a protobuf graph file which can be read by the interface in CMSSW.
Naturally, you only want to save that part of your model that is required to run the network prediction, i.e., it should ==not== contain operations related to model training or loss functions (unless explicitely required).
Also, to reduce the memory footprint and to accelerate the inference, variables should be converted to constant tensors.
Both of these model transformations are provided by the `cmsml` package.

Instructions on how to transform and save your model are shown below, depending on whether you use Keras or plain TensorFlow with `tf.function`'s.

=== "Keras"

    The code below saves a Keras `Model` instance as a protobuf graph file using [`cmsml.tensorflow.save_graph`](https://cmsml.readthedocs.io/en/latest/api/tensorflow.html#cmsml.tensorflow.save_graph).
    In order for Keras to built the internal graph representation before saving, make sure to either compile the model, or pass an `input_shape` to the first layer:

    ```python linenums="1" hl_lines="18"
    # coding: utf-8

    import tensorflow as tf
    import tf.keras.layers as layers
    import cmsml

    # define your model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(10,), name="input"))
    model.add(layers.Dense(100, activation="tanh"))
    model.add(layers.Dense(3, activation="softmax", name="output"))

    # train it
    ...

    # convert to binary (.pb extension) protobuf
    # with variables converted to constants
    cmsml.tensorflow.save_graph("graph.pb", model, variables_to_constants=True)
    ```

    Following the Keras naming conventions for certain layers, the input will be named `"input"` while the output is named `"sequential/output/Softmax"`.
    To cross check the names, you can save the graph in text format by using the extension `".pb.txt"`.

=== "tf.function"

    Let's consider you write your network model in a single `tf.function`.

    ```python linenums="1"
    # coding: utf-8

    import tensorflow as tf
    import cmsml

    # define the model
    @tf.function
    def model(x):
        # lift variable initialization to the lowest context so they are
        # not re-initialized on every call (eager calls or signature tracing)
        with tf.init_scope():
            W = tf.Variable(tf.ones([10, 1]))
            b = tf.Variable(tf.ones([1]))

        # define your "complex" model here
        h = tf.add(tf.matmul(x, W), b)
        y = tf.tanh(h, name="y")

        return y
    ```

    In TensorFlow terms, the `model` function is ==polymorphic== - it accepts different types of the input tensor `x` (`tf.float32`, `tf.float64`, ...).
    For each type, TensorFlow will create a ==concrete== function with an associated ``tf.Graph`` object.
    This mechanism is referred to as ==signature tracing==.
    For deeper insights into `tf.function`, the concepts of signature tracing, polymorphic and concrete functions, see the guide on [Better performance with `tf.function`](https://www.tensorflow.org/guide/function).

    To save the model as a protobuf graph file, you explicitely need to create a concrete function.
    However, this is fairly easy once you know the exact type and shape of all input arguments.

    ```python linenums="20" hl_lines="7"
    # create a concrete function
    cmodel = model.get_concrete_function(
        tf.TensorSpec(shape=[2, 10], dtype=tf.float32),
    )

    # convert to binary (.pb extension) protobuf
    # with variables converted to constants
    cmsml.tensorflow.save_graph("graph.pb", cmodel, variables_to_constants=True)
    ```

    The input will be named `"x"` while the output is named `"y"`.
    To cross check the names, you can save the graph in text format by using the extension `".pb.txt"`.

    ??? hint "Different method: Frozen signatures"
        Instead of creating a polymorphic `tf.function` and extracting a concrete one in a second step, you can directly define an input signature upon definition.

        ```python
        @tf.function(input_signature=(tf.TensorSpec(shape=[2, 10], dtype=tf.float32),))
        def model(x):
            ...
        ```

        This disables signature tracing since the input signature is frozen.
        However, you can directly pass it to [`cmsml.tensorflow.save_graph`](https://cmsml.readthedocs.io/en/latest/api/tensorflow.html#cmsml.tensorflow.save_graph).


## Inference in CMSSW

The inference can be implemented to run in a [single thread](#single-threaded-inference).
In general, this does not mean that the module cannot be executed with multiple threads (`#!shell cmsRun --numThreads <N> <CFG_FILE>`), but rather that its performance in terms of evaluation time and especially memory consumption is likely to be suboptimal.
Therefore, for modules to be integrated into CMSSW, ==the [multi-threaded implementation](#multi-threaded-inference) is strongly recommended==.


### CMSSW module setup

If you aim to use the TensorFlow interface in a CMSSW ==plugin==, make sure to include

```xml linenums="1"
<use name="PhysicsTools/TensorFlow" />

<flags EDM_PLUGIN="1" />
```

in your `plugins/BuildFile.xml` file.
If you are using the interface inside the `src/` or `interface/` directory of your module, make sure to create a global `BuildFile.xml` file next to theses directories, containing (at least):

```xml linenums="1"
<use name="PhysicsTools/TensorFlow" />

<export>
    <lib name="1" />
</export>
```


### Single-threaded inference

Despite `tf.Session` being removed in the Python interface as of TensorFlow 2, the concepts of

- `Graph`'s, containing the *constant* computational structure and trained variables of your model,
- `Session`'s, handling execution and data exchange, and
- the separation between them

live on in the C++ interface.
Thus, the overall inference approach is **1)** include the interface, **2)** initialize `Graph` and `session`, **3)** per event create input tensors and run the inference, and **4)** cleanup.


##### 1. Includes

```cpp linenums="1"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
// further framework includes
...
```


##### 2. Initialize objects

```cpp linenums="1"
// configure logging to show warnings (see table below)
tensorflow::setLogging("2");

// load the graph definition
tensorflow::GraphDef* graphDef = tensorflow::loadGraphDef("/path/to/constantgraph.pb");

// create a session
tensorflow::Session* session = tensorflow::createSession(graphDef);
```


##### 3. Inference

```cpp linenums="1"
// create an input tensor
// (example: single batch of 10 values)
tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 });


// fill the tensor with your input data
// (example: just fill consecutive values)
for (size_t i = 0; i < 10; i++) {
    input.matrix<float>()(0, i) = float(i);
}

// run the evaluation
std::vector<tensorflow::Tensor> outputs;
tensorflow::run(session, { { "input", input } }, { "output" }, &outputs);

// process the output tensor
// (example: print the 5th value of the 0th (the only) example)
std::cout << outputs[0].matrix<float>()(0, 5) << std::endl;
// -> float
```


##### 4. Cleanup

```cpp linenums="1"
tensorflow::closeSession(session);
delete graphDef;
```

##### Full example

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
        └── graph.pb
    ```

    === "plugins/MyPlugin.cpp"

        ```cpp linenums="1" hl_lines="2"
        --8<-- "content/inference/code/tensorflow/tensorflow2_st_plugin.cpp"
        ```

    === "plugins/BuildFile.xml"

        ```xml linenums="1"
        --8<-- "content/inference/code/tensorflow/tensorflow2_buildfile.xml"
        ```

    === "test/my_plugin_cfg.py"

        ```python linenums="1"
        --8<-- "content/inference/code/tensorflow/tensorflow2_cfg.py"
        ```


### Multi-threaded inference

Compared to the single-threaded implementation [above](#single-threaded-inference), the multi-threaded version has one major difference: ==both the `Graph` and the `Session` are no longer members of a particular module instance, but rather shared between all instances in all threads==.
See the documentation on the [C++ interface of `stream` modules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface) for details.

!!! danger "Recommendation updated"

    The previous recommendation stated that the `Session` is not constant and thus, should not be placed in the global cache, but rather created once per stream module instance.
    However, it was discovered that, although not explicitely declared as constant in the `tensorflow::run()` / `Session::run()` interface, the session is actually not changed during evaluation and can be treated as being effectively constant.

    As a result, it is safe to move it to the global cache, next to the `Graph` object.
    The TensorFlow interface in CMSSW was adjusted in order to accept `const` objects in [cmssw#40161](https://github.com/cms-sw/cmssw/pull/40161).

Thus, the overall inference approach is **1)** include the interface, **2)** let your plugin inherit from `#!c++ edm::stream::EDAnalyzerasdasd` and declare the `GlobalCache`, **3)** store in c`#!cpp const Session*`, pointing to the cached session, and **4)** per event create input tensors and run the inference.


##### 1. Includes

```cpp linenums="1" hl_lines="2"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
// further framework includes
...
```

Note that `stream/EDAnalyzer.h` is included rather than `one/EDAnalyzer.h`.


##### 2. Define and use the global cache

The cache definition is done by declaring a simple struct.
However, for the purpose of just storing a graph and a session object, a so-called [`tensorflow::SessionCache`](https://github.com/cms-sw/cmssw/blob/cf78146456e111d5943da222cfac22e57f3c0355/PhysicsTools/TensorFlow/interface/TensorFlow.h#L184-L215) struct is already provided centrally.
It was added in [cmssw#40284](https://github.com/cms-sw/cmssw/pull/40284) and its usage is shown in the following.
In case the `tensorflow::SessionCache` is not (yet) available in your version of CMSSW, expand the "Custom cache struct" section below.

Use it in the `#!cpp edm::GlobalCache` template argument and adjust the plugin accordingly.

```cpp linenums="1"
class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<tensorflow::SessionCache>> {
public:
    explicit GraphLoadingMT(const edm::ParameterSet&, const tensorflow::SessionCache*);
    ~GraphLoadingMT();

    // an additional static method for initializing the global cache
    static std::unique_ptr<tensorflow::SessionCache> initializeGlobalCache(const edm::ParameterSet&);
    static void globalEndJob(const CacheData*);
...
```

Implement `initializeGlobalCache` to control the behavior of how the cache object is created.
The destructor of `tensorflow::SessionCache` already handles the closing of the session itself and the deletion of all objects.

```cpp
std::unique_ptr<tensorflow::SessionCache> MyPlugin::initializeGlobalCache(const edm::ParameterSet& config) {
  std::string graphPath = edm::FileInPath(params.getParameter<std::string>("graphPath")).fullPath();
  return std::make_unique<tensorflow::SessionCache>(graphPath);
}
```

??? hint "Custom cache struct"

    ```cpp linenums="1"
    struct MyCache {
      MyCache() : {
      }

      std::atomic<tensorflow::GraphDef*> graph;
      std::atomic<tensorflow::Session*> session;
    };
    ```

    Use it in the `#!cpp edm::GlobalCache` template argument and adjust the plugin accordingly.

    ```cpp linenums="1"
    class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
    public:
        explicit GraphLoadingMT(const edm::ParameterSet&, const CacheData*);
        ~GraphLoadingMT();

        // two additional static methods for handling the global cache
        static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
        static void globalEndJob(const CacheData*);
    ...
    ```

    Implement `initializeGlobalCache` and `globalEndJob` to control the behavior of how the cache object is created and destroyed.

See the [full example](#full-example_1) below for more details.


##### 3. Initialize objects

In your module constructor, you can get a pointer to the constant session to perform model evaluation during the event loop.

```cpp linenums="1"
// declaration in header
const tensorflow::Session* _session;

// get a pointer to the const session stored in the cache in the constructor init
MyPlugin::MyPlugin(const edm::ParameterSet& config,  const tensorflow::SessionCache* cache)
    : session_(cache->getSession()) {
  ...
}
```


##### 4. Inference

```cpp linenums="1"
// create an input tensor
// (example: single batch of 10 values)
tensorflow::Tensor input(tensorflow::DT_FLOAT, { 1, 10 });


// fill the tensor with your input data
// (example: just fill consecutive values)
for (size_t i = 0; i < 10; i++) {
    input.matrix<float>()(0, i) = float(i);
}

// define the output
std::vector<tensorflow::Tensor> outputs;

// evaluate
// note: in case this line causes the compiler to complain about the const'ness of the session_ in
//       this call, your CMSSW version might not yet support passing a const session, so in this
//       case, pass "const_cast<tensorflow::Session*>(session_)"
tensorflow::run(session_, { { inputTensorName, input } }, { outputTensorName }, &outputs);

// process the output tensor
// (example: print the 5th value of the 0th (the only) example)
std::cout << outputs[0].matrix<float>()(0, 5) << std::endl;
// -> float
```

!!! danger "Note"

    If the TensorFlow interface in your CMSSW release does not yet accept `#!cpp const` sessions, line 19 in the example above will cause an error during compilation.
    In this case, replace `#!cpp session_` in that line to

    ```cpp
    const_cast<tensorflow::Session*>(session_)
    ```


##### Full example

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
        └── graph.pb
    ```

    === "plugins/MyPlugin.cpp"

        ```cpp linenums="1" hl_lines="2"
        --8<-- "content/inference/code/tensorflow/tensorflow2_mt_plugin.cpp"
        ```

    === "plugins/BuildFile.xml"

        ```xml linenums="1"
        --8<-- "content/inference/code/tensorflow/tensorflow2_buildfile.xml"
        ```

    === "test/my_plugin_cfg.py"

        ```python linenums="1"
        --8<-- "content/inference/code/tensorflow/tensorflow2_cfg.py"
        ```


### GPU backend

By default the TensorFlow sessions get created for CPU running. Since CMSSW_13_1_X the GPU backend for TensorFlow is
available in the cmssw release. 

Minimal changes are needed in the inference code to move the model on the GPU. 
A `tensorflow::Options` struct is available to setup the backend. 

```cpp linenums="1"
tensorflow::Options options { tensorflow::Backend::cuda};

# Initialize the cache
tensorflow::SessionCache cache(pbFile, options);
# or a single session
const tensorflow::Session* session = tensorflow::createSession(graphDef, options);

``` 

CMSSW modules should add an options in the `PSets` of the producers and analyzers to configure on the fly the
TensorFlow backend for the sessions created by the plugins. 

### Optimization

Depending on the use case, the following approaches can optimize the inference performance.
It could be worth checking them out in your algorithm.

Further optimization approaches can be found in the [integration checklist](checklist.md).


#### Reusing tensors

In some cases, instead of creating new input tensors for each inference call, you might want to store input tensors as members of your plugin.
This is of course possible if you know its exact shape a-prioro and comes with the cost of keeping the tensor in memory for the lifetime of your module instance.

You can use

```cpp
tensor.flat<float>().setZero();
```

to reset the values of your tensor prior to each call.


#### Tensor data access via pointers

As shown in the examples above, tensor data can be accessed through methods such as `flat<type>()` or `matrix<type>()` which return objects that represent the underlying data in the requested structure ([`tensorflow::Tensor` C++ API](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)).
To read and manipulate particular elements, you can directly call this object with the coordinates of an element.

```cpp
// matrix returns a 2D representation
// set element (b,i) to f
tensor.matrix<float>()(b, i) = float(f);
```

However, doing this for a large input tensor might entail some overhead.
Since the data is actually contiguous in memory (C-style "row-major" memory ordering), a faster (though less explicit) way of interacting with tensor data is using a pointer.


```cpp
// get the pointer to the first tensor element
float* d = tensor.flat<float>().data();
```

Now, the tensor data can be filled using simple and fast pointer arithmetic.

```cpp hl_lines="5"
// fill tensor data using pointer arithmethic
// memory ordering is row-major, so the most outer loop corresponds dimension 0
for (size_t b = 0; b < batchSize; b++) {
    for (size_t i = 0; i < nFeatures; i++, d++) {  // note the d++
        *d = float(i);
    }
}
```


#### Inter- and intra-operation parallelism

!!! danger "Debugging and local processing only"
    Parallelism between (inter) and within (intra) operations can greatly improve the inference performance.
    However, this allows TensorFlow to manage and schedule threads on its own, possibly interfering with the thread model inherent to CMSSW.
    For inference code that is to be officially integrated, you should ==avoid== inter- and intra-op parallelism and rather adhere to the examples shown above.

You can configure the amount of inter- and infra-op threads via the second argument of the `#!cpp tensorflow::createSession` method.

=== "Simple"

    ```cpp linenums="1"
    tensorflow::Session* session = tensorflow::createSession(graphDef, nThreads);
    ```

=== "Verbose"

    ```cpp linenums="1"
    tensorflow::SessionOptions sessionOptions;
    sessionOptions.config.set_intra_op_parallelism_threads(nThreads);
    sessionOptions.config.set_inter_op_parallelism_threads(nThreads);

    tensorflow::Session* session = tensorflow::createSession(graphDef, sessionOptions);
    ```

Then, when calling `#!cpp tensorflow::run`, pass the internal name of the TensorFlow threadpool, i.e. `"tensorflow"`, as the last argument.

```cpp linenums="1"
std::vector<tensorflow::Tensor> outputs;
tensorflow::run(
    session,
    { { inputTensorName, input } },
    { outputTensorName },
    &outputs,
    "tensorflow"
);
```


## Miscellaneous

### Logging

By default, TensorFlow logging is quite verbose.
This can be changed by either setting the `TF_CPP_MIN_LOG_LEVEL` environment varibale before calling `cmsRun`, or within your code through `#!c++ tensorflow::setLogging(level)`.

| Verbosity level | `TF_CPP_MIN_LOG_LEVEL` |
| --------------- | ---------------------- |
| debug           | "0"                    |
| info            | "1" (default)          |
| warning         | "2"                    |
| error           | "3"                    |
| none            | "4"                    |

Forwarding logs to the `MessageLogger` service is not possible yet.


## Links and further reading

- [`cmsml` package](https://cmsml.readthedocs.io)
- CMSSW
    - [TensorFlow interface documentation](http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_11_1_2/doc/html/de/d86/namespacetensorflow.html)
    - [TensorFlow interface header](https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/TensorFlow/interface/TensorFlow.h)
    - [CMSSW process options](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMParametersForModules#The_options_Parameter_Set)
    - [C++ interface of `stream` modules](https://twiki.cern.ch/twiki/bin/view/CMSPublic/FWMultithreadedFrameworkStreamModuleInterface)
- TensorFlow
    - [TensorFlow 2 tutorial](https://indico.cern.ch/event/882992/contributions/3721506/attachments/1994721/3327402/TensorFlow_2_Workshop_CERN_2020.pdf)
    - [`tf.function`](https://www.tensorflow.org/guide/function)
    - [C++ API](https://www.tensorflow.org/api_docs/cc)
    - [`tensorflow::Tensor`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)
    - [`tensorflow::Operation`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/operation)
    - [`tensorflow::ClientSession`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/client-session)
- Keras
    - [API](https://keras.io/api)

---

Authors: [Marcel Rieger](mailto:marcel.rieger@cern.ch)
