# Direct inference with TensorFlow 2
---

TensorFlow 2 is available since CMSSW\_11\_1\_X ([cmssw#28711](https://github.com/cms-sw/cmssw/pull/28711)). The integration into the software stack can be found in [cmsdist/tensorflow.spec](https://github.com/cms-sw/cmsdist/blob/latest/tensorflow.spec) and the interface is located in [cmssw/PhysicsTools/TensorFlow](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/TensorFlow).

The current version is 2.1.0 and, at the moment, only supports inference on CPU. GPU support is planned for the integration of version 2.2.

See the guide on [inference with TensorFlow 1](tensorflow1.md) or earlier versions.


## Software setup

To run the examples shown below, create a mininmal inference setup with the following snippet.

```shell linenums="1"
export SCRAM_ARCH="slc7_amd64_gcc820"
export CMSSW_VERSION="CMSSW_11_1_2"

source /cvmfs/cms.cern.ch/cmsset_default.sh

cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"

cmsenv
scram b
```

Below, the [cmsml](https://github.com/cms-ml/cmsml) Python package is used to convert models from TensorFlow objects (`tf.function`'s or Keras models) to protobuf graph files ([documentation](https://cmsml.readthedocs.io)). It should be available after executing the commands above. You can check its version via

```shell
python -c "import cmsml; print(cmsml.__version__)"
```

and compare to the [released tags](https://github.com/cms-ml/cmsml/tags). If you want to install a newer version, you can simply do that via pip

```shell
# into your user directory (usually ~/.local)
pip install --upgrade --user cmsml

# _or_

# into a custom directory
pip install --upgrade --install-option="--prefix=CUSTOM_DIRECTORY" cmsml
```

to a location of your choice.


## Saving your model

After successfully training, you should save your model in a protobuf graph file which can be read by the interface in CMSSW. Naturally, you only want to save that part of your model is required to run the network prediction, i.e., it should ==not== contain operations related to model training or loss functions (unless explicitely required). Also, to reduce the memory footprint and to accelerate the inference, variables should be converted to constant tensors. Both of these model transformations are provided by the `cmsml` package.

Instructions on how to transform and save your model are shown below, depending on whether you use Keras or plain TensorFlow with `tf.function`'s.

=== "Keras"

    The code below saves a Keras `Model` instance as a protobuf graph file using [`cmsml.tensorflow.save_graph`](https://cmsml.readthedocs.io/en/latest/api/tensorflow.html#cmsml.tensorflow.save_graph). In order for Keras to built the internal graph representation before saving, make sure to either compile the model, or pass an `input_shape` to the first layer:

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

=== "tf.function"

    Let's consider you write your network model in a single `tf.function`:

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
       y = tf.tanh(h, name="output")

       return y
    ```

    In TensorFlow terms, the `model` function is ==polymorphic== - it accepts different types of the input tensor `x` (`tf.float32`, `tf.float64`, ...). For each type, TensorFlow will create a ==concrete== function with an associated ``tf.Graph`` object. This mechanism is referred to as ==signature tracing==. For deeper insights into `tf.function`, the concepts of signature tracing, polymorphic and concrete functions, see the guide on [Better performance with `tf.function`](https://www.tensorflow.org/guide/function).

    To save the model as a protobuf graph file, you explicitely need to create a concrete function. However, this is fairly easy once you know the exact type of shape of all input arguments.

    ```python linenums="20"
    # create a concrete function
    cmodel = model.get_concrete_function(
        tf.TensorSpec(shape=[2, 10], dtype=tf.float32))

    # convert to binary (.pb extension) protobuf
    # with variables converted to constants
    cmsml.tensorflow.save_graph("graph.pb", cmodel, variables_to_constants=True)
    ```

    ??? success "Different method: Frozen signatures"
        Instead of creating a polymorphic `tf.function` and extracting a concrete one in a second step, you can directly define an input signature upon definition.

        ```python
        @tf.function(input_signature=(tf.TensorSpec(shape=[2, 10], dtype=tf.float32),))
        def model(x):
            ...
        ```

        This attaches a graph object to `model` but disables signature tracing since the input signature is frozen. However, you can directly pass it to `cmsml.tensorflow.save_graph`.


## Inference in CMSSW


### CMSSW module setup

If you are aiming to use the TensorFlow interface in your personal CMSSW ==plugin==, make sure to include

```xml
<use name="PhysicsTools/TensorFlow" />
```

in your `plugins/BuildFile.xml`. If you are working on a file in the using the interface in a file in the `src/` or `interface/` directory of your module, make sure to create a global `BuildFile.xml` next to your `src/` or `interface/` directories containing (at least):

```xml
<use name="PhysicsTools/TensorFlow" />

<export>
    <lib name="1" />
</export>
```


### Single-threaded inference

Todo.

??? success "Full example"
    Todo.


### Multi-threaded inference

Todo.

??? success "Full example"
    Todo.


### Links and further reading

- [cmsml package](https://cmsml.readthedocs.io)
- [TensorFlow C++ API](https://www.tensorflow.org/api_docs/cc)
    - [`tensorflow::Tensor`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/tensor)
    - [`tensorflow::Operation`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/operation)
    - [`tensorflow::ClientSession`](https://www.tensorflow.org/api_docs/cc/class/tensorflow/client-session)
- [`tf.function`](https://www.tensorflow.org/guide/function)
- [TensorFlow 2 tutorial](https://indico.cern.ch/event/882992/contributions/3721506/attachments/1994721/3327402/TensorFlow_2_Workshop_CERN_2020.pdf)
- [Keras API](https://keras.io/api)
