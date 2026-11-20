# PyTorch in CMSSW
The torch interface is split into a [general torch wrapper](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/PyTorch) and an [Alpaka supported interface](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/PyTorchAlpaka). A full ML CMSSW pipeline is implemented and tested in [PyTorchAlpakaTest](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/PyTorchAlpakaTest).

It enables seamless integration between PyTorch and the CMSSW SoA implementation. It provides:
- Support for automatic conversion of optimized SoA to torch tensors, with memory blobs reusage.
- Support for both just-in-time (JIT) and ahead-of-time (AOT) model execution (Beta version for AOT).
## PyTorchService
To not interfere with CMSSW threading model, `PyTorchService` **MUST** be included in the `cmsRun` configuration path, whenever PyTorch is used. The service will disable internal threading of PyTorch. 

An example setup can be found in [PyTorchAlpakaTest](https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PyTorchAlpakaTest/test/runPyTorchAlpakaTest.py). More on PyTorch threading model: [CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).

## Inference: JIT and AOT Model Execution
A Wrapper for the torch model stored with Just-in-Time (JIT), `Model` class, is provided enabling inference with native torch `Tensor` objects. To run direct inference on CMSSW PortableObjects and SoAs, the custom wrapper in package `PyTorchAlpaka` (see: [AlpakaModel.h](https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h)) has to be used.

### Just-in-Time:
- Loads `torch::jit::script::Module` at runtime.
- Compiles model on-the-fly.
- Introduces warm-up overhead without additional optimization.
- When storing model through tracing, compatibility and correctness have to be checked

Example how to export models from PyTorch Python API, more can be found in scripts for creating models in [PyTorchAlpakaTest](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/PyTorchAlpakaTest):
```py
batch_size = 10
input_tensor = torch.randn(batch_size, shape)
tm = torch.jit.trace(simplenet.eval(), input_tensor)
tm.save(f"traced_model.pt")
```
### Ahead-of-Time (example code, not production ready):

**The scripts provided for compiling the AOT model are purely for illustrative purposes to simplify further development, and are by no means intended for production use.**

- Uses PyTorch AOT compiler to generate `.cpp` and `.so` files. (prerequisite done manually by end-user)
- Package provide helper scripts to automate compilation process with CMSSW provided tools to some extent
- Loads compiled model via [AOTIModelPackageLoader](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/csrc/inductor/aoti_package/model_package_loader.h).
- Eliminates JIT overhead, enable optimization, but requires architecture-specific handling 

More in depth introduction to the concepts used with AOT compilation see: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747

## PyTorchAlpaka package

This package extends the PyTorch implementation and enables seamless integration with the Alpaka-based heterogeneous computing backend, supporting inference workflows with usage of `pytorch` library with `PortableCollection`s objects allowing users to run direct inference with reduced memory footprint. It provides:
- Compatibility with Alpaka device/queue abstractions.
- Single-threading and CUDA stream management are handled by `QueueGuard` objects specialized for each supported backend.

### Interface for Alpaka Modules
All Pytorch based modules should add `PyTorchService` to disable internal torchlib threading. It enforces single-threaded execution on CPU backends.

Examples demonstrating the interoperability of PyTorch with Alpaka in the CMSSW environment can be found in the [PyTorchAlpakaTest](https://github.com/cms-sw/cmssw/tree/master/PhysicsTools/PyTorchAlpakaTest) directory. The basic test pipeline includes:
- *SimpleNet* composed with few Dense layers, that operate on SoA style portable data structures
- *MaskedNet* shows how to use multiple input data with `Eigen::Vector` and `SOA_SCALAR`
- *TinyResNet* emulate more complex scenario with `Eigen::Matrix` and how one can implement image-like Tensor implementation
- *MulitHeadNet* handle networks that return more than one output tensor 

### Direct Inference on SoA 
The interface provides a converter to dynamically wrap SoA data into one or more `torch::tensors` without the need to copy data (or minimal copy overhead).

**Due to the lack of const correctness ensured by PyTorch, `const` data is currently being copied.**

#### TensorCollection
The structural information of the inputs/outputs SoA are stored in an `TensorCollection`. Which is a high level object to register column lists from which tensors are created

The `TensorCollection` can be defined by first initializing the object and then adding data blocks with `add` to the internal metadata. Each registered tensor is transformed into a PyTorch tensor (without taking ownership) whose size and type are derived from the columns provided.

For two example SoAs Templates, which are stored in PortableCollections, columns can be added to `TensorCollection`, by using the Metarecords implementation of SoAs.

- **Input SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoATemplate,
    SOA_EIGEN_COLUMN(Eigen::Vector3d, a),
    SOA_EIGEN_COLUMN(Eigen::Vector3d, b),
    SOA_EIGEN_COLUMN(Eigen::Matrix2f, c),
    SOA_COLUMN(double, x),
    SOA_COLUMN(double, y),
    SOA_COLUMN(double, z),
    SOA_SCALAR(float, type),
    SOA_SCALAR(int, someNumber));
```
- **Output SoA:**
```cpp
GENERATE_SOA_LAYOUT(SoAOutputTemplate,
                    SOA_COLUMN(int, cluster));
```
- **Get Metarecords from Portable Collections:**
```cpp
PortableCollection<SoA, Device> deviceCollection(batch_size, queue);
PortableCollection<SoA_Result, Device> deviceResultCollection(batch_size, queue);
fill(queue, deviceCollection);
auto records = deviceCollection.view().records();
auto result_records = deviceResultCollection.view().records();
```
- **For each function call** of `add` (i.e. one tensor), **add the columns** that should be merged into a single tensor. The datatypes must be the same, and the columns must be contiguous. This means, only columns that are defined directly after each other in the SoA layout can be used for the same tensor. However, not all columns of an SoA have to be used. Only those mentioned in the `registry_tensor` are selected for the tensor creation. Any holes in contiguity created by the alignment are automatically taken care of by the stride calculation.

**IMPORTANT:** continuity of memory is a strict requirement!
```
TensorCollection input(batch_size);
input.add<SoA>("eigen_vector", records.a(), records.b());
input.add<SoA>("eigen_matrix", records.c());
input.add<SoA>("column", records.x(), records.y(), records.z());
input.add<SoA>("scalar", records.type());
input.change_order({"column", "scalar", "eigen_matrix", "eigen_vector"});

TensorCollection output(batch_size);
output.add<SoA>("result", result_view.cluster());
```

For Eigen column types, providing a single Eigen::Vector or Eigen::Matrix is interpreted as a 2D tensor where each vector entry corresponds to a column. In this case, the resulting tensor shape is (nElements, dimension) instead of (nElements, 1, dimension).
In other words, if you pass only one Eigen vector, its components are treated as feature dimensions rather than as a batch of size 1. This matches the typical layout used in machine learning, where each row (or element) represents one sample, and each column represents one feature.

After adding all the blocks to the `TensorCollection`, the order of the blocks for inference can be adapted by calling `change_order()`. The order should match the expected input configuration of the PyTorch model.

### Limitations
- Current implementation supports `SerialSync` and `CudaAsync` backends only. `ROCmAsync` backend is supported via SerialSync fallback mechanism due to missing `pytorch-hip` library in CMSSW (see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75), with explicit `alpaka::wait()` call to copy data to host and back to device.
- Const correctness and thread-safety relies on `torch::from_blob()` mechanism which currently does not ensure that data will not be modified internally. There is ongoing work to support COW tensors but until this support will be integrated in mainstream PyTorch the provided solution materialises (copies) the tensors if passed registry points to `const` memory. For more information please check [Const correctness and thread-safety of torch::from_blob with external memory](https://discuss.pytorch.org/t/const-correctness-and-thread-safety-of-torch-from-blob-with-external-memory/223521) and [pytorch:#97856](https://github.com/pytorch/pytorch/issues/97856)
- For multi output branch models the intermediate copy of output is done, so there is no "true" no-copy mechanism under the hood.
- AOT support is under active development and subject to changes that obey CMSSW releasing rules.
