# PyTorch Inference
PyTorch is an open source ML library developed by Facebook's AI Research lab. Initially released in late-2016, PyTorch is a relatively new tool, but has become increasingly popular among ML researchers (in fact, [some analyses](http://horace.io/pytorch-vs-tensorflow/) suggest it's becoming more popular than TensorFlow in academic communities!). PyTorch is written in idiomatic Python, so its syntax is easy to parse for experienced Python programmers. Additionally, it is highly compatible with graphics processing units (GPUs), which can substantially accelerate many deep learning workflows. To date PyTorch has not been integrated into CMSSW. Trained PyTorch models may be evaluated in CMSSW via ONNX Runtime, but model construction and training workflows must currently exist outside of CMSSW. Given the considerable interest in PyTorch within the HEP/ML community, we have reason to believe it will soon be available, so stay tuned! 

## Introductory References

- [PyTorch Install Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LPC HATs: PyTorch](https://github.com/FNALLPC/machine-learning-hats/blob/master/3.1-dense-pytorch.ipynb)
- [Deep Learning w/ PyTorch Course Repo](https://github.com/Atcold/pytorch-Deep-Learning)
- [CODAS-HEP](https://codas-hep.org/)

## The Basics

### Tensors 
The fundamental PyTorch object is the tensor. At a glance, tensors behave similarly to NumPy arrays. For example, they are broadcasted, concatenated, and sliced in exactly the same way. However, tensors have been ''upgraded'' from Numpy arrays in two key ways:
1) Tensors have native GPU support. If a GPU is available at runtime, tensors can be transferred from CPU to GPU, where computations such as matrix operations are substantially faster. Note that tensor operations must be performed on objects on the same device. PyTorch supports CUDA tensor types for GPU computation (see the [PyTorch Cuda Semantics](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics) guide).
2) Tensors support automatic gradient (audograd) calculations, such that operations on tensors flagged with `requires_grad=True` are automatically tracked. The flow of tracked tensor operations defines a *computation graph* in which nodes are tensors and edges are functions mapping input tensors to output tensors. Gradients are calculated numerically via autograd by walking through this computation graph. 

The complete list of Torch Tensor operations is available in the [docs](https://pytorch.org/docs/stable/torch.html?highlight=mm). 

### Neural Networks 
The PyTorch *nn* package specifies a set of modules that correspond to different neural network (NN) components and operations. For example, the `torch.nn.Linear` module defines a linear transform with learnable parameters and the `torch.nn.Flatten   ` module flattens two contiguous tensor dimensions. The `torch.nn.Sequential` module contains a set of modules such as `torch.nn.Linear` and `torch.nn.Sequential`, chaining them together to form the forward pass of a forward network. Furthermore, one may specify various pre-implemented loss functions, for example `torch.nn.BCELoss` and `torch.nn.KLDivLoss`. The full set of PyTorch NN building blocks is available in the [docs](https://pytorch.org/docs/stable/nn.html). 

### Optimizers 
Training a neural network involves minimizing a loss function; classes in the `torch.optim` package implement various optimization strategies for example stochastic gradient descent and Adam through `torch.optim.SGD` and `torch.optim.Adam` respectively. Optimizers are configurable through parameters such as the learning rate (configuring the optimizer's step size). The full set of optimizers and accompanying tutorials are available in the [docs](https://pytorch.org/docs/stable/optim.html). 

### Data Utils
PyTorch is equipped with many useful data-handling utilities. For example, the `torch.utils.data` package implements datasets (`torch.utils.data.Dataset`) and iterable data loaders (`torch.utils.data.DataLoader`). Additionally, various batching and sampling schemes are available. The full set of data utils is available in the [docs](https://pytorch.org/docs/stable/data.html?highlight=dataset). 

## PyTorch in CMSSW
### Via ONNX
One way to incorporate your PyTorch models into CMSSW is through the [Open Neural Network Exchange](https://www.onnxruntime.ai/about.html) (ONNX) Runtime tool. In brief, ONNX supports training and inference for a variety of ML frameworks, and is currently integrated into CMSSW (see the CMS ML tutorial).  PyTorch hosts an excellent tutorial on [exporting a model from PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html). ONNX is available in CMSSW (see a relevant [discussion](https://github.com/cms-sw/cmssw/issues/27458) in the CMSSW git repo). 

#### Example Use Cases 
The ZZ\\(\rightarrow\\)4b analysis utilizes trained PyTorch models via ONNX in CMSSW (see the corresponding [repo](https://github.com/patrickbryant/ZZ4b/blob/master/README.md)). Briefly, they run ONNX in CMSSW_11_X via the CMSSW package `PhysicsTools/ONNXRuntime`, using it to define a [multiClassifierONNX](https://github.com/patrickbryant/ZZ4b/blob/5931a21d8005683e23166c0b44b9594b52ad1126/nTupleAnalysis/interface/multiClassifierONNX.h) class. This multiclassifier is capable of loading pre-trained PyTorch models specified by a `modelFile` string as follows:

``` C++
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

std::unique_ptr<cms::Ort::ONNXRuntime> model;
Ort::SessionOptions* session_options = new Ort::SessionOptions();
session_options->SetIntraOpNumThreads(1);
model = std::make_unique<cms::Ort::ONNXRuntime>(modelFile, session_options);
```

### Via Triton
Coprocessors (GPUs, FPGAs, etc.) are frequently used to accelerate ML operations such as inference and training. In the 'as-a-service' paradigm, users can access cloud-based applications through lightweight client inferfaces. The Services for Optimized Network Inference on Coprocessors ([SONIC](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore)) framework implements this paradigm in CMSSW, allowing the optimal integration of GPUs into event processing workflows. One powerful implementation of SONIC is the the NVIDIA Triton Inference Server, which is flexible with respect to ML framework, storage source, and hardware infrastructure. For more details, see the corresponding [NVIDIA developer blog entry](https://developer.nvidia.com/blog/scaling-inference-in-high-energy-particle-physics-at-fermilab-using-nvidia-triton-inference-server/). 

A Graph Attention Network (GAN) is available via Triton in CMSSW, and can be accessed here: https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton/test

## Training Tips
- When instantiating a `DataLoader`, `shuffle=True` should be enabled for training data but not for validation and testing data. At each training epoch, this will vary the order of data objects in each batch; accordingly, it is not efficient to load the full dataset (in its original ordering) into GPU  memory before training. Instead, enable `num_workers>1`; this allows the `DataLoader` to load batches to the GPU as they're prepared. Note that this launches muliple processerson the CPU. For more information, see a corresponding [discussion](https://discuss.pytorch.org/t/keras-trains-significantly-faster-than-pytorch-for-simple-network/124303/5) in the PyTorch forum. 