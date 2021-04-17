# PyTorch Inference
PyTorch is an open source ML library developed by Facebook's AI Research lab. Initially released in late-2016, PyTorch is a relatively new tool, but has become increasingly popular among ML researchers (in fact, [some analyses](http://horace.io/pytorch-vs-tensorflow/) suggest it's becoming more popular than TensorFlow in academic communities!). PyTorch is written in idiomatic Python, such that its syntax is easy to parse for experienced Python programmers. Additionally, it is highly compatible with graphics processing units (GPUs), which can substantially accelerate many deep learning workflows. To date PyTorch has not been integrated into CMSSW. Trained PyTorch models may be evaluated in CMSSW via ONNX Runtime, but model construction and training workflows must currently exist outside of CMSSW. Given the considerable interest in PyTorch within the HEP/ML community, we have reason to believe it will soon be available, so stay tuned! 

## Introductory References

- [PyTorch Install Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LPC HATs: PyTorch](https://github.com/FNALLPC/machine-learning-hats/blob/master/3.1-dense-pytorch.ipynb)
- [Deep Learning w/ PyTorch Course Repo](https://github.com/Atcold/pytorch-Deep-Learning)
- [CODAS-HEP](https://codas-hep.org/)

## The Basics

### Tensors 
The fundamental PyTorch object is the tensor. At a glance, tensors behave similarly to NumPy arrays. For example, they are broadcasted, concatenated, and sliced in exactly the same way. However, tensors have been ''upgraded'' from Numpy arrays in two key ways:
1) Tensors have native GPU support. If a GPU is available at runtime, tensors can be transferred from CPU to GPU, where computations such as matrix operations are substantially faster. 
2) Tensors support automatic gradient (audograd) calculations, such that operations on tensors flagged with `requires_grad=True` are automatically tracked. The flow of tracked tensor operations defines a *computation graph* in which nodes are tensors and edges are functions mapping input tensors to output tensors. Gradients are calculated numerically via autograd by walking through this computation graph. 

### Neural Networks 
The PyTorch *nn* package specifies a set of modules that correspond to different neural network (NN) components and operations. For example, the `torch.nn.Linear` module defines a linear transform with learnable parameters and the `torch.nn.Flatten   ` module flattens two contiguous tensor dimensions. The `torch.nn.Sequential` module contains a set of modules such as `torch.nn.Linear` and `torch.nn.Sequential`, chaining them together to form the forward pass of a forward network. 

### Custom NNs 


## Working Examples 


## PyTorch at CMS 


