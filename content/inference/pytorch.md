# PyTorch Inference
PyTorch is an open source ML library developed by Facebook's AI Research lab. Initially released in late-2016, PyTorch is a relatively new tool, but has become increasingly popular among ML researchers (in fact, [some analyses](http://horace.io/pytorch-vs-tensorflow/) suggest it's becoming more popular than TensorFlow in academic communities!). PyTorch is written in idiomatic Python, so its syntax is easy to parse for experienced Python programmers. Additionally, it is highly compatible with graphics processing units (GPUs), which can substantially accelerate many deep learning workflows. To date PyTorch has not been integrated into CMSSW. Trained PyTorch models may be evaluated in CMSSW via ONNX Runtime, but model construction and training workflows must currently exist outside of CMSSW. Given the considerable interest in PyTorch within the HEP/ML community, we have reason to believe it will soon be available, so stay tuned! 

## Introductory References

- [PyTorch Install Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LPC HATs: PyTorch](https://github.com/FNALLPC/machine-learning-hats/blob/master/3.1-dense-pytorch.ipynb)
- [Deep Learning w/ PyTorch Course Repo](https://github.com/Atcold/pytorch-Deep-Learning)
- [CODAS-HEP](https://codas-hep.org/)

## The Basics
The following documentation surrounds a set of code snippets designed to highlight some important ML features made available in PyTorch. In the following sections, we'll break down snippets from this script, highlighting specifically the PyTorch objects in it. 

### Tensors 
The fundamental PyTorch object is the tensor. At a glance, tensors behave similarly to NumPy arrays. For example, they are broadcasted, concatenated, and sliced in exactly the same way. The following examples highlight some common numpy-like tensor transformations:
```python
a = torch.randn(size=(2,2))
>>> tensor([[ 1.3552, -0.0204],
            [ 1.2677, -0.8926]])
a.view(-1, 1)
>>> tensor([[ 1.3552],
            [-0.0204],
            [ 1.2677],
            [-0.8926]])
a.transpose(0, 1)
>>> tensor([[ 1.3552,  1.2677],
            [-0.0204, -0.8926]])
a.unsqueeze(dim=0)
>>> tensor([[[ 1.3552, -0.0204],
             [ 1.2677, -0.8926]]])
a.squeeze(dim=0)
>>> tensor([[ 1.3552, -0.0204],
            [ 1.2677, -0.8926]])
```
Additionally, torch supports familiar matrix operations with various syntax options: 
``` python
m1 = torch.randn(size=(2,3))
m2 = torch.randn(size=(3,2))
x = torch.randn(3)

m1 @ m2 == m1.mm(m2) # matrix multiplication
>>> tensor([[True, True],
            [True, True]])

m1 @ x == m1.mv(x) # matrix-vector multiplication
>>> tensor([True, True])

m1.t() == m1.transpose(0, 1) # matrix transpose
>>> tensor([[True, True],
            [True, True],
            [True, True]])
```
Note that `tensor.transpose(dim0, dim1)` is a more general operation than `tensor.t()`. 
It is important to note that tensors have been ''upgraded'' from Numpy arrays in two key ways:
1) Tensors have native GPU support. If a GPU is available at runtime, tensors can be transferred from CPU to GPU, where computations such as matrix operations are substantially faster. Note that tensor operations must be performed on objects on the same device. PyTorch supports CUDA tensor types for GPU computation (see the [PyTorch Cuda Semantics](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics) guide). 
2) Tensors support automatic gradient (audograd) calculations, such that operations on tensors flagged with `requires_grad=True` are automatically tracked. The flow of tracked tensor operations defines a *computation graph* in which nodes are tensors and edges are functions mapping input tensors to output tensors. Gradients are calculated numerically via autograd by walking through this computation graph. 

#### GPU Support
Tensors are created on the host CPU by default:
```python
b = torch.zeros([2,3], dtype=torch.int32)
b.device
>>> cpu
```

You can also create tensors on any available GPUs:
```python
torch.cuda.is_available() # check that a GPU is available
>>> True 
cuda0 = torch.device('cuda:0')
c = torch.ones([2,3], dtype=torch.int32, device=cuda0)
c.device
>>> cuda:0
```

You can also move tensors between devices:
```python
b = b.to(cuda0)
b.device
>>> cuda:0
```

There are trade-offs between computations on the CPU and GPU. GPUs have limited memory and there is a cost associated with transfering data from CPUs to GPUs. However, GPUs perform heavy matrix operations much faster than CPUs, and are therefore often used to speed up training routines.  

```python
N = 1000 # 
for i, N in enumerate([10, 100, 500, 1000, 5000]):
    print("({},{}) Matrices:".format(N,N))
    M1_cpu = torch.randn(size=(N,N), device='cpu')
    M2_cpu = torch.randn(size=(N,N), device='cpu')
    M1_gpu = torch.randn(size=(N,N), device=cuda0)
    M2_gpu = torch.randn(size=(N,N), device=cuda0)
    if (i==0):
        print('Check devices for each tensor:')
        print('M1_cpu, M2_cpu devices:', M1_cpu.device, M2_cpu.device)
        print('M1_gpu, M2_gpu devices:', M1_gpu.device, M2_gpu.device)

    def large_matrix_multiply(M1, M2):
        return M1 * M2.transpose(0,1)
    
    n_iter = 1000
    t_cpu = Timer(lambda: large_matrix_multiply(M1_cpu, M2_cpu))
    cpu_time = t_cpu.timeit(number=n_iter)/n_iter
    print('cpu time per call: {:.6f} s'.format(cpu_time))

    t_gpu = Timer(lambda: large_matrix_multiply(M1_gpu, M2_gpu))
    gpu_time = t_gpu.timeit(number=n_iter)/n_iter
    print('gpu time per call: {:.6f} s'.format(gpu_time))
    print('gpu_time/cpu_time: {:.6f}\n'.format(gpu_time/cpu_time))

>>> (10,10) Matrices:
Check devices for each tensor:
M1_cpu, M2_cpu devices: cpu cpu
M1_gpu, M2_gpu devices: cuda:0 cuda:0
cpu time per call: 0.000008 s
gpu time per call: 0.000015 s
gpu_time/cpu_time: 1.904711

(100,100) Matrices:
cpu time per call: 0.000015 s
gpu time per call: 0.000015 s
gpu_time/cpu_time: 0.993163

(500,500) Matrices:
cpu time per call: 0.000058 s
gpu time per call: 0.000016 s
gpu_time/cpu_time: 0.267371

(1000,1000) Matrices:
cpu time per call: 0.000170 s
gpu time per call: 0.000015 s
gpu_time/cpu_time: 0.089784

(5000,5000) Matrices:
cpu time per call: 0.025083 s
gpu time per call: 0.000011 s
gpu_time/cpu_time: 0.000419
```

The complete list of Torch Tensor operations is available in the [docs](https://pytorch.org/docs/stable/torch.html?highlight=mm). 

#### Autograd

Backpropagation occurs automatically through autograd. For example, consider the following function and its derivatives:

$$\begin{aligned} 
f(\textbf{a}, \textbf{b}) &= \textbf{a}^T \textbf{X} \textbf{b} \\ 
\frac{\partial f}{\partial \textbf{a}} &= \textbf{b}^T \textbf{X}^T\\
\frac{\partial f}{\partial \textbf{b}} &= \textbf{a}^T \textbf{X}
\end{aligned}$$

Given specific choices of $\textbf{X}$, $\textbf{a}$, and $\textbf{b}$, we can calculate the corresponding derivatives via autograd by requiring a gradient to be stored in each relevant tensor:
```python 
X = torch.ones((2,2), requires_grad=True)
a = torch.tensor([0.5, 1], requires_grad=True)
b = torch.tensor([0.5, -2], requires_grad=True)
f = a.T @ X @ b
f
>>> tensor(-2.2500, grad_fn=<DotBackward>) 
f.backward() # backprop 
a.grad
>>> tensor([-1.5000, -1.5000])
b.T @ X.T 
>>> tensor([-1.5000, -1.5000], grad_fn=<SqueezeBackward3>)
b.grad
>>> tensor([1.5000, 1.5000])
a.T @ X
>>> tensor([1.5000, 1.5000], grad_fn=<SqueezeBackward3>)
```
The `tensor.backward()` call initiates backpropagation, accumulating the gradient backward through a series of `grad_fn` labels tied to each tensor (e.g. `<DotBackward>`, indicating the dot product $(\textbf{a}^T\textbf{X})\textbf{b}$). 

### Data Utils
PyTorch is equipped with many useful data-handling utilities. For example, the `torch.utils.data` package implements datasets (`torch.utils.data.Dataset`) and iterable data loaders (`torch.utils.data.DataLoader`). Additionally, various batching and sampling schemes are available. 

You can create custom iterable datasets via `torch.utils.data.Dataset`, for example a dataset collecting the results of XOR on two binary inputs:
``` Python
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, device):
        self.samples = torch.tensor([[0,0], [0,1], [1,0], [1,1]]).float().to(device)
        self.targets = np.logical_xor(self.samples[:,0], 
                                      self.samples[:,1]).float().to(device)
        
    def __len__(self):
        return len(self.targets)
     
    def __getitem__(self,idx):
        return({'x': self.samples[idx],
                'y': self.targets[idx]})

```
Dataloaders, from `torch.utils.data.DataLoader`, can generate shuffled batches of data via multiple workers. Here, we load our datasets onto the GPU: 
``` Python
from torch.utils.data import DataLoader

device = 'cpu'
train_data = Data(device)
test_data = Data(device)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
for i, batch in enumerate(train_loader):
    print(i, batch)
 
>>> 0 {'x': tensor([[0., 0.]]), 'y': tensor([0.])}
    1 {'x': tensor([[1., 0.]]), 'y': tensor([1.])}
    2 {'x': tensor([[1., 1.]]), 'y': tensor([0.])}
    3 {'x': tensor([[0., 1.]]), 'y': tensor([1.])}

```
The full set of data utils is available in the [docs](https://pytorch.org/docs/stable/data.html?highlight=dataset). 

### Neural Networks 
The PyTorch *nn* package specifies a set of modules that correspond to different neural network (NN) components and operations. For example, the `torch.nn.Linear` module defines a linear transform with learnable parameters and the `torch.nn.Flatten   ` module flattens two contiguous tensor dimensions. The `torch.nn.Sequential` module contains a set of modules such as `torch.nn.Linear` and `torch.nn.Sequential`, chaining them together to form the forward pass of a forward network. Furthermore, one may specify various pre-implemented loss functions, for example `torch.nn.BCELoss` and `torch.nn.KLDivLoss`. The full set of PyTorch NN building blocks is available in the [docs](https://pytorch.org/docs/stable/nn.html). 

As an example, we can design a simple neural network designed to reproduce the output of the XOR operation on binary inputs. To do so, we can compute a simple NN of the form:

$$\begin{aligned}
x_{in}&\in\{0,1\}^{2}\\
l_1 &= \sigma(W_1^Tx_{in} + b_1); \ W_1\in\mathbb{R}^{2\times2},\ b_1\in\mathbb{R}^{2}\\
l_2 &= \sigma(W_2^Tx + b_2); \ W_2\in\mathbb{R}^{2},\ b_1\in\mathbb{R}\\
\end{aligned}$$

``` Python
import torch.nn as nn

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x
  
model = Network().to(device)
model(train_data['x'])

>>> tensor([[0.5000],
            [0.4814],
            [0.5148],
            [0.4957]], grad_fn=<SigmoidBackward>)
```

### Optimizers 
Training a neural network involves minimizing a loss function; classes in the `torch.optim` package implement various optimization strategies for example stochastic gradient descent and Adam through `torch.optim.SGD` and `torch.optim.Adam` respectively. Optimizers are configurable through parameters such as the learning rate (configuring the optimizer's step size). The full set of optimizers and accompanying tutorials are available in the [docs](https://pytorch.org/docs/stable/optim.html).

To demonstrate the use of an optimizer, let's train the NN above to produce the results of the XOR operation on binary inputs. Here we'll use the [Adam optimizer](https://arxiv.org/abs/1412.6980):

```python    
from torch import optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt

# helpful references:
# Learning XOR: exploring the space of a classic problem
# https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7
# https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html

# the training function initiates backprop and 
# steps the optimizer towards the weights that 
# optimize the loss function 
def train(model, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch['x'])
        y, output = batch['y'], output.squeeze(1)
        
        # optimize binary cross entropy:
        # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
        loss = F.binary_cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return np.mean(losses)

# the test function does not adjust the model's weights
def test(model, test_loader):
    model.eval()
    losses, n_correct, n_incorrect = [], 0, 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            output = model(batch['x'])
            y, output = batch['y'], output.squeeze(1)
            loss = F.binary_cross_entropy(output, y, 
                                          reduction='mean').item()
            losses.append(loss)
            
            # determine accuracy by thresholding model output at 0.5
            batch_correct = torch.sum(((output>0.5) & (y==1)) |
                                      ((output<0.5) & (y==0)))
            batch_incorrect = len(y) - batch_correct
            n_correct += batch_correct
            n_incorrect += batch_incorrect
            
    return np.mean(losses), n_correct/(n_correct+n_incorrect)


# randomly initialize the model's weights
for module in model.modules():
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0, 1)

# send weights to optimizer 
lr = 2.5e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 500
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, test_loader)
    if epoch%25==0:
        print('epoch={}: train_loss={:.3f}, test_loss={:.3f}, test_acc={:.3f}'
              .format(epoch, train_loss, test_loss, test_acc))
        
>>> epoch=25: train_loss=0.683, test_loss=0.681, test_acc=0.500
    epoch=50: train_loss=0.665, test_loss=0.664, test_acc=0.750
    epoch=75: train_loss=0.640, test_loss=0.635, test_acc=0.750
    epoch=100: train_loss=0.598, test_loss=0.595, test_acc=0.750
    epoch=125: train_loss=0.554, test_loss=0.550, test_acc=0.750
    epoch=150: train_loss=0.502, test_loss=0.498, test_acc=0.750
    epoch=175: train_loss=0.435, test_loss=0.432, test_acc=0.750
    epoch=200: train_loss=0.360, test_loss=0.358, test_acc=0.750
    epoch=225: train_loss=0.290, test_loss=0.287, test_acc=1.000
    epoch=250: train_loss=0.230, test_loss=0.228, test_acc=1.000
    epoch=275: train_loss=0.184, test_loss=0.183, test_acc=1.000
    epoch=300: train_loss=0.149, test_loss=0.148, test_acc=1.000
    epoch=325: train_loss=0.122, test_loss=0.122, test_acc=1.000
    epoch=350: train_loss=0.102, test_loss=0.101, test_acc=1.000
    epoch=375: train_loss=0.086, test_loss=0.086, test_acc=1.000
    epoch=400: train_loss=0.074, test_loss=0.073, test_acc=1.000
    epoch=425: train_loss=0.064, test_loss=0.063, test_acc=1.000
    epoch=450: train_loss=0.056, test_loss=0.055, test_acc=1.000
    epoch=475: train_loss=0.049, test_loss=0.049, test_acc=1.000
    epoch=500: train_loss=0.043, test_loss=0.043, test_acc=1.000
```
Here, the model has converged to 100% test accuracy, indicating that it has learned to reproduce the XOR outputs perfectly. Note that even though the test accuracy is 100%, the test loss (BCE) decreases steadily; this is because the BCE loss is nonzero when $y_{output}$ is not exactly 0 or 1, while accuracy is determined by thresholding the model outputs such that each prediction is the boolean $(y_{output} > 0.5)$. This highlights that it is important to choose the correct performance metric for an ML problem. In the case of XOR, perfect test accuracy is sufficient. Let's check that we've recovered the XOR output by extracting the model's weights and using them to build a custom XOR function:

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
        
>>> l1.weight tensor([[ 7.2888, -6.4168],
                      [ 7.2824, -8.1637]])
    l1.bias tensor([ 2.6895, -3.9633])
    l2.weight tensor([[-6.3500,  8.0990]])
    l2.bias tensor([2.5058])
```

Because our model was built with `nn.Linear` modules, we have weight matrices and bias terms. Next, we'll hard-code the matrix operations into a custom XOR function based on the architecture of the NN: 

```python
def XOR(x):
    w1 = torch.tensor([[ 7.2888, -6.4168],
                       [ 7.2824, -8.1637]]).t()
    b1 = torch.tensor([ 2.6895, -3.9633])
    layer1_out = torch.tensor([x[0]*w1[0,0] + x[1]*w1[1,0] + b1[0],
                               x[0]*w1[0,1] + x[1]*w1[1,1] + b1[1]])
    layer1_out = torch.sigmoid(layer1_out)

    w2 = torch.tensor([-6.3500,  8.0990])
    b2 = 2.5058
    layer2_out = layer1_out[0]*w2[0] + layer1_out[1]*w2[1] + b2
    layer2_out = torch.sigmoid(layer2_out)
    return layer2_out, (layer2_out > 0.5)

XOR([0.,0.])
>>> (tensor(0.0359), tensor(False))
XOR([0.,1.])
>>> (tensor(0.9135), tensor(True))
XOR([1.,0.])
>>> (tensor(0.9815), tensor(True))
XOR([1.,1.])
>>> (tensor(0.0265), tensor(False))
```

There we have it - the NN learned XOR! 


## PyTorch in CMSSW
### Via ONNX
One way to incorporate your PyTorch models into CMSSW is through the [Open Neural Network Exchange](https://www.onnxruntime.ai/about.html) (ONNX) Runtime tool. In brief, ONNX supports training and inference for a variety of ML frameworks, and is currently integrated into CMSSW (see the CMS ML tutorial).  PyTorch hosts an excellent tutorial on [exporting a model from PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html). ONNX is available in CMSSW (see a relevant [discussion](https://github.com/cms-sw/cmssw/issues/27458) in the CMSSW git repo). 

#### Example Use Cases 
The $ZZ\rightarrow 4b$ analysis utilizes trained PyTorch models via ONNX in CMSSW (see the corresponding [repo](https://github.com/patrickbryant/ZZ4b/blob/master/README.md)). Briefly, they run ONNX in CMSSW_11_X via the CMSSW package `PhysicsTools/ONNXRuntime`, using it to define a [multiClassifierONNX](https://github.com/patrickbryant/ZZ4b/blob/5931a21d8005683e23166c0b44b9594b52ad1126/nTupleAnalysis/interface/multiClassifierONNX.h) class. This multiclassifier is capable of loading pre-trained PyTorch models specified by a `modelFile` string as follows:

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
- When instantiating a `DataLoader`, `shuffle=True` should be enabled for training data but not for validation and testing data. At each training epoch, this will vary the order of data objects in each batch; accordingly, it is not efficient to load the full dataset (in its original ordering) into GPU  memory before training. Instead, enable `num_workers>1`; this allows the `DataLoader` to load batches to the GPU as they're prepared. Note that this launches muliple threads on the CPU. For more information, see a corresponding [discussion](https://discuss.pytorch.org/t/keras-trains-significantly-faster-than-pytorch-for-simple-network/124303/5) in the PyTorch forum. 