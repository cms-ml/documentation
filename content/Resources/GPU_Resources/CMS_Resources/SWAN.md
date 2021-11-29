# SWAN
## Preparation

1. Registration:
    To require GPU resources for SWAN: According to [this thread](https://swan-community.web.cern.ch/t/gpu-support-in-swan/108), one can create a ticket through [this link](https://cern.service-now.com/service-portal?id=functional_element&name=swan) to ask for GPU support at SWAN, it is now in beta version and limited to a small scale.
2. Setup SWAN with GPU resources.
    1. Once the registration is done, one can login [SWAN with Kerberes8 support](https://swan-k8s.cern.ch/) and then create his SWAN environment.
        
        <aside>
        💡 Note. When configuring SWAN environment, one should carefully choose the software stack he want to use. Not only one should choose to use the releases with GPU Support, but also she/he should keep the CUDA version in mind. Once additional software installation is needed, it must be configured to use the version compatible with current CUDA version.
        
        </aside>
        

![Untitled](SWAN_figs/Untitled.png)

![Untitled](SWAN_figs/Untitled%201.png)

Another important aspects is the environment script, which will be discussed later in this document.

1. After setting up the SWAN environment, one will browse the SWAN main directory `My Project` where all existing projects are displayed. A new project can be created by clicking the upper right "+" button. After creation one will be redirected to the new created project, whose "+" button at upper right panel can be used for creating new **notebook**.
    
    ![Untitled](SWAN_figs/Untitled%202.png)
    
    ![Untitled](SWAN_figs/Untitled%203.png)
    
2. Also it is possible to use the terminal for installing new packages or monitoring computational resources. 
    1. For package installation, one can install packages with package management tools, e.g. `pip` for `python`. To use the installed package, one need to wrap environment configuration into a script and let SWAN to execute. Detailed documentation can be found by clicking the upper right "?" button.
    2. To monitor the computational resources, in addition to ordinary CPU resources `top` and `htop` , you can also use the `nvidia-smi` to monitor GPU usage.
    
    ![Untitled](SWAN_figs/Untitled%204.png)
    

## Examples 

After installing package, you can then use GPU based machine learning algorithms. Two examples are supplied as an example.

> 1). The first example aims at using a CNN to perform handwritten digits classification with `MNIST` dataset. The whole notebook can be found at [PytorchMNIST](Notebooks/PytorchMNIST.md). This example is modified from [an official `pytorch` example](https://github.com/pytorch/examples/tree/master/mnist).
> 2). The second example is modified from the simple MLP example from [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark). The whole notebook can be found at [TopTaggingMLP](Notebooks/TopTaggingMLP.md).