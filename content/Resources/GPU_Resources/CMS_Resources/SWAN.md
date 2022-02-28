# SWAN
## Preparation

1. Registration:

    To require GPU resources for SWAN: According to [this thread](https://swan-community.web.cern.ch/t/gpu-support-in-swan/108), one can create a ticket through [this link](https://cern.service-now.com/service-portal?id=functional_element&name=swan) to ask for GPU support at SWAN, it is now in beta version and limited to a small scale.
2. Setup SWAN with GPU resources:

    1. Once the registration is done, one can login [SWAN with Kerberes8 support](https://swan-k8s.cern.ch/) and then create his SWAN environment.
        
        <aside>
        💡 Note: When configuring the SWAN environment you will be given your choice of software stack. Be careful to use a software release with GPU support as well as an appropriate CUDA version. If you need to install additional software, it must be compatible with your chosen CUDA version.
        </aside>
        

![Untitled](./SWAN_figs/Conf_Env.png)

![Untitled](./SWAN_figs/Select_Release.png)

Another important option is the environment script, which will be discussed later in this document.

## Working with SWAN

1. After creation, one will browse the SWAN main directory `My Project` where all existing projects are displayed. A new project can be created by clicking the upper right "+" button. After creation one will be redirected to the newly created project, at which point the "+" button on the upper right panel can be used for creating new **notebook**.
    
    ![Untitled](SWAN_figs/New_proj.png)
    
    ![Untitled](SWAN_figs/Example_Proj.png)
    
2. It is possible to use the terminal for installing new packages or monitoring computational resources. 

    1. For package installation, one can install packages with package management tools, e.g. `pip` for `python`. To use the installed packages, you will need to wrap the environment configuration in a scrip, which will be executed by SWAN. Detailed documentation can be found by clicking the upper right "?" button.

    2. In addition to using top and htop to monitor ordinary resources, you can use nvidia-smi to monitor GPU usage.

    ![Untitled](SWAN_figs/SWAN_Terminal.png)
    

## Examples 

After installing package, you can then use GPU based machine learning algorithms. Two examples are supplied as an example.

1. The first example aims at using a CNN to perform handwritten digits classification with `MNIST` dataset. The whole notebook can be found at [PytorchMNIST](Notebooks/PytorchMNIST.md). This example is modified from [an official `pytorch` example](https://github.com/pytorch/examples/tree/master/mnist).

2. The second example is modified from the simple MLP example from [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark). The whole notebook can be found at [TopTaggingMLP](Notebooks/TopTaggingMLP.md).