# lxplus-gpu.cern.ch

## How to use it?

`lxplus-gpu` are special lxplus nodes with GPU support. Therefore, via executing 

```bash
ssh <your_user_name>@lxplus-gpu.cern.ch
```

one can log in to such a node equipped with GPU support.

![Untitled](LxplusGPU_figs/Untitled.png)

## Software Environment

Three examples are given below to show how to set up a software environment properly.

1. Using LCG release software: after checking out an ideal software bundle with Cuda support at [`http://lcginfo.cern.ch/`](http://lcginfo.cern.ch/), one can set up an LCG environment by executing
  
    ```bash
    source /cvmfs/sft.cern.ch/lcg/views/<name of bundle>/**x86_64-centos7-gcc8-opt**/setup.sh
    ```
    
2. Using `pip`, especially with `virtualenv`:  using `pip` only to install software may mess up the global environment. Thus, it is better to create a "virtual environment" with `virtualenv` in order to eliminate potential issues in the package environment.
    1. As on lxplus, the default `virtualenv` command is installed with `python2` , it better to firstly install `virtualenv` with `python3`
      
        ```bash
        pip3 install virtualenv --user
        # Add following line to .bashrc and re-log in or source .bashrc
        # export PATH="/afs/cern.ch/user/<first letter of your username>/<username>/.local/bin:$PATH"
        ```
        
    2. Make sure you have `virtualenv` with `python3` correctly. Then go to the desired directory and create a virtual environment
      
        ```bash
        virtualenv <env name>
        source <env name>/bin/activate
        # now you are inside the virtual environment, your shell prompt will begin with "(<env name>)"
        ```
        
    3. To install packages properly, one should carefully check the CUDA version with `nvidia-smi` (as shown in figure before), and then find a proper version, pytorch is used as an example.
      
        ![Untitled](LxplusGPU_figs/Untitled1.png)
        
        ```bash
        # Execute the command shown in your terminal
        pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
        pip3 install jupyterlab matplotlib scikit-hep # install other packages if they are needed
        ```
3. Using `conda` package manager: `conda` pacakge manager is more convenient to install and use. To begin with, obtaining an `Anaconda` or `Miniconda` installer for Linux x86_64 platform. Then execute it on Lxplus.

    1. Please note that if you update your shell configuration (e.g. `.bashrc` file) by `conda init`, you may encounter failure due to inconsistent environment configuration.
    2. Installing packages via `conda` also needs special consideration on selecting proper CUDA version as discussed in `pip` part.

4. Container based solution: There exists several container images in `unpacked.cern.ch` inside `CVMFS` which often contain a large set of softwares and thus provide a complete computing environment. If you find a image helpful, you can use `singularity` command to load and use it.


## Examples 

After installing package, you can then use GPU based machine learning algorithms. Two examples are supplied as an example.

> 1). The first example aims at using a CNN to perform handwritten digits classification with `MNIST` dataset. The whole notebook can be found at [PytorchMNIST](Notebooks/PytorchMNIST.md). This example is modified from [an official `pytorch` example](https://github.com/pytorch/examples/tree/master/mnist).
> 2). The second example is modified from the simple MLP example from [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark). The whole notebook can be found at [TopTaggingMLP](Notebooks/TopTaggingMLP.md).