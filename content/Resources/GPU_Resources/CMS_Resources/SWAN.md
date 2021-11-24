# SWAN

1. Registration:
    1. On CERN service-now, there is a [ticket](https://cern.service-now.com/service-portal/?id=sc_cat_item&name=GPU-request-access&fe=gpu-platform), while we have not managed to access GPU resources with it.
    2. In the ticket thread, we were informed another approach in requiring GPU resources for SWAN: According to [this thread](https://swan-community.web.cern.ch/t/gpu-support-in-swan/108), one can create a ticket through [this link](https://cern.service-now.com/service-portal?id=functional_element&name=swan).
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
    
3. After installing package, you can then use GPU based machine learning algorithms. Two examples are supplied as an example.