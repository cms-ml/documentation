Containers are a great solution to isolate a software environment, especially in batch systems like lxplus. 
At the moment two container solutations are supported  **Apptainer** ( previously called Singularity), and **Docker**.

### Using Singularity

The unpacked.cern.ch service mounts on CVMFS contains many singularity images, some of which are suitable for machine
 learning applications.
A description of each of the images is beyond the scope of this document. 
However, if you find an image which is useful for your application, you can use if by running a Singularity container
 with the appropriate options. 
 For example:
```bash
singularity run --nv --bind <bind_mount_path> /cvmfs/unpacked.cern.ch/<path_to_image>
```

#### Examples

After installing package, you can then use GPU based machine learning algorithms. Two examples are supplied as an
example.

1. The first example aims at using a CNN to perform handwritten digits classification with `MNIST` dataset. The whole notebook can be found at [pytorch_mnist](notebooks/pytorch_mnist.md). This example is modified from [an official `pytorch` example](https://github.com/pytorch/examples/tree/master/mnist).

2. The second example is modified from the simple MLP example from [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark). The whole notebook can be found at [toptagging_mlp](notebooks/toptagging_mlp.md).



### Using Docker
Docker is not supported at the moment in the interactive node of lxplus (like lxplus-gpu). However Docker is supported
on HTCondor for job submission. 

This option can be very handy for users, as HTCondor can pull images from any public registry, like
[DockerHub](https://hub.docker.com/) or [GitLab registry](https://gitlab.cern.ch/). 
The user can follow this workflow:
1. Define a custom image on top of a commonly available pytorch or tensorflow image
2. Add the desidered packages and configuration
3. Push the docker image on a registry
4. Use the image in a HTCondor job

The rest of the page is a step by step tutorial for this workflow.

#### Define the image

1. Define a file `Dockerfile` 

    ```
    FROM pytorch/pytorch:latest

    ADD localfolder_with_code /opt/mycode


    RUN  cd /opt/mycode && pip install -e . # or pip install requirements

    # Install the required Python packages
    RUN pip install \
        numpy \
        sympy \
        scikit-learn \
        numba \
        opt_einsum \
        h5py \
        cytoolz \
        tensorboardx \
        seaborn \
        rich \
        pytorch-lightning==1.7

    or 
    ADD requirements.txt 
    pip install -r requirements.txt

    ```

2. Build the image

    ```bash
    docker build -t username/pytorch-condor-gpu:tag .
    ```
    
    and push it (after having setup the credentials with `docker login hub.docker.com`)
    
    ```bash
    docker push username/pytorch-condor-gpu:tag
    ```
    
3. Setup the condor job with a submission file `submitfile` as:

    ```bash
    universe                = docker
    docker_image            = user/pytorch-condor-gpu:tag
    executable              = job.sh
    when_to_transfer_output = ON_EXIT
    output                  = $(ClusterId).$(ProcId).out
    error                   = $(ClusterId).$(ProcId).err
    log                     = $(ClusterId).$(ProcId).log
    request_gpus            = 1
    request_cpus            = 2
    +Requirements           = OpSysAndVer =?= "CentOS7"
    +JobFlavour = espresso
    queue 1
    ```
    
    
4. For testing purpose one can start a job interactively and debug

    ```bash
    condor_submit -interactive submitfile
    ```
    
    
