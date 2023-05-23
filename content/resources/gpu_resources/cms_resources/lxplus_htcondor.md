# HTCondor With GPU resources
In general, HTCondor supports GPU jobs if there are some worker nodes which are configured with GPU devices. CMS Connect and lxplus both have access to worker nodes equipped with GPUs.

## How to require GPUs in HTCondor

People can require their jobs to have GPU support by adding the following requirements to the condor submission file.

```bash
request_gpus = n # n equal to the number of GPUs required
```
## Further documentation

There are good materials providing detailed documentation on how to run HTCondor jobs with GPU support at both
machines. 


The configuration of the software environment for lxplus-gpu and HTcondor is described in the [Software
Environments](../../../software_envs/lcg_environments.md) page. Moreover the page [Using
container](../../../software_envs/containers.md) explains step by step how to build a docker image to be run on HTCondor
jobs. 

#### More available resources

1. A complete documentation can be found from the `GPUs` section in [CERN Batch Docs](https://batchdocs.web.cern.ch/tutorial/exercise10.html). Where a `Tensorflow` example is supplied. This documentation also contains instructions on advanced HTCondor configuration, for instance constraining GPU device or CUDA version.
2. A good example on submitting GPU HTCondor job @ Lxplus is the [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark) project. It provides a concrete example on how to setup environment for `weaver` framework and operate trainning and testing process within a single job. Detailed description can be found at section [`ParticleNet`](../../../inference/particlenet.md) of this documentation.  
>In principle, this example can be run elsewhere as HTCondor jobs. However, paths to the datasets should be modified to meet the requirements. 
    
3. CMS Connect also provides a [documentation](https://ci-connect.atlassian.net/wiki/spaces/CMS/pages/80117822/Requesting+GPUs) on GPU job submission. In this documentation there is also a `Tensorflow` example. 
>When submitting GPU jobs @ CMS Connect, especially for Machine Learning purpose, EOS space @ CERN are not accessible as a directory, therefore one should consider using `xrootd` utilities as documented in [this page](https://ci-connect.atlassian.net/wiki/spaces/CMS/pages/850264068/Using+stashcp+and+XrootD+from+OSG+TensorFlow+containers)
    
