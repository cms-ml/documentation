# lxplus condor

The `HTCondor` service @ `Lxplus` has GPU support. Therefore one can submit jobs on machine learning tasks to make use of this resources.

A complete documentation can be found from the `GPUs` section in [CERN Batch Docs](https://batchdocs.web.cern.ch/tutorial/exercise10.html). Where a `Tensorflow` example is supplied. For `pytorch` users, it is worthwhile to take a look at [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark). This repository not only contains tools for benchmarking, but also offers a good example on how to use `HTCondor` @ `Lxplus` to train `pytorch` ML models with `weaver`. Detailed explanation and tutorial on how to use [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark) can be found at section [`ParticleNet`](../../../inference/particlenet.md) of this documentation.

# HTCondor With GPU resources
In general, HTCondor supports GPU jobs if there are some GPU devices configured. For two main collaboration-wide HTCondor facilities, Lxplus and CMS Connect, they both have GPU resources equipped. 

## How to require GPUs in HTCondor

People can require their jobs to have GPU support by adding the following requirements to the condor submission file.

>```bash
>request_gpus = n # n equal to the number of GPUs required

## Further documentation

There are good materials providing detailed documentation on how to run HTCondor jobs with GPU support at both machines. 
1. A complete documentation can be found from the `GPUs` section in [CERN Batch Docs](https://batchdocs.web.cern.ch/tutorial/exercise10.html). Where a `Tensorflow` example is supplied. This documentation also contains instructions on advanced HTCondor configuration, for instance constraining GPU device or CUDA version.
2. A paradigm example on submitting GPU HTCondor job @ Lxplus is the [`weaver-benchmark`](https://github.com/colizz/weaver-benchmark) project. It provides a concrete example on how to setup environment for `weaver` framework and operate trainning and testing process within a single job. Detailed description can be found at section [`ParticleNet`](../../../inference/particlenet.md) of this documentation.
3. CMS Connect also provides a [documentation](https://ci-connect.atlassian.net/wiki/spaces/CMS/pages/80117822/Requesting+GPUs) on GPU job submission. In this documentation there is also a `Tensorflow` example. 
   
        💡 Note. When submitting GPU jobs @ CMS Connect, especially for Machine Learning purpose, EOS space @ CERN are not accessible as directory, therefore one should consider using `xrootd` service as documented in [this page](https://ci-connect.atlassian.net/wiki/spaces/CMS/pages/850264068/Using+stashcp+and+XrootD+from+OSG+TensorFlow+containers)
    