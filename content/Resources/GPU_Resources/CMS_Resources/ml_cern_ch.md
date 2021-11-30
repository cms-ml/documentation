# ml.cern.ch

[ml.cern.ch](https://ml.cern.ch) is a `Kubeflow` based ML solution provided by CERN. 

## `Kubeflow` 
[`Kubeflow`](https://www.kubeflow.org/docs/started/introduction/) is a Kubernetes based ML toolkits aiming at making deployments of ML workflows simple, portable and scalable. In Kubeflow, *pipeline* is an important concept. Machine Learning workflows are discribed as a Kubeflow *pipeline* for execution.

## How to access
[`ml.cern.ch`](https://ml.cern.ch) only accepts connection from CERN internet. Therefore, if you are outside of CERN a network tunneling will be needed (e.g. via `ssh -D` dynamical port forwarding as proxy). The main website are shown below.

![Untitled](./MLCERN_figs/Untitled.png)
## Examples
After logging into the main website, you can click on the `Examples` entry to browser a [gitlab repository](https://gitlab.cern.ch/ai-ml/examples) containing a lot of examples. For instance, below are two examples from that repository with a well-documented `readme` file.

1. [`mnist-kfp`](https://gitlab.cern.ch/ai-ml/examples/-/tree/master/mnist-kfp) is an example on how to use jupyter notebooks to create a Kubeflow pipeline (kfp) and how to access CERN EOS files.
2. [`katib`](https://gitlab.cern.ch/ai-ml/examples/-/tree/master/katib) gives an example on how to use the `katib` to operate hyperparameter tuning for jet tagging with ParticleNet.
