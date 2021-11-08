## Machine Learning as a Service for HEP

MLaaS for HEP is a set of Python-based modules to support reading HEP data and
stream them to the ML tool of the user's choice. It consists of three independent layers:
- Data Streaming layer to handle remote data, see [reader.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/reader.py)
- Data Training layer to train ML model for given HEP data, see [workflow.py](https://github.com/vkuznet/MLaaS4HEP/blob/master/src/python/MLaaS4HEP/workflow.py)
- Data Inference layer, see [tfaas_client.py](https://github.com/vkuznet/TFaaS/blob/master/src/python/tfaas_client.py)

The MLaaS4HEP resopitory can be found [here](https://github.com/vkuznet/MLaaS4HEP).

The general architecture of MLaaS4HEP looks like this:
![MLaaS4HEP-architecture](https://github.com/vkuznet/MLaaS4HEP/blob/master/images/MLaaS4HEP_arch_root_white_bkg.png)

Even though this architecture was originally developed for dealing with
HEP ROOT files, we extend it to other data formats. As of right now, following
data formats are supported: JSON, CSV, Parquet, and ROOT. All of the formats 
support reading files from the local file system or HDFS, while the 
ROOT format supports reading files via the [XRootD](https://xrootd.slac.stanford.edu) protocol.

The pre-trained models can be easily uploaded to
[TFaaS](https://github.com/vkuznet/TFaaS) inference server for serving them to clients.
The TFaaS documentation can be found [here](https://github.com/cms-ml/documentation/blob/master/content/inference/tfaas.md).

### Dependencies
Here is a list of the dependencies:
- [pyarrow](https://arrow.apache.org) for reading data from HDFS file system
- [uproot](https://github.com/scikit-hep/uproot) for reading ROOT files
- [numpy](https://www.numpy.org), [pandas](https://pandas.pydata.org) for data representation
- [modin](https://github.com/modin-project/modin) for fast panda support
- [numba](https://numba.pydata.org) for speeing up individual functions

### Installation
The easiest way to install and run [MLaaS4HEP](https://cloud.docker.com/u/veknet/repository/docker/veknet/mlaas4hep) and [TFaaS](https://cloud.docker.com/u/veknet/repository/docker/veknet/tfaas) is to use pre-build docker images
```
# run MLaaS4HEP docker container
docker run veknet/mlaas4hep
# run TFaaS docker container
docker run veknet/tfaas
```

### Reading ROOT files
MLaaS4HEP python repository provides the `reader.py` module that defines a DataReader class able to read either local or remote ROOT files (via xrootd) in chunks. It is based on the
[uproot](https://github.com/scikit-hep/uproot) framework.

Basic usage
```
# setup the proper environment, e.g. 
# export PYTHONPATH=/path/src/python # path to MLaaS4HEP python framework
# export PATH=/path/bin:$PATH # path to MLaaS4HEP binaries

# get help and option description
reader --help

# here is a concrete example of reading local ROOT file:
reader --fin=/opt/cms/data/Tau_Run2017F-31Mar2018-v1_NANOAOD.root --info --verbose=1 --nevts=2000

# here is an example of reading remote ROOT file:
reader --fin=root://cms-xrd-global.cern.ch//store/data/Run2017F/Tau/NANOAOD/31Mar2018-v1/20000/6C6F7EAE-7880-E811-82C1-008CFA165F28.root --verbose=1 --nevts=2000 --info

# both of aforementioned commands produce the following output
Reading root://cms-xrd-global.cern.ch//store/data/Run2017F/Tau/NANOAOD/31Mar2018-v1/20000/6C6F7EAE-7880-E811-82C1-008CFA165F28.root
# 1000 entries, 883 branches, 4.113945007324219 MB, 0.6002757549285889 sec, 6.853425235896175 MB/sec, 1.6659010326328503 kHz
# 1000 entries, 883 branches, 4.067909240722656 MB, 1.3497390747070312 sec, 3.0138486148558896 MB/sec, 0.740883937302516 kHz
###total time elapsed for reading + specs computing: 2.2570559978485107 sec; number of chunks 2
###total time elapsed for reading: 1.9500117301940918 sec; number of chunks 2

--- first pass: 1131872 events, (648-flat, 232-jagged) branches, 2463 attrs
VMEM used: 29.896704 (MB) SWAP used: 0.0 (MB)
<__main__.RootDataReader object at 0x7fb0cdfe4a00> init is complete in 2.265552043914795 sec
Number of events  : 1131872
# flat branches   : 648
CaloMET_phi values in [-3.140625, 3.13671875] range, dim=N/A
CaloMET_pt values in [0.783203125, 257.75] range, dim=N/A
CaloMET_sumEt values in [820.0, 3790.0] range, dim=N/A
```

More examples about using uproot may be found [here](https://github.com/jpivarski/jupyter-talks/blob/master/2017-10-13-lpc-testdrive/uproot-introduction-evaluated.ipynb) and [here](https://github.com/jpivarski/jupyter-talks/blob/master/2017-10-13-lpc-testdrive/nested-structures-evaluated.ipynb).

### How to train ML models on HEP ROOT data
The training phase is managed by the `workflow.py` module which performs the following actions:
- read all input ROOT files in chunks to compute a specs file
- perform the training cycle (each time using a new chunk of events)
  - create a new chunk of events taken proportionally from the input ROOT files
    - extract and convert each event in a list of NumPy arrays
    - normalize the events
    - fix the Jagged Arrays dimension
    - create the masking vector
  - use the chunk to train the ML model provided by the user

A schematic representation of the steps performed in the MLaaS4HEP pipeline, in particular those inside the Data Streaming and Data Training layers, is:
![MLaaS4HEP-workflow](https://github.com/vkuznet/MLaaS4HEP/blob/master/images/mlaas4hep_workflow.png)

This training procedure can be applied to a variety of use-cases but it should not be viewed as the only way to train data sets using the MLaaS4HEP framework. We left to the end user the final choice of ML strategy for concrete use-cases, where appropriate steps should be taken to check the convergence of the model, a proper set of metrics to monitor the training cycle, etc. If some ML algorithms don’t benefit much from splitting the dataset in chunks, the user has the way to  customize the chunk size. In extreme cases the user can always fix the chunk size equal to the total number of events, restoring in this way the possibility to use the entire dataset in one shot for the training process.

For instance, when a data set does not fit into the RAM of the training node other solutions can be adopted, e.g., using an SGD model. In such case, the ML training workflow should be adapted to use the entire data set during each epoch.

A detailed description of how to use the `workflow.py` module for training a ML model reading ROOT files from the opendata portal, can be found [here](https://github.com/vkuznet/MLaaS4HEP/blob/master/doc/workflow_recipe.md).

For a complete description of MLaaS4HEP see [this](https://link.springer.com/content/pdf/10.1007/s41781-021-00061-3.pdf) paper.

