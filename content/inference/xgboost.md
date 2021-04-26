# Direct inference with XGBoost

## General 
XGBoost is avaliable (at least) since CMSSW\_9\_2\_4 ([cmssw#19377])(https://github.com/cms-sw/cmssw/pull/19377).

In CMSSW environment, XGBoost can be used via its ([Python API])(https://xgboost.readthedocs.io/en/latest/python/python_api.html).

The Current version is **x.x.x** and the GPU support is **status**.

## Existing Examples

There are some existing good examples of using XGBoost under CMSSW, as listed below:
1. ([Offical sample])(https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/PhysicsTools/PythonAnalysis/test/testXGBoost_and_sklearn.py) for testing the integration of XGBoost library with CMSSW.
2. ([Useful codes])(https://github.com/hqucms/NanoHRT-tools/blob/master/python/helpers/xgbHelper.py) created by Dr. Huilin Qu.
3. ?([C++ Interface])(https://github.com/simonepigazzini/XGBoostCMSSW)

We will provide a simple example to show the basic usage of XGBoost's python interface.
