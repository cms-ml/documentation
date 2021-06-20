# Direct inference with XGBoost

## General 
XGBoost is avaliable (at least) since CMSSW\_9\_2\_4 ([cmssw#19377])(https://github.com/cms-sw/cmssw/pull/19377).

In CMSSW environment, XGBoost can be used via its ([Python API])(https://xgboost.readthedocs.io/en/latest/python/python_api.html).

The Current version is **x.x.x**.

### Existing Examples

There are some existing good examples of using XGBoost under CMSSW, as listed below:

1. [Offical sample](https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/PhysicsTools/PythonAnalysis/test/testXGBoost_and_sklearn.py) for testing the integration of XGBoost library with CMSSW.
   
2. [Useful codes](https://github.com/hqucms/NanoHRT-tools/blob/master/python/helpers/xgbHelper.py) created by Dr. Huilin Qu for inference with existing trained model.
   
3. [C/C++ Interface](https://github.com/simonepigazzini/XGBoostCMSSW) for inference with existing trained model.

We will provide examples for both C/C++ interface and python interface of XGBoost under CMSSW environment.

### General CMSSW Setup
To run both examples properly, please execute the shell command below:
```bash
export CMSSW_VERSION="CMSSW_X_Y_Z"

source /cvmfs/cms.cern.ch/cmsset_default.sh

cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"

cmsenv
scram b
```
It is not necessary to convert an XGBoost model into a [`cmsml`](https://github.com/cms-ml/cmsml) model. 

## Example: Classification of points from joint-Gaussian distribution.

In this specific example, you will use XGBoost to classify data points generated from two 8-dimension joint-Gaussian distribution. 

| Feature Index  | 0 | 1 | 2 | 3 | 4 | 5  | 6  | 7 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   μ~1~    |      |      |      |      |      |      |      |      |
|   μ~2~   |      |      |      |      |      |      |      |      |
|   σ~1/2~ =  σ  |      |      |      |      |      |      |      |      |
|   (μ~1~ - μ~2~) / σ   |      |      |      |      |      |      |      |      |


### Preparing Model
The training process of a XGBoost model can be done outside of CMSSW. We provide a python script for illustration. 
```python
# importing necessary models
import numpy as np
import pandas as pd 
from xgboost import XGBRegressor # Or XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd

# specify parameters via map
param = {'n_estimators':50}
xgb = XGBRegressor(param)

# using Pandas.DataFrame data-format, other available format are XGBoost's DMatrix and numpy.ndarray

train_data = pd.read_csv("path/to/the/data") # The training data is located in 

# supposing the data are well prepared and named as train_Variable, train_Score and test_Variable, test_Score.

xgb.fit(train_Variable, train_Score) # Training

xgb.predict(test_Variable) # Infering

xgb.save_model("\Path\To\Where\You\Want\ModelName.model") # Saving model
```
The saved model `ModelName.model` is thus available for python and C/C++ api to load.

While training with data from different datasets, proper treatment of weights are necessary for better model performance. Please refer to [Official Recommendation](https://xxxx) for more details.

## C/C++ Usage with CMSSW
To use a saved XGBoost model one with C/C++ code, it is convenient to use the [`XGBoost's offical C api`](https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html). Here we provide a simple example as following.

### Module setup

For using XGBoost as a plugin of CMSSW, it is necessary to add
```xml
<use name="xgboost"/>
<export>
  <lib   name="1"/>
</export>
```
in your `plugins/BuildFile.xml`. Unlike Tensorflow, the library of XGBoost is documented 

### Basic Usage of C API
The C API can be used as following:
```c
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>

#include <xgboost/c_api.h>

std::string model_file

vars_ = vars;
//---load the model
XGBoosterCreate(NULL, 0, &booster_);
XGBoosterLoadModel(booster_, model_file.c_str());

float values[1][vars_->size()];

int ivar=0;
for(auto& var : *vars_)
{
    values[0][ivar] = std::get<1>(var);
    ++ivar;
}
//---preparing data
DMatrixHandle dvalues;
XGDMatrixCreateFromMat(reinterpret_cast<float*>(values), 1, vars_->size(), 0., &dvalues);

bst_ulong out_len=0;
const float* score;

auto ret = XGBoosterPredict(booster_, dvalues, 0, 0, &out_len, &score);

XGDMatrixFree(dvalues);
//---load the model
std::vector<float> results;
if(ret==0)
{
    for(unsigned int ic=0; ic<out_len; ++ic)
        results.push_back(score[ic]);
} 
    
```

## Python Usage

To import XGBoost's python interface, using the snippet as
```python  
# importing necessary models
import numpy as np
import pandas as pd 
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd


xgb = XGBRegressor()
xgb.load_model('ModelName.model')

xgb.predict(Variable_to_use)

```

## Reference

## To be Added

? Plotting feature importance and trained BDT

? Interpreting dump model file

? Brief introduction to XGBoost

Test
