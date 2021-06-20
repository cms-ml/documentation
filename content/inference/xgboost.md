# Direct inference with XGBoost

## General 
XGBoost is avaliable (at least) since CMSSW\_9\_2\_4 [cmssw#19377](https://github.com/cms-sw/cmssw/pull/19377).

In CMSSW environment, XGBoost can be used via its [Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html).

For UL era, there are different verisons available for different `SCRAM_ARCH`:

1. For `slc7_amd64_gcc700` and above, *ver.0.80* is available.

2. For `slc7_amd64_gcc900` and above, *ver.1.3.3* is available.

3. Please note that different major versions have different behavior( See [Caveat](#caveat) Session).


### Existing Examples

There are some existing good examples of using XGBoost under CMSSW, as listed below:

1. [Offical sample](https://github.com/cms-sw/cmssw/blob/6d2f66057131baacc2fcbdd203588c41c885b42c/PhysicsTools/PythonAnalysis/test/testXGBoost_and_sklearn.py) for testing the integration of XGBoost library with CMSSW.
   
2. [Useful codes](https://github.com/hqucms/NanoHRT-tools/blob/master/python/helpers/xgbHelper.py) created by Dr. Huilin Qu for inference with existing trained model.
   
3. [C/C++ Interface](https://github.com/simonepigazzini/XGBoostCMSSW) for inference with existing trained model.

We will provide examples for both C/C++ interface and python interface of XGBoost under CMSSW environment.

## Example: Classification of points from joint-Gaussian distribution.

In this specific example, you will use XGBoost to classify data points generated from two 8-dimension joint-Gaussian distribution. 

| Feature Index  | 0 | 1 | 2 | 3 | 4 | 5  | 6  | 7 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|   μ~1~    |   1   |   2   |   3   |   4   |   5   |   6   |   7   |    8  |
|   μ~2~   |   0   |   1.9   |   3.2   |   4.5   |    4.8  |   6.1   |    8.1  |   11   |
|   σ~1/2~ =  σ  |   1   |   1   |   1   |   1   |   1   |   1   |  1    |   1   |
|   \|μ~1~ - μ~2~\| / σ   |   1   |   0.1   |   0.2   |   0.5   |   0.2   |   0.1   |    1.1  |    3  |

All generated data points for train(1:10000,2:10000) and test(1:1000,2:1000) are stored as [`Train_data.csv`](code/XGBoost/Train_data.csv)/[`Test_data.csv`](code/XGBoost/Test_data.csv).
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

train_data = pd.read_csv("path/to/the/data") # The training dataset is code/XGBoost/Train_data.csv

train_Variable = train_data['0', '1', '2', '3', '4', '5', '6', '7']
train_Score = train_data['Type']

test_data = pd.read_csv("path/to/the/data") # The testing dataset is code/XGBoost/Test_data.csv

test_Variable = test_data['0', '1', '2', '3', '4', '5', '6', '7']
test_Score = test_data['Type']

# Now the data are well prepared and named as train_Variable, train_Score and test_Variable, test_Score.

xgb.fit(train_Variable, train_Score) # Training

xgb.predict(test_Variable) # Infering

xgb.save_model("\Path\To\Where\You\Want\ModelName.model") # Saving model
```
The saved model `ModelName.model` is thus available for python and C/C++ api to load.

While training with data from different datasets, proper treatment of weights are necessary for better model performance. Please refer to [Official Recommendation](https://xxxx) for more details.

## C/C++ Usage with CMSSW
To use a saved XGBoost model with C/C++ code, it is convenient to use the [`XGBoost's offical C api`](https://xgboost.readthedocs.io/en/stable/dev/c__api_8h.html). Here we provide a simple example as following.

### Module setup

There is no official CMSSW interface for XGBoost while its library are placed in `cvmfs` of CMSSW. Thus we have to use the raw `c_api` as well as setting up the library manually.

   1. To run XGBoost's `c_api` within CMSSW framework, in addition to the following standard setup.
```bash
export SCRAM_ARCH="slc7_amd64_gcc700" # To use higher version, please switch to slc7_amd64_900
export CMSSW_VERSION="CMSSW_X_Y_Z"

source /cvmfs/cms.cern.ch/cmsset_default.sh

cmsrel "$CMSSW_VERSION"
cd "$CMSSW_VERSION/src"

cmsenv
scram b
```
The addtional effort is to add corresponding xml file(s) to `$CMSSW_BASE/toolbox$CMSSW_BASE/config/toolbox/$SCRAM_ARCH/tools/selected/` for setting up XGBoost.

> 1. For lower version (<1), add **two** xml files as below.
>> `xgboost.xml`
>>```xml
>> <tool name="xgboost" version="0.80">
>> <lib name="xgboost"/>
>> <client>
>>    <environment name="LIBDIR" default="/cvmfs/cms.cern.ch/$SCRAM_ARCH/external/py2-xgboost/0.80-ikaegh/lib/python2.7/site-packages/xgboost/lib"/>
>>    <environment name="INCLUDE" default="/cvmfs/cms.cern.ch/$SCRAM_ARCH/external/py2-xgboost/0.80-ikaegh/lib/python2.7/site-packages/xgboost/include/"/>
>>  </client>
>>  <runtime name="ROOT_INCLUDE_PATH" value="$INCLUDE" type="path"/>
>>  <runtime name="PATH" value="$INCLUDE" type="path"/>
>>  <use name="rabit"/>
>></tool>
>>```
>>`rabit.xml`
>>```xml
>> <tool name="rabit" version="0.80">
>>   <client>
>>     <environment name="INCLUDE" default="/cvmfs/cms.cern.ch/$SCRAM_ARCH/external/py2-xgboost/0.80-ikaegh/lib/python2.7/site-packages/xgboost/rabit/include/"/>
>>   </client>
>>   <runtime name="ROOT_INCLUDE_PATH" value="$INCLUDE" type="path"/>
>>   <runtime name="PATH" value="$INCLUDE" type="path"/>  
>> </tool>
>>```
>Please note that the path in `cvmfs` is not fixed, one can list all available versions in the `py2-xgboost` directory and choose one to use.
>
> 2. For higher version (>=1), and **one** xml file
>> `xgboost.xml`
>>```xml
>><tool name="xgboost" version="0.80">
>>  <lib name="xgboost"/>
>>  <client>
>>    <environment name="LIBDIR" default="/cvmfs/cms.cern.ch/$SCRAM_ARCH/external/xgboost/1.3.3/lib64"/>
>>    <environment name="INCLUDE" default="/cvmfs/cms.cern.ch/$SCRAM_ARCH/external/xgboost/1.3.3/include/"/>
>>  </client>
>>  <runtime name="ROOT_INCLUDE_PATH" value="$INCLUDE" type="path"/>
>>  <runtime name="PATH" value="$INCLUDE" type="path"/>  
>></tool>
>>```
> Also one has the freedom to choose the available xgboost version inside `xgboost` directory.

2. After adding xml file(s), the following commands should be executed for setting up.
> 1. For lower version (<1), use 
>```shell
>scram setup rabit
>scram setup xgboost
>```
>2. For higher version (>=1), use
>```shell
>scram setup xgboost
>```

3. For using XGBoost as a plugin of CMSSW, it is necessary to add
```xml
<use name="xgboost"/>
<export>
  <lib name="1"/>
</export>
```
in your `plugins/BuildFile.xml`. If you are using the interface inside the `src/` or `interface/` directory of your module, make sure to create a global `BuildFile.xml` file next to theses directories, containing (at least):
```xml
<use name="xgboost"/>
<export>
  <lib   name="1"/>
</export>
```
   
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

## Caveat

It is worth mentioning that both behavior and APIs of different XGBoost version can have difference. 

1. When using `c_api` for C/C++ inference, for **ver.<1**, the API is `XGBoosterPredict(booster_, dvalues, 0, 0, &out_len, &score)`, while for **ver.>=1** the API changes to 
`XGBoosterPredict(booster_, dvalues, 0, 0, 0, &out_len, &score)`.

2. Model from **ver.>=1** **cannot be used** for **ver.<1**.
## Appendix: Tips for XGBoost users

### Importance Plot

### ROC Curve and AUC
