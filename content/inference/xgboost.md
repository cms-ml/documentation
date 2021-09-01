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
from xgboost import XGBClassifier # Or XGBRegressor for Logistic Regression
import matplotlib.pyplot as plt
import pandas as pd

# specify parameters via map
param = {'n_estimators':50}
xgb = XGBClassifier(param)

# using Pandas.DataFrame data-format, other available format are XGBoost's DMatrix and numpy.ndarray

train_data = pd.read_csv("path/to/the/data") # The training dataset is code/XGBoost/Train_data.csv

train_Variable = train_data['0', '1', '2', '3', '4', '5', '6', '7']
train_Score = train_data['Type'] # Score should be integer, 0, 1, (2 and larger for multiclass)

test_data = pd.read_csv("path/to/the/data") # The testing dataset is code/XGBoost/Test_data.csv

test_Variable = test_data['0', '1', '2', '3', '4', '5', '6', '7']
test_Score = test_data['Type']

# Now the data are well prepared and named as train_Variable, train_Score and test_Variable, test_Score.

xgb.fit(train_Variable, train_Score) # Training

xgb.predict(test_Variable) # Outputs are integers

xgb.predict_proba(test_Variable) # Output scores , output structre: [prob for 0, prob for 1,...]

xgb.save_model("\Path\To\Where\You\Want\ModelName.model") # Saving model
```
The saved model `ModelName.model` is thus available for python and C/C++ api to load. Please use the XGBoost major version consistently (see [Caveat](#caveat)).

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
<flags EDM_PLUGIN="1"/>
```
in your `plugins/BuildFile.xml`. If you are using the interface inside the `src/` or `interface/` directory of your module, make sure to create a global `BuildFile.xml` file next to theses directories, containing (at least):
```xml
<use name="xgboost"/>
<export>
  <lib   name="1"/>
</export>
```

4. The `libxgboost.so` would be too large to load for `cmsRun` job, please using the following commands for pre-loading:
```shell
export LD_PRELOAD=$CMSSW_BASE/external/$SCRAM_ARCH/lib/libxgboost.so
```
### Basic Usage of C API
In order to use `c_api` of XGBoost to load model and operate inference, one should construct necessaries objects:

1. Files to include
```c
#include <xgboost/c_api.h> 
```

2. `BoosterHandle`: worker of XGBoost
```c
// Declare Object
BoosterHandle booster_;
// Allocate memory in C style
XGBoosterCreate(NULL,0,&booster_);
// Load Model
XGBoosterLoadModel(booster_,model_path.c_str()); // second argument should be a const char *.
```

3. `DMatrixHandle`: handle to dmatrix, the data format of XGBoost
```c
float TestData[2000][8] // Suppose 2000 data points, each data point has 8 dimension
// Assign data to the "TestData" 2d array ... 
// Declare object
DMatrixHandle data_;
// Allocate memory and use external float array to initialize
XGDMatrixCreateFromMat((float *)TestData,2000,8,-1,&data_); // The first argument takes in float * namely 1d float array only, 2nd & 3rd: shape of input, 4th: value to replace missing ones
```

4. `XGBoosterPredict`: function for inference
```c
bst_ulong outlen; // bst_ulong is a typedef of unsigned long
const float *f; // array to store predictions
XGBoosterPredict(booster_,data_,0,0,&out_len,&f);// lower version API
// XGBoosterPredict(booster_,data_,0,0,0,&out_len,&f);// higher version API
/*
lower version (ver.<1) API
XGB_DLL int XGBoosterPredict(	
BoosterHandle 	handle,
DMatrixHandle 	dmat,
int 	option_mask, // 0 for normal output, namely reporting scores
int 	training, // 0 for prediction
bst_ulong * 	out_len,
const float ** 	out_result 
)

higher version (ver.>=1) API
XGB_DLL int XGBoosterPredict(	
BoosterHandle 	handle,
DMatrixHandle 	dmat,
int 	option_mask, // 0 for normal output, namely reporting scores
int ntree_limit, // how many trees for prediction, set to 0 means no limit
int 	training, // 0 for prediction
bst_ulong * 	out_len,
const float ** 	out_result 
)
*/
```

### Full Example

??? hint "Click to expand full example"

    The example assumes the following directory structure:

    ```
    MySubsystem/MyModule/
    │
    ├── plugins/
    │   ├── XGBoostExample.cc
    │   └── BuildFile.xml
    │
    ├── python/
    │   └── xgboost_cfg.py
    │
    ├── toolbox/ (storing necessary xml(s) to be copied to toolbox/ of $CMSSW_BASE)
    │   └── xgboost.xml
    │   └── rabit.xml (lower version only)
    │
    └── data/
        └── Test_data.csv
        └── lowVer.model / highVer.model 
    ```
    Please also note that in order to operate inference in an event-by-event way, please put `XGBoosterPredict` in `analyze` rather than `beginJob`.

    === "plugins/XGBoostExample.cc for lower version XGBoost"

        ```cpp linenums="1" hl_lines="2"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Lower/XGBoostExample/plugins/XGBoostExample.cc"
        ```

    === "plugins/BuildFile.xml for lower version XGBoost"

        ```xml linenums="1"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Lower/XGBoostExample/plugins/BuildFile.xml"
        ```

    === "python/xgboost_cfg.py for lower version XGBoost"

        ```python linenums="1"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Lower/XGBoostExample/python/xgboost_cfg.py"
        ```
    === "plugins/XGBoostExample.cc for higher version XGBoost"

        ```cpp linenums="1" hl_lines="2"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Higher/XGBoostExample/plugins/XGBoostExample.cc"
        ```

    === "plugins/BuildFile.xml for higher version XGBoost"

        ```xml linenums="1"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Higher/XGBoostExample/plugins/BuildFile.xml"
        ```

    === "python/xgboost_cfg.py for higher version XGBoost"

        ```python linenums="1"
        --8<-- "content/inference/code/XGBoost/XGB_Example_Higher/XGBoostExample/python/xgboost_cfg.py"
        ```
## Python Usage

To use XGBoost's python interface, using the snippet below **under CMSSW environment**
```python  
# importing necessary models
import numpy as np
import pandas as pd 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd


xgb = XGBClassifier()
xgb.load_model('ModelName.model')

# After loading model, usage is the same as discussed in the model preparation section.
```

## Caveat

It is worth mentioning that both behavior and APIs of different XGBoost version can have difference. 

1. When using `c_api` for C/C++ inference, for **ver.<1**, the API is `XGB_DLL int XGBoosterPredict(BoosterHandle 	handle, DMatrixHandle 	dmat,int 	option_mask, int 	training, bst_ulong * out_len,const float ** 	out_result)`, while for **ver.>=1** the API changes to 
`XGB_DLL int XGBoosterPredict(BoosterHandle 	handle, DMatrixHandle 	dmat,int 	option_mask, unsigned int ntree_limit, int 	training, bst_ulong * out_len,const float ** 	out_result)`.

2. Model from **ver.>=1** **cannot be used** for **ver.<1**.

Other important issue for C/C++ user is that DMatrix **only** takes in single precision floats (`float`), **not** double precision floats (`double`).

## Appendix: Tips for XGBoost users

### Importance Plot

XGBoost uses [F-score](https://en.wikipedia.org/wiki/F-score) to describe feature importance quantatitively. XGBoost's python API provides a nice tool,`plot_importance`, to plot the feature importance conveniently **after finishing train**. 

```python
# Once the training is done, the plot_importance function can thus be used to plot the feature importance.
from xgboost import plot_importance # Import the function

plot_importance(xgb) # suppose the xgboost object is named "xgb"
plt.savefig("importance_plot.pdf") # plot_importance is based on matplotlib, so the plot can be saved use plt.savefig()
```
![image](../images/inference/xgboost/importance_plot.png)
The importance plot is consistent with our expectation, as in our toy-model, the data points differ by most on the feature "7". (see [toy model setup](#example-classification-of-points-from-joint-gaussian-distribution)).
### ROC Curve and AUC
The [receiver operating characteristic (ROC)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and auccrency (AUC) are key quantities to describe the model performance. For XGBoost, ROC curve and auc score can be easily obtained with the help of [sci-kit learn (sklearn)](https://scikit-learn.org/) functionals, which is also in CMSSW software.
```python
from sklearn.metrics import roc_auc_score,roc_curve,auc
# ROC and AUC should be obtained on test set
# Suppose the ground truth is 'y_test', and the output score is named as 'y_score'

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# plt.show() # display the figure when not using jupyter display
plt.savefig("roc.png") # resulting plot is shown below
```
![image](../images/inference/xgboost/roc.png)

## Reference of XGBoost
1. XGBoost Wiki: https://en.wikipedia.org/wiki/XGBoost
2. XGBoost Github Repo.: https://github.com/dmlc/xgboost
3. XGBoost offical api tutorial
   1. Latest, Python: https://xgboost.readthedocs.io/en/latest/python/index.html
   2. Latest, C/C++: https://xgboost.readthedocs.io/en/latest/tutorials/c_api_tutorial.html
   3. Older (0.80), Python: https://xgboost.readthedocs.io/en/release_0.80/python/index.html
   4. No Tutorial for older version C/C++ api, source code: https://github.com/dmlc/xgboost/blob/release_0.80/src/c_api/c_api.cc

