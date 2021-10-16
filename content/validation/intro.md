# Introduction

In general, ML models don't really work out of the box. For example, most often it is not sufficient to simply instantiate the model class, call its `fit()` method followed by `predict()`, and then proceed straight to the inference step of the analysis.

```python
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

clf = SVC(kernel="linear", C=0.025)
clf.fit(X_train, y_train)
print(f'Accuracy: {clf.score(X_test, y_test)}')
# Accuracy: 0.4
```

Being an extremely simplified and naive example, one would be lucky to have the code above produce a valid and optimal model. This is because it explicitly doesn't check for those things which could've gone wrong and therefore is prone to producing undesirable results. Indeed, there are several pitfalls which one may encounter on the way towards implementation of ML into their analysis pipeline. These can be easily avoided by being aware of those and performing a few simple checks here and there.

Therefore, this section is intended to **review potential issues on the ML side and how they can be approached** in order to train a robust and optimal model. The section is designed to be, to a large extent, analysis-agnostic. It will focus on common, generalized validation steps from ML perspective, without paying particular emphasis on the physical context. However, for illustrative purposes, it will be supplemented with some examples from HEP and additional links for further reading.  As the last remark, in the following there will mostly an emphasis on the validation items specific to _supervised learning_. This includes classification and regression problems as being so far the most common use cases amongst HEP analysts.

The validation chapter is divided into into 3 sections. Things become logically aligned if presented from the perspective of the training procedure (fitting/loss minimisation part). That is, the sections will group validation items as they need to be investigated:

* Before training
* Throughout training
* After training

---   

Authors: [Oleg Filatov](mailto:oleg.filatov@cern.ch)
