# Introduction

// TODO: think about adding more illustrations and code examples here and there

In general, ML models don't really work out of the box: for example, mostly often it is not sufficient to simply instantiantiate the model class, call its `fit()` method followed by `predict()` and proceed straight to the inference step of the analysis:

// dummy code example

Being an extremely simplified and naive example, one would be lucky to have the code above producing valid and optimal physical results. This is because it explicitly doesn't check for those things which could've gone wrong and therefore is prone to producing undesirable results. Indeed, there are several pitfalls which one may encounter on the way towards implementation of ML into their analysis pipeline. These can be easily avoided by being aware of those and performing a few simple checks here and there.

Therefore this section is intended to overview potential issues on the ML side and how they can be approached in order to train a robust and optimal model. The section is designed to be to a large extent analysis-agnostic, focusing on general and most common validation steps from ML perspective, without paying particular emphasis on the physical context. However, for illustrative purposes it will be supplemented with some examples from HEP and additional links for further reading.  As the last remark, in the following there will mostly an emphasis on the validation items specific to _supervised learning_. This includes classification and regression problems as being the most common use cases amongst HEP analysts.

The section is divided into into 3 subsections. Things become logically aligned if presented from the perspective of the training step (i.e. fitting/loss minimisation part). That is, the subsections will group validation items as to be investigated:

* Before training
* Throughout training
* After training
