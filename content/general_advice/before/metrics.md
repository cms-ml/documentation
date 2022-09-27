## Metric

> **Metric** is a function which evaluates model's performance given true labels and model predictions for a particular data set. 

That makes it an important ingredient in the model training as being a measure of the model's quality. However, metrics as estimators can be sensitive to some effects (e.g. class imbalance) and provide biased or over/underoptimistic results. Additionally, they might not be relevant to a physical problem in mind and to the undestanding of what is a "good" model[^1]. This in turn can result in suboptimally tuned hyperparameters or in general to suboptimally trained model.

Therefore, it is important to choose metrics wisely, so that they reflect the physical problem to be solved and additionaly don't introduce any biases in the performance estimate. The whole topic of metrics would be too broad to get covered in this section, so please refer to [a corresponding documentation](https://scikit-learn.org/stable/modules/model_evaluation.html) of `sklearn` as it provides an exhaustive list of available metrics with additional materials and can be used as a good starting point.  

??? example "Examples of HEP-specific metrics"
    Speaking of those metrics which were developed in the HEP field, the most prominent one is _approximate median significance_ (AMS), firstly introduced in [Asymptotic formulae for likelihood-based tests of new physics](https://arxiv.org/abs/1007.1727) and then adopted in the [HiggsML challenge](https://www.kaggle.com/c/higgs-boson) on Kaggle.

    Essentially being an estimate of the expected signal sensitivity and hence being closely related to the final result of analysis, it can also be used not only as a metric but also as a loss function to be [directly optimised in the training](https://arxiv.org/abs/1806.00322).

## Loss function

In fact, metrics and loss functions are very similar to each other: they both give an estimate of how well (or bad) model performs and both used to monitor the quality of the model. So the same comments as in the metrics section apply to loss functions too. However, loss function plays a crucial role because it is additionally used in the training as a functional to be optimised. That makes its choice a handle to explicitly steer the training process towards a more optimal and relevant solution.

??? warning "Example of things going wrong"
    It is known that L2 loss (MSE) is sensitive to outliers in data and L1 loss (MAE) on the other hand is robust to them. Therefore, if outliers were overlooked in the training data set and the model was fitted, it may result in significant bias in its predictions. As an illustration, [this toy example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_huber_vs_ridge.html) compares Huber vs Ridge regressors, where the latter shows a more robust behaviour.

A simple example of that was already mentioned in [domains section](domains.md) - namely, one can emphasise specific regions in the phase space by attributing events there a larger weight in the loss function. Intuitively, for the same fraction of mispredicted events in the training data set, the class with a larger attributed weight should bring more penalty to the loss function. This way model should be able to learn to pay more attention to those "upweighted" events[^2].

??? example "Examples in HEP beyond classical MSE/MAE/cross entropy"
    * [b-jet energy regression](https://arxiv.org/abs/1912.06046), being a part of [nonresonant HH to bb gamma gamma](https://arxiv.org/abs/2011.12373) analysis, uses _Huber and two quantile loss terms_ for simultaneous prediction of point and dispersion estimators of the target disstribution.  
    * [DeepTau](https://cms.cern.ch/iCMS/analysisadmin/cadilines?line=TAU-20-001&tp=an&id=2333&ancode=TAU-20-001), a CMS deployed model for tau identification, uses several _focal loss_ terms to give higher weight to more misclassified cases

However, one can go further than that and consider the training procedure from a larger, statistical inference perspective. From there, one can try to construct a loss function which would _directly_ optimise the end goal of the analysis. [INFERNO](https://github.com/GilesStrong/pytorch_inferno) is an example of such an approach, with a loss function being an expected uncertainty on the parameter of interest. Moreover, one can try also to make the model aware of nuisance parameters which affect the analysis by incorporating those into the training procedure, please see [this review](https://arxiv.org/abs/2007.09121) for a comprehensive overview of the corresponding methods.    

[^1]: For example, that corresponds to asking oneself a question: "what is more suitable for the purpose of the analysis: F1-score, accuracy, recall or ROC AUC?"
[^2]: However, these are expectations one may have in theory. In practise, optimisation procedure depends on many variables and can go in different ways. Therefore, the weighting scheme should be studied by running experiments on the case-by-case basis.
