# Feature Importance
Feature importance is the impact a specific input field has on a prediction model's output. In general, these impacts can range from no impact (i.e. a feature with no variance) to perfect correlation with the ouput. There are several reasons to consider feature importance: 

  - Important features can be used to create simplified models, e.g. to mitigate overfitting.
  - Using only important features can reduce the latency and memory requirements of the model. 
  - The relative importance of a set of features can yield insight into the nature of an otherwise opaque model (improved interpretability). 
  - If a model is sensitive to noise, rejecting irrelevant inputs may improve its performance. 

In the following subsections, we detail several strategies for evaluating feature importance. We begin with a general discussion of feature importance at a high level before offering a code-based tutorial on some common techniques. We conclude with additional notes and comments in the last section. 

## General Discussion
Most feature importance methods fall into one of three broad categories: filter methods, embedding methods, and wrapper methods. Here we give a brief overview of each category with relevant examples: 

### Filter Methods
Filter methods do not rely on a specific model, instead considering features in the context of a given dataset. In this way, they may be considered to be pre-processing steps. In many cases, the goal of feature filtering is to reduce high dimensional data. However, these methods are also applicable to data exploration, wherein an analyst simply seeks to learn about a dataset without actually removing any features. This knowledge may help interpret the performance of a downstream predictive model. Relevant examples include, 

- **Domain Knowledge**:
Perhaps the most obvious strategy is to select features relevant to the domain of interest. 

- **Variance Thresholding**:
One basic filtering strategy is to simply remove features with low variance. In the extreme case, features with zero variance do not vary from example to example, and will therefore have no impact on the model's final prediction. Likewise, features with variance below a given threshold may not affect a model's downstream performance.

- **Fisher Scoring**:
Fisher scoring can be used to rank features; the analyst would then select the highest scoring features as inputs to a subsequent model. 

- **Correlations**:
Correlated features introduce a certain degree of redundancy to a dataset, so reducing the number of strongly correlated variables may not impact a model's downstream performance. 

### Embedded Methods
Embedded methods are specific to a prediction model and independent of the dataset. Examples:

- **L1 Regularization (LASSO)**:
L1 regularization directly penalizes large model weights. In the context of linear regression, for example, this amounts to enforcing sparsity in the output prediction; weights corresponding to less relevant features will be driven to 0, nullifying the feature's effect on the output. 

### Wrapper Methods
Wrapper methods iterate on prediction models in the context of a given dataset. In general they may be computationally expensive when compared to filter methods. Examples:

- **Permutation Importance**: Direct interpretation isn't always feasible, so other methods have been developed to inspect a feature's importance. One common and broadly-applicable method is to randomly shuffle a given feature's input values and test the degredation of model performance. This process allows us to measure [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html) as follows. First, fit a model ($f$) to training data, yielding $f(X_\mathrm{train})$, where $X_\mathrm{train}\in\mathbb{R}^{n\times d}$ for $n$ input examples with $d$ features. Next, measure the model's performance on testing data for some loss $\mathcal{L}$, i.e. $s=\mathcal{L}\big(f(X_\mathrm{test}), y_\mathrm{test}\big)$. For each feature $j\in[1\ ..\ d]$, randomly shuffle the corresponding column in $X_\mathrm{test}$ to form $X_\mathrm{test}^{(j)}$. Repeat this process $K$ times, so that for $k\in [1\ ..\ K]$ each random shuffling of feature column $j$ gives a corrupted input dataset $X_\mathrm{test}^{(j,k)}$. Finally, define the permutation importance of feature $j$ as the difference between the un-corrupted validation score and average validation score over the corrupted $X_\mathrm{test}^{(j,k)}$ datasets: 

$$\texttt{PI}_j = s - \frac{1}{K}\sum_{k=1}^{K} \mathcal{L}[f(X_\mathrm{test}^{(j,k)}), y_\mathrm{test}]$$

- **Recursive Feature Elimination (RFE)**: Given a prediction model and test/train dataset splits with $D$ initial features, RFE returns the set of $d < D$ features that maximize model performance. First, the model is trained on the full set of features. The importance of each feature is ranked depending on the model type (e.g. for regression, the slopes are a sufficient ranking measure; permutation importance may also be used). The least important feature is rejected and the model is retrained. This process is repeated until the most significant $d$ features remain. 

## Introduction by Example

### Direct Interpretation
Linear regression is particularly interpretable because the prediction coefficients themselves can be interpreted as a measure of feature importance. Here we will compare this direct interpretation to several model inspection techniques. In the following examples we use the [Diabetes Dataset](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html) available as a [Scikit-learn toy dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset). This dataset maps 10 biological markers to a 1-dimensional quantitative measure of diabetes progression: 

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X_train, X_val, y_train, y_val = train_test_split(diabetes.data, diabetes.target, random_state=0)
print(X_train.shape)
>>> (331,10)
print(y_train.shape)
>>> (331,)
print(X_val.shape)
>>> (111, 10)
print(y_val.shape)
>>> (111,)
print(diabetes.feature_names)
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
```
To begin, let's use Ridge Regression (L2-regularized linear regression) to model diabetes progression as a function of the input markers. The absolute value of a regression coefficient (slope) corresponding to a feature can be interpreted the impact of a feature on the final fit:

```python
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE

model = Ridge(alpha=1e-2).fit(X_train, y_train)
print(f'Initial model score: {model.score(X_val, y_val):.3f}')

for i in np.argsort(-abs(model.coef_)):
    print(diabetes.feature_names[i], abs(model.coef_[i]))

>>> Initial model score: 0.357
>>> bmi: 592.253
>>> s5: 580.078
>>> bp: 297.258
>>> s1: 252.425
>>> sex: 203.436
>>> s3: 145.196
>>> s4: 97.033
>>> age: 39.103
>>> s6: 32.945
>>> s2: 20.906
```
These results indicate that the bmi and s5 fields have the largest impact on the output of this regression model, while age, s6, and s2 have the smallest. Further interpretation is subject to the nature of the input data (see [Common Pitfalls in the Interpretation of Coefficients of Linear Models](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py)). Note that scikit-learn has [tools](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel) available to faciliate feature selections. 

### Permutation Importance
In the context of our ridge regression example, we can calculate the permutation importance of each feature as follows (based on [scikit-learn docs](https://scikit-learn.org/stable/modules/permutation_importance.html])):

```python
from sklearn.inspection import permutation_importance

model = Ridge(alpha=1e-2).fit(X_train, y_train)
print(f'Initial model score: {model.score(X_val, y_val):.3f}')
 
r = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    print(f"{diabetes.feature_names[i]:<8}"
          f"{r.importances_mean[i]:.3f}"
          f" +/- {r.importances_std[i]:.3f}")

>>> Initial model score: 0.357
>>> s5      0.204 +/- 0.050
>>> bmi     0.176 +/- 0.048
>>> bp      0.088 +/- 0.033
>>> sex     0.056 +/- 0.023
>>> s1      0.042 +/- 0.031
>>> s4      0.003 +/- 0.008
>>> s6      0.003 +/- 0.003
>>> s3      0.002 +/- 0.013
>>> s2      0.002 +/- 0.003
>>> age     -0.002 +/- 0.004
```
These results are roughly consistent with the direct interpretation of the linear regression parameters; s5 and bmi are the most permutation-important features. This is because both have significant permutation importance scores (0.204, 0.176) when compared to the initial model score (0.357), meaning their random permutations significantly degraded the model perforamnce. On the other hand, s2 and age have approximately no permutation importance, meaning that the model's performance was robust to random permutations of these features. 

### L1-Enforced Sparsity
In some applications it may be useful to reject features with low importance. Models biased towards sparsity are one way to achieve this goal, as they are designed to ignore a subset of features with the least impact on the model's output. In the context of linear regression, sparsity can be enforced by imposing L1 regularization on the regression coefficients (LASSO regression):

$$\mathcal{L}_\mathrm{LASSO} = \frac{1}{2n}||y - Xw||^2_2 + \alpha||w||_1$$

Depending on the strength of the regularization $(\alpha)$, this loss function is biased to zero-out features of low importance. In our diabetes regression example, 

```python
model = Lasso(alpha=1e-1).fit(X_train, y_train)
print(f'Model score: {model.score(X_val, y_val):.3f}')

for i in np.argsort(-abs(model.coef_)):
    print(f'{diabetes.feature_names[i]}: {abs(model.coef_[i]):.3f}')

>>> Model score: 0.355
>>> bmi: 592.203
>>> s5: 507.363
>>> bp: 240.124
>>> s3: 219.104
>>> sex: 129.784
>>> s2: 47.628
>>> s1: 41.641
>>> age: 0.000
>>> s4: 0.000
>>> s6: 0.000
```
For this value of $\alpha$, we see that the model has rejected the age, s4, and s6 features as unimportant (consistent with the permutation importance measures above) while achieving a similar model score as the previous ridge regression strategy. 

### Recursive Feature Elimination
Another common strategy is recursive feature elimination (RFE). Though RFE can be used for regression applications as well, we turn our attention to a classification task for the sake of variety. The following discussions are based on the [Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+Diagnostic), which maps 30 numeric features corresponding to digitized breast mass images to a binary classification of benign or malignant. 

```python
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

data = load_breast_cancer()
X_train, X_val, y_train, y_val = train_test_split(data.data, data.target, random_state=0)
print(X_train.shape)
>>> (426, 30)
print(y_train.shape)
>>> (426,)
print(X_val.shape)
>>> (143, 30)
print(y_val.shape)
>>> (143,)
print(breast_cancer.feature_names)
>>> ['mean radius' 'mean texture' 'mean perimeter' 'mean area' 'mean smoothness' 'mean compactness' 'mean concavity' 'mean concave points' 'mean symmetry' 'mean fractal dimension' 'radius error' 'texture error' 'perimeter error' 'area error' 'smoothness error' 'compactness error' 'concavity error' 'concave points error' 'symmetry error' 'fractal dimension error' 'worst radius' 'worst texture' 'worst perimeter' 'worst area' 'worst smoothness' 'worst compactness' 'worst concavity' 'worst concave points' 'worst symmetry' 'worst fractal dimension']
```

Given a classifier and a classification task, recursive feature elimination (RFE, [see original paper](https://link.springer.com/content/pdf/10.1023/A:1012487302797.pdf)) is the process of identifying the subset of input features leading to the most performative model. Here we employ a support vector machine classifier (SVM) with a linear kernel to perform binary classification on the input data. We ask for the top $j\in[1\ .. \ d]$ most important features in a for loop, computing the classification accuracy when only these features are leveraged. 

```python
from sklearn.feature_selection import RFE

features = np.array(breast_cancer.feature_names)
svc = SVC(kernel='linear')
for n_features in np.arange(1, 30, 1):
    rfe = RFE(estimator=svc, step=1, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    print(f'n_features={n_features}, accuracy={rfe.score(X_val, y_val):.3f}')
    print(f' - selected: {features[rfe.support_]}')

>>> n_features=1, accuracy=0.881
>>>  - selected: ['worst concave points']
>>> n_features=2, accuracy=0.874
>>>  - selected: ['worst concavity' 'worst concave points']
>>> n_features=3, accuracy=0.867
>>>  - selected: ['mean concave points' 'worst concavity' 'worst concave points']
 ...
>>> n_features=16, accuracy=0.930
>>> n_features=17, accuracy=0.965
>>> n_features=18, accuracy=0.951
...
>>> n_features=27, accuracy=0.958
>>> n_features=28, accuracy=0.958
>>> n_features=29, accuracy=0.958
```
Here we've shown a subset of the output. In the first output lines, we see that the 'worst concave points' feature alone leads to 88.1% accuracy. Including the next two most important features actually degrades the classification accuracy. We then skip to the top 17 features, which in this case we observe to yield the best performance for the linear SVM classifier. The addition of more features does not lead to additional perforamnce boosts. In this way, RFE can be treated as a model wrapper introducing an additional hyperparameter, n_features_to_select, which can be used to optimize model performance. A more principled optimization using k-fold cross validation with RFE is available in the [scikit-learn docs](https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py). 

### Feature Correlations 
In the above, we have focused specifically on interpreting the importance of single features. However, it may be that several features are correlated, sharing the responsibility for the overall prediction of the model. In this case, some measures of feature importance may inappropriately downweight correlated features in a so-called correlation bias (see [Classification with Correlated Features: Unrelability of Feature Ranking and Solutions](https://pubmed.ncbi.nlm.nih.gov/21576180/)). For example, the permutation invariance of $d$ correlated features is shown to decrease (as a function of correlation strength) faster for higher $d$ (see [Correlation and Variable importance in Random Forests](https://link.springer.com/article/10.1007/s11222-016-9646-1)). 

We can see these effects in action using the breast cancer dataset, following the corresponding [scikit-learn example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

>>> Accuracy on test data: 0.97
```
Here we've implemented a random forest classifier and achieved a high accuracy (97%) on the benign vs. malignent predictions. The permutation importances for the 10 most important training features are:

```python
r = permutation_importance(clf, X_train, y_train, n_repeats=10, random_state=42)
for i in r.importances_mean.argsort()[::-1][:10]:
    print(f"{breast_cancer.feature_names[i]:<8}"
          f"  {r.importances_mean[i]:.5f}"
          f" +/- {r.importances_std[i]:.5f}")

>>> worst concave points  0.00681 +/- 0.00305
>>> mean concave points  0.00329 +/- 0.00188
>>> worst texture  0.00258 +/- 0.00070
>>> radius error  0.00235 +/- 0.00000
>>> mean texture  0.00188 +/- 0.00094
>>> mean compactness  0.00188 +/- 0.00094
>>> area error  0.00188 +/- 0.00094
>>> worst concavity  0.00164 +/- 0.00108
>>> mean radius  0.00141 +/- 0.00115
>>> compactness error  0.00141 +/- 0.00115
```

In this case, even the most permutation important features have mean importance scores $<0.007$, which doesn't indicate much importance. This is surprising, because we saw via RFE that a linear SVM can achieve $\approx 88\%$ classification accuracy with this feature alone. This indicates that worst concave points, in addition to other meaningful features, may belong to subclusters of correlated features. In the corresponding [scikit-learn example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py), the authors show that subsets of correlated features can be extracted by calculating a dendogram and selecting representative features from each correlated subset. They achieve $97\%$ accuracy (the same as with the full dataset) by selecting only five such representative variables. 

## Feature Importance in Decision Trees
Here we focus on decision trees, which are particularly interpretable classifiers that often appear as ensembles (or *boosted decision tree (BDT)* algorithms) in HEP. Consider a classification dataset $X=\{x_n\}_{n=1}^{N}$, $x_n\in\mathbb{R}^{D}$, with truth labels $Y=\{y_n\}_{n=1}^N$, $y_n\in\{1,...,C\}$ corresponding $C$ classes. These truth labels naturally partition $X$ into subsets $X_c$ with class probabilities $p(c)=|X_c|/|X|$. Decision trees begin with a root node $t_0$ containing all of $X$. The tree is grown from the root by recursively splitting the input set $X$ in a principled way; internal nodes (or branch nodes) correspond to a decision of the form 

$$\begin{aligned}
&(x_n)_d\leq\delta \implies\ \text{sample}\ n\ \text{goes to left child node}\\
&(x_n)_d>\delta \implies\ \text{sample}\ n\ \text{goes to right child node}
\end{aligned}$$

We emphasize that the decision boundary is drawn by considering a single feature field $d$ and partitioning the $n^\mathrm{th}$ sample by the value at that feature field. Decision boundaries at each internal parent node $t_P$ are formed by choosing a "split criterion," which describes how to partition the set of elements at this node into left and right child nodes $t_L$, $t_R$ with $X_{t_L}\subset X_{t_P}$ and $X_{t_R}\subset X_{t_P}$, $X_{t_L}\cup X_{t_R}=X_{t_P}$. This partitioning is optimal if $X_{t_L}$ and $X_{t_R}$ are pure, each containing only members of the same class. *Impurity measures* are used to evaluate the degree to which the set of data points at a given tree node $t$ are not pure. One common impurity measure is Gini Impurity, 

$$\begin{aligned} 
I(t) = \sum_{c=1}^C p(c|t)(1-p(c|t))
\end{aligned}$$

Here, $p(c|t)$ is the probability of drawing a member of class $c$ from the set of elements at node $t$. For example, the Gini impurity at the root node (corresponding to the whole dataset) is 

$$\begin{aligned} 
I(t_0) = \sum_{c=1}^C \frac{|X_c|}{|X|}(1-\frac{|X_c|}{|X|})
\end{aligned}$$

In a balanced binary dataset, this would give $I(t_0)=1/2$. If the set at node $t$ is pure, i.e. class labels corresponding to $X_t$ are identical, then $I(t)=0$. We can use $I(t)$ to produce an optimal splitting from parent $t_p$ to children $t_L$ and $t_R$ by defining an *impurity gain*, 

$$\begin{aligned}
\Delta I = I(t_P) - I(t_L) - I(t_R)
\end{aligned}$$

This quantity describes the relative impurity between a parent node and its children. If $X_{t_P}$ contains only two classes, an optimal splitting would separate them into $X_{p_L}$ and $X_{p_R}$, producing pure children nodes with $I(t_L)=I(t_R)=0$ and, correspondingly, $\Delta I(t_p) = I(t_P)$. Accordingly, good splitting decisions should maximize impurity gain. Note that the impurity gain is often weighted, for example Scikit-Learn defines:

$$\begin{aligned}
\Delta I(t_p) = \frac{|X_{t_p}|}{|X|}\bigg(I(t_p) - \frac{|X_{t_L}|}{|X_{t_p}|} I(t_L) - \frac{|X_{t_R}|}{|X_{t_p}|} I(t_R) \bigg)
\end{aligned}$$

In general, a pure node cannot be split further and must therefore be a leaf. Likewise, a node for which there is no splitting yielding $\Delta I > 0$ must be labeled a leaf. These splitting decisions are made recursively at each node in a tree until some stopping condition is met. Stopping conditions may include maximum tree depths or leaf node counts, or threshhold on the maximum impurity gain. 

Impurity gain gives us insight into the importance of a decision. In particular, larger $\Delta I$ indicates a more important decision. If some feature $(x_n)_d$ is the basis for several decision splits in a decision tree, the sum of impurity gains at these splits gives insight into the importance of this feature. Accordingly, one measure of the feature importance of $d$ is the average (with respect to the total number of internal nodes) impurity gain imparted by decision split on $d$. This method generalizes to the case of BDTs, in which case one would average this quantity across all weak learner trees in the ensemble. 

Note that though decision trees are based on the feature $d$ producing the best (maximum impurity gain) split at a given branch node, *surrogate splits* are often used to retain additional splits corresponding to features other than $d$. Denote the feature maximizing the impurity gain $d_1$ and producing a split boundary $\delta_1$. Surrogte splitting involves tracking secondary splits with boundaries $\delta_2, \delta_3,...$ corresponding to $d_2,d_3,...$ that have the highest correlation with the maximum impurity gain split. The upshot is that in the event that input data is missing a value at field $d_1$, there are backup decision boundaries to use, mitigating the need to define multiple trees for similar data. Using this generalized notion of a decision tree, wherein each branch node contains a primary decision boundary maximizing impurity gain and several additional surrogate split boundaries, we can average the impurity gain produced at feature field $d$ over all its occurances as a decision split or a surrogate split. This definition of feature importance generalizes the previous to include additional correlations. 

### Example
Let us now turn to an example: 
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

wine_data = load_wine() 
print(wine_data.data.shape)
print(wine_data.feature_names)
print(np.unique(wine_data.target))
>>> (178, 13)
>>> ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
>>> [0 1 2]
```

This sklearn wine dataset has 178 entries with 13 features and truth labels corresponding to membership in one of $C=3$ classes. We can train a decision tree classifier as follows: 

```python
X, y = wine_data.data, wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
classifier = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=27)
classifier.fit(X_train, y_train)
X_test_pred = classifier.predict(X_test)
print('Test Set Performance')
print('Number misclassified:', sum(X_test_pred!=y_test))
print(f'Accuracy: {classifier.score(X_test, y_test):.3f}')
>>> Test Set Performance
>>> Number misclassified: 0
>>> Accuracy: 1.000
```

In this case, the classifier has generalized perfectly, fitting the test set with $100\%$ accuracy. Let's take a look into how it makes predictions:

```python
tree = classifier.tree_
n_nodes = tree.node_count
node_features = tree.feature
thresholds = tree.threshold
children_L = tree.children_left
children_R = tree.children_right
feature_names = np.array(wine_data.feature_names)

print(f'The tree has {n_nodes} nodes')
for n in range(n_nodes):
    if children_L[n]==children_R[n]: continue # leaf node
    print(f'Decision split at node {n}:',
          f'{feature_names[node_features[n]]}({node_features[n]}) <=',
          f'{thresholds[n]:.2f}')

>>> The tree has 13 nodes
>>> Decision split at node 0: color_intensity(9) <= 3.46
>>> Decision split at node 2: od280/od315_of_diluted_wines(11) <= 2.48
>>> Decision split at node 3: flavanoids(6) <= 1.40
>>> Decision split at node 5: color_intensity(9) <= 7.18
>>> Decision split at node 8: proline(12) <= 724.50
>>> Decision split at node 9: malic_acid(1) <= 3.33
```

Here we see that several features are used to generate decision boundaries. For example, the dataset is split at the root node by a cut on the $\texttt{color_intensity}$ feature. The importance of each feature can be taken to be the average impurity gain it generates across all nodes, so we expect that one (or several) of the five unique features used at the decision splits will be the most important features by this definition. Indeed, we see,

```python
feature_names = np.array(wine_data.feature_names)
importances = classifier.feature_importances_
for i in range(len(importances)):
    print(f'{feature_names[i]}: {importances[i]:.3f}')
print('\nMost important features', 
      feature_names[np.argsort(importances)[-3:]])

>>> alcohol: 0.000
>>> malic_acid: 0.021
>>> ash: 0.000
>>> alcalinity_of_ash: 0.000
>>> magnesium: 0.000
>>> total_phenols: 0.000
>>> flavanoids: 0.028
>>> nonflavanoid_phenols: 0.000
>>> proanthocyanins: 0.000
>>> color_intensity: 0.363
>>> hue: 0.000
>>> od280/od315_of_diluted_wines: 0.424
>>> proline: 0.165

>>> Most important features ['proline' 'color_intensity' 'od280/od315_of_diluted_wines']
```

This is an embedded method for generating feature importance - it's cooked right into the decision tree model. Let's verify these results using a wrapper method, permutation importance:

```python
from sklearn.inspection import permutation_importance

print(f'Initial classifier score: {classifier.score(X_test, y_test):.3f}')

r = permutation_importance(classifier, X_test, y_test, n_repeats=30, random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    print(f"{feature_names[i]:<8}"
          f" {r.importances_mean[i]:.3f}"
          f" +/- {r.importances_std[i]:.3f}")

>>> Initial classifier score: 1.000

>>> color_intensity 0.266 +/- 0.040
>>> od280/od315_of_diluted_wines 0.237 +/- 0.049
>>> proline  0.210 +/- 0.041
>>> flavanoids 0.127 +/- 0.025
>>> malic_acid 0.004 +/- 0.008
>>> hue      0.000 +/- 0.000
>>> proanthocyanins 0.000 +/- 0.000
>>> nonflavanoid_phenols 0.000 +/- 0.000
>>> total_phenols 0.000 +/- 0.000
>>> magnesium 0.000 +/- 0.000
>>> alcalinity_of_ash 0.000 +/- 0.000
>>> ash      0.000 +/- 0.000
>>> alcohol  0.000 +/- 0.000
```

The tree's performance is hurt the most if the $\texttt{color_intensity}$, $\texttt{od280/od315_of_diluted_wines}$, or $\texttt{proline}$ features are permuted, consistent with the impurity gain measure of feature importance. 