---
Metrics
---

Various metrics exist in order to quantify the performance, and check the sanity of a NN. Below are the main ones.

### Training & Validation losses versus epoch

The training of the NN by definition minimizes the loss, so this latter should decrease versus the epochs for events used for training. So the calculated loss in the validation sample where the weights, as calculated for a given epoch, should have the effect of decreasing the loss in a sample non-exposed to the training. However, a training is never perfect and the loss will never reach nil. As a result, and after a number of epochs, the losses will reach and remain at a minimal and non-zero value. A typical good and realistic example of how training and validation losses should be is given in the plot below: both training and validation losses decrease and plateau after a while, and the validation loss is superior or equal to the training loss, as we do not expect the same weights to yield a better result in the validation sample than for the training sample itself.

> <img src="../../images/tutorials/nn_in_cms/TrnVal_loss.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 5

The following situations are cases where one should pay attention or pitfalls:

* <b>The training & validation losses are increasing after having reached a minimum</b>. This can happen when, ie. from an epoch, the NN is over-fitting the data. In such a case, collecting the weights of the NN at the end of the training will not be optimal, as they will correspond to a case where the loss function isn't minimal, resulting in an under-performing NN. In such a case, as illustrated above, one can/should collect the weights of the epoch where the validation loss is at its minimum.
* <b>Validation loss smaller than training loss</b>. There can be 3 reasons why this can happen.
      * Regularization is applied during training, but not during validation. Regularization leads to a set of weights which generalize better, ie NN results which are more stable for different samples, but also to slightly lower classification performance (eg. higher loss, lower accuracy).
      * Depending on the software used, the training loss can be calculated at the end of each batch and then averaged over all batches of a given epoch, while the validation loss after a full epoch. In such a case, the validation loss is calculated over the full validation set (all batches), without updating at the end of each batch. For example, the default option in Keras is to return the training loss as averaged over the individual batch losses.
      * The validation can be easier (to learn) than the training sample. This can happen if the validation and training sample are not formed from the same dataset, or if, for any reason, the validation data isn't as hard to classify as the training data. This can also happen if some training events are also mixed in the validation sample. If the code creating the training, validation and testing samples splits them correctly, from the same dataset, these shouldn't happen.

### True/False positive/negative

In a binary classification, each event is either in one or the other class, classes that we call "positive" and "negative" (eg. S or B). And each event is predicted to be in one of these two classes by the NN. We can define a confusion matrix which gives the fraction of each true category as classified in a predicted class, see the figure below. Obviously, the more diagonal is this matrix, the better is the classification of the NN; we will cover a quantitative measure of the entire matrix in the subsequent section.

> <img src="../../images/tutorials/nn_in_cms/TFpn.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 6

A criterion for a good classifier can be to maximize the fraction of true positive events (ie. S events classified as S) while also minimizing the false positive (ie. B events classified as S). We can report the rate of each of these events for various cuts on the NN output, as illustrated in the figure below. With this criterion, the classification power of the NN is optimal when the curve the peaks as much as possible to the top-left corner of this plot (green curve). In the opposite case, when the classifier produces as much true positive as false positive, it means that it does not have any classification power (red curve). A classifier whose curve lies between these limits is fine, but again, the closer it gets to the top-left corner, the better.

> <img src="../../images/tutorials/nn_in_cms/Good-vs-Bad_ROC.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 7

A quantitative measure of this "peaking" by the area under the (roc) curve, is reported in the figure below: the greater is this area, the better is the classification.

> <img src="../../images/tutorials/nn_in_cms/TrnVal_roc.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 8

### Accuracy versus epoch

A quantity measuring in a single shot all the elements of the binary confusion matrix is the accuracy, defined as the ratio of events classified in their correct classes to all events:

\begin{align}
Accuracy = \frac{True_{positive} + True_{negative} }{ True_{positive} + True_{negative} + False_{positive} + False_{negative}} & (12),
\end{align}

In the example below, we can observe an NN improving the accuracy after each epoch, and somehow plateau after a certain number of epochs:

> <img src="../../images/tutorials/nn_in_cms/TrnVal_accuracy.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 9

### Multi-class NN: confusion matrix

In the case of a multi-class NN, the confusion simply has more than two classes, as illustrated below. Again, the more diagonal is the matrix, the more correct is the classification.

> <img src="../../images/tutorials/nn_in_cms/Confusion_Matrix.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 10

### Over-training

We briefly mentioned at the start of the previous section the justification for having a validation sample. Through a comparison of the performance of the NN in the training and validation samples, one is testing the reproducibility of the NN: whether the NN 's response is similar for events that it has been trained upon (training sample) or events that it hasn't been exposed to (validation sample); this response should not depend on events. Therefore, <b>the NN's response should be similar in the training and validation samples, for both S and B events</b>.

\rightarrow One simple way to check for over-training is to overlay the NN output distribution in the training and validation samples (for S events on one hand, and for B events on the other), and make sure that that they are compatible within statistical uncertainties. In the example below, a comparison is provided between the validation and test samples.

> <img src="../../images/tutorials/nn_in_cms/Perf_NNoutput.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 11

### Assess performance of classification in analysis

An area under the receiver operating characteristics curve (auroc) is essentially measuring true S and B events being predicted as S events. Maximizing the auroc is mainly maximizing the S/B ratio. Here, there is no uncertainty of any kind taken into account. This is obviously insufficient for gauging the classification power of an ML tool in a real analysis situation, where both statistical and systematic uncertainties have to be accounted for. Beyond the auroc's, and in order to capture a more complete statistical picture of an analysis, one can define a Figure Of Merit, which is a quantity :

\begin{align}
FOM = \frac{S}{\sigma_T} & (13),
\end{align}

where S and σT are the signal yield and the total uncertainty, respectively. The total uncertainty is the quadratic sum of the systematic uncertainty on the background σB and the total statistical uncertainty, itself the quadratic sum of the statistical uncertainties on the signal and background. If we assume Poisson uncertainty for the statistical uncertainty on the yields, we have:

\begin{align}
FOM = \frac{S}{\sqrt{S+B+\sigma_B^2}} & (14),
\end{align}

where B is the background yield. If the analyzer knows that the measurement is going to be dominated by statistical uncertainties, then the expression above simplifies to:

\begin{align}
FOM = \frac{S}{\sqrt{S+B}} & (14-a).
\end{align}

If the analyzer further thinks that the statistical uncertainty on the signal is negligible when compared to the one on the background, the expression further simplifies to the well known expression:

\begin{align}
FOM = \frac{S}{\sqrt{B}} & (14-b).
\end{align}

It has to be noted that even with the most simplifying assumptions, the FOM above is close to, but not the same than S/B, which is effectively what the auroc is about.

### Code snippets

<b>Loss & accuracy curves</b>

For obtaining the loss and accuracy (versus epoch) curves, we should first include them among the very list of arguments to be compiled. For the loss, we should specify which type of loss we want, and for the accuracy, we should include it among the metrics:

```python
compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
```

Then, and after having trained the model, we can obtain these curves for the training and validation samples with the following lines:

```python
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history["accuracy"]
val_acc = history.history['val_accuracy']
```

<b>ROC curves</b>

In order calculate the roc curve and the auroc, one should first import the corresponding libraries:

```python
from sklearn.metrics import roc_curve, auc
```

We can then obtain the roc cruve and the auroc, for both the training and validation samples, with the following lines:

```python
y_pred_Trn = model.predict(xTrn).ravel()
fpr_Trn, tpr_Trn, thresholds_Trn = roc_curve(yTrn, y_pred_Trn)
auc_Trn = auc(fpr_Trn, tpr_Trn)
y_pred_Val = model.predict(xVal).ravel()
fpr_Val, tpr_Val, thresholds_Val = roc_curve(yVal, y_pred_Val)
auc_Val = auc(fpr_Val, tpr_Val)
```
