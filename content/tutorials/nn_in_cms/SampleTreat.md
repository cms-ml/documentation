---
Sample treatment
---

A NN is a numerical machine, and numbers that it grinds need to be controlled, often balanced. In this section, we will cover the most common aspects of sample treatment for a NN. Apart from the first which deals with the splitting of the samples, all other subsections can be considered as data pre-processing.

### Splitting: training, validation, and testing samples

We need 3 samples: one to train the NN, another to validate the training, and another where we run the analysis and get results. The main purpose of the validation sample is to see how well the NN classifies the same type of events as either S or B, this, once exposed to events which were not included in the training; we will cover more this aspect below. Therefore, <b>the training and validation samples should be exactly similar</b>: the fraction of the total number of events that they each represent, the composition of the samples, etc. For example: (1) If the training is performed over 25% of the signal sample, then the validation sample has to include a different still 25% of the signal sample as well; the same for the background sample. (2) If the background sample is made of 75% of Wjets and 25% of ttbar events for training, one needs to keep the same percentages for the background validation sample.

### Event normalization

In the case where the values of the input variables vary by several orders of magnitude and/or are different for the various input variables, the adjusting of the NN parameters might be difficult, typically because the same weights will have to cover the possibly wildly different values. It is therefore better to render these values comparable, while they should naturally retain their discriminating power.

* One possibility is to decrease the order of magnitude of an input variable $x_i$ while taking into account its mean value $<x_i>$ and standard deviation $\sigma_i$: $x'_i = \frac{(x_i - <x_i>)}{\sigma_i}$.
* Another possibility is to normalize the input variables to the $[-1,+1]$ interval. This has the property of being simple, zero-centered (which can be interesting in some cases), and will be illustrated in among the snippets below.

### Shuffling, seeding

It is important to randomly mix, ie. shuffle, S and B events in a given (training and/or validation) sample. This allows to not expose the training first or last to an event of a specific kind, which would bias the training. Having a data sample with eg. S events which come first can easily be the case when putting together, ie. concatenating, S and B events in a given (training or validation) sample.

The seeding of a NN is random if left unspecified. The specification of the seeding can be useful in the case one needs an exact reproduction of results.

### Event weighting & balancing

Each event i in the training should be weighted by appropriate weight wi which reflects some physics aspect(s) and the sample specificity. Namely:

* B weights should be like : $w_i(B) = \frac{\sigma}{N_{tot}} \times \Pi_i SF(i) \times w(i)$, where $\sigma$ is the cross-section of the event under consideration, $SF(i)$ is the scale factor, and $N_{tot}$ is the total number of events. This is because the NN has to:
    * Be exposed to processes proportionally to their production rate $\sigma$ in SM. It has to be noted that the cross section $\sigma$ has to be taken into account when the B sample includes various background processes; in the case of a single background process per sample, this can be omitted and be taken as 1.
    * "Be free" of number of generated events per sample.
    * Be trained with simulated events resembling as much as possible to a Data event via the appropriate scale factor(s).
* S weights should be like: $w_i(S) = \frac{1}{N_{tot}} \times \Pi_i SF(i) \times w(i)$.

Now these S & B weights, given their definition, can be very different and lead to numerical problems in the training of the NN: for example, the validation loss can reach O($10^{-6,-7}$) very early on, and lead to weird behaviors and/or under-performance. We should therefore put events on equal footing by properly balancing each event for the training. <b>In the case of binary classification</b>, we should have something like:

* B weights should be: $w_i(B) \times (\frac{N_{evt}(B)}{\Sigma_i w_i(B)})$.
* S weights should be: $w_i(S) \times (\frac{N_{evt}(B)}{\Sigma_i w_i(S)})$.

Here, both B and S weights contain the same total number of eg. B events $N_{evt}(B)$ as to render the respective weights numerically comparable, while naturally preserving their event-by-event differences through $w_i(B,S)$.

### Code snippets

<b>Event normalization</b>

For normalizing the values of the input variables to the $[-1,+1]$ interval, we can do the following in a loop over the full, eg. training sample:

```python
top = np.max(full_train[:, var]) # checks all lines by value of (variable in) column var
bot = np.min(full_train[:, var])
full_train[:, var] = (2*full_train[:, var] - top - bot)/(top - bot)
```

<b>Event shuffling</b>

For shuffling events, we can apply the following numpy command on the full eg. training sample:

```python
np.random.shuffle(full_train) 
```

Without the former line, the NN would be first exposed to S for many events, this because we have concatenated the entire training sample with this rather common command:

```python
full_train = np.concatenate((train_sig, train_bkg))
```

To make sure, and with only the risk of redundancy, we can include the shuffle command in the very line which takes care of the training of the NN:

```python
history = model.fit(xTrn, yTrn, validation_data=(xVal,yVal,weightVal), sample_weight=weightTrn, shuffle=True, callbacks=[checkpoint], **trainParams)
```

<b>Event balancing</b>

For balancing the events, we normalize the weights of S and B training sample as mentioned earlier. In the case of a binary classification, we can simply have:

```python
train_bkg[:,-2] *=  train_bkg.shape[0] / np.sum(train_bkg[:,-2])
train_sig[:,-2] *=  train_bkg.shape[0] / np.sum(train_sig[:,-2])
```

where the weight of each data sample is the penultimate element of datasets train_bkg and train_sig, hence the $[:-2]$ numpy notation.
