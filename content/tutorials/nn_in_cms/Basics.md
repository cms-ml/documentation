
---
The basics of a NN
---
In this section, we will cover the basic concepts of a Neural Network (NN) classifier. First, we will cover the quantity which is minimized and is guiding the training of a NN, quantity which can also be used by other classifiers. Then we will cover the specifics of a NN classifier, namely its architecture and how one builds its output discriminator as function of the input variables. We will then cover how a NN is trained, and go over the activation functions of a NN. Finally, we will provide code snippets illustrating the mentioned functionalities.

### Loss functions: binary & multi-class

When <b>classifying</b> events, we need to optimize a quantity which quantifies our classification; this quantity can be a loss function. If we are dealing with a binary classification of Signal (S) versus Background (B), we need a measure of how much we have classified signal events as S, and background events as B. The binary cross-entropy, which is one such quantity, is one of the loss functions used for binary studies, and is averaged over N events:

\begin{align}
L =  - \frac{1}{N} \times \Sigma_{i=1}^{N} [ z(i) \times ln(y(i)) + (1 - z(i)) \times ln(1 - y(i)) ] & (1),
\end{align}

where:

* $z(i)$ is the true classification: it is 1 for S, and 0 for B; this can be viewed as the <b>tag</b> of the event, ie. the prior knowledge that we provide to the classifier for this latter to know which event is S or B.
* $y(i)$ is the classifier's output, with its value between 0 and 1, ie. its <b>prediction</b> for whether an event is S(1) or B(0).

$L$ is reflective of the morphology of the classifier: the measure of the classification, itself a function of the separation achieved by the classifier. In equation (1), the first and second terms are "signal" and "background" term respectively. Indeed:

* $L = - \frac{1}{N} \times \Sigma_{i=1}^{N} ln(y(i))$. If the prediction is S: $y(i) \rightarrow 1$, then we have $L \rightarrow 0$.
* $L = - \frac{1}{N} \times \Sigma_{i=1}^{N} ln(1 - y(i))$. If the prediction is B: $y(i) \rightarrow 0$, then we have $L \rightarrow 0$.

The general, ie. multi-class, cross-entropy is given by:

\begin{align}
L =  - \frac{1}{N} \times \Sigma_{i=1}^{N} [ \Sigma_{j=1}^{m} z_j(i) \times ln(y_j(i)) ] & (1'),
\end{align}

where $j$ is the index of $m$ different classes. Eq. (1) is easily obtained by considering $m=2$, and considering that for each event $i$, we have $z_1 + z_2 = 1$ and $y_1 + y_2 = 1$.

It has to be noted that Keras, an open-source library in Python for artificial neural networks, minimizes a slightly different loss function. For example, for binary classification, the cross-entropy that Keras minimizes is given by:

\begin{align}
L_K = - \frac{1}{N} \times \Sigma_{i=1}^{N} w_i \times [ z(i) \times ln(y(i)) + (1 - z(i)) \times ln(1 - y(i)) ] & (2),
\end{align}

where $w_i$ is the event weight, reflecting the number of events in the sample, cross section, etc; it takes into account the (signal and background) samples, both in their shape (through bins of a distribution) and normalization. Therefore, to have a numerically balanced problem to solve, the weights should be made to be comparable:

\begin{align}
\Sigma_{i=1}^{N} w_i^S = \Sigma_{i=1}^{M} w_i^B & (3).
\end{align}

We will cover more this latest aspect in the subsection "Event balancing". Finally, it should be noted that for NNs performing tasks other than classification, loss functions different from cross-entropy are minimized. For example, for the case of a NN performing a regression, the minimized loss function is often the Mean Squared Error.

### Architecture & weights

In a NN, the information of the $n$ input variables $x_i$ is propagated to different nodes as illustrated in figure 1, where we represent a NN with one hidden layer of $m$ nodes. The information is propagated from the input nodes to the output node(s) via the hidden layer(s), representing the foward propagation of the NN. In this example, there is only output node. In the case of a multi-class NN, there is as much nodes as classes for classification.

> <img src="../../images/tutorials/nn_in_cms/NNarch-forw.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 1

Here, each input node $i$ sends the same input variable $x_i$ to all nodes of the hidden layer. Overall, the NN is fully connected, meaning that each node of a given layer has a connection with all nodes of the subsequent layer. The lines, ie. the numerical connections, between the nodes are the weights $w$, all different from one another, and are updated during the training of the NN (see section on learning). Higher/Lower weights indicate a stronger/weaker influence of one neuron on another. The hidden layers of the NN can be viewed as functions of the variables/nodes of the previous layer.

In the case of a NN with one hidden layer, the output discriminant $y$ of a NN at the output layer is given as a function of input variables $x_i$:

\begin{align}
y = \Sigma_j^{N_{nodes}} [ g(\Sigma_i^{N_{inputs}} w_{ij} \times x_i) \times w_j ] + O & (4),
\end{align}

where $w_{ij}$ is the weight between node $i$ of a layer and node $j$ of another layer, $w_j$ is the weight between node $j$ of penultimate layer and the output. $g$ is the activation function (see section on activation functions) operating at each node $h_j$ of the hidden layer. As such, y retains the information about the input variables plus a set of weights optimized to minimize the cross-entropy loss.

### Learning/Training of a NN

The NN is trained iteratively on the (training) data to adjust the weights, aiming to find their optimal values that minimize the loss, ie. minimize the difference between its predictions and the true values. This is done in the backward propagation step of the training, as illustrated in the figure below. The full iteration of a forward- & backward-propagation is called an epoch, ie. <b>epoch</b> in the training of the NN.

> <img src="../../images/tutorials/nn_in_cms/NNarch-forwbck.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 2

The prediction of the NN for an event $y_p$ is compared with the true value $y_t$, comparison upon which the loss value is calculated. This latter value is in turn fed to the optimizer, which updates the weights, injecting them back in the forward propagation part of the training. We cover the two most known optimizers in the two subsections below.

#### Gradient descent

The equation giving the evolution of the weights $w_{tij}$ at epoch $t$, is:

\begin{align}
w^{t+1}_{ij} = w^t_{ij} − [R \times e^{−D t} \times \Delta w^t_{ij} ] & (5),
\end{align}

where $R$ and $D$ are the learning and decay rates, respectively. $R$ is usually tested in the $[10^{-5},10^{-2}]$ interval. $\Delta w^t_{ij}$ is the partial derivative (versus $w$) of the back-propagation between two epochs:

\begin{align}
\Delta w^t_{ij} = \frac{\delta L_K}{\delta w} & (6).
\end{align}

This gradient gives the direction of the steepest increase of the loss function, and the learning rate $R$ controls the step size taken during an update in that direction. By moving in the opposite direction of the gradient, the algorithm iteratively adjusts the weights to reduce the error and improve the NN's performance. Let us explain this latter point more explicitly. The derivative of Eq. (6), within Eq. (5), points in the direction where the loss function $L_K$ decreases the most:

* If $\frac{\delta L_K}{\delta w} \geq  0$: $L_K$ is increasing, then $w_{ij}^{t+1} \leq w_{ij}^t$, thus weights will decrease and one can only have $\frac{\delta L_K}{\delta w} \sim 0$.
* If $\frac{\delta L_K}{\delta w} \leq 0$: $L_K$ is decreasing, meaning that there is less cross-entropy, thus $L_K \rightarrow 0$, thus: $\frac{\delta L_K}{\delta w} \sim 0$.

Let us now consider a numerical case. Let's consider a case where the weights $w_i$ of Eq. (2) are too small, eg. O($10^{−7}$), while the weights $w_{ij}$ are O($1$). In such a case, $L_K$ (see Eq. (2)), thus $\Delta w^t_{ij}$ (see Eq.(6)), will also be too small. In such a case, one can see from Eq. (5) that the NN learns almost nothing. This is one illustration of the fact that in a classification problem, the weights should be properly balanced; we will address this point in the section "Event balancing".

#### Adam

Equations (7) to (10) summarize the Adam algorithm [1]. $g_t$ gives the gradient of the function $f(\theta)$ to minimize, which can be a loss function, and which is a stochastic scalar function that is differentiable versus parameters $\theta$:

\begin{align}
g_t = \nabla \theta f_t (\theta_{t−1}) & (7).
\end{align}

$g_t$ can be viewed as the equivalent of the gradient of Eq. (6), where the function $f(\theta)$ can be identified with the loss function $L_K$, and parameters $\theta$ with the weights $w$ to be optimized. The algorithm updates exponential moving averages of the gradient ($m_t$) and of the squared gradient ($v_t$) as follow:

\begin{align}
m_t = e^{−\beta_1t} \times m_{t-1} + (1 - e^{−\beta_1t}) \times g_t & , & v_t = e^{−\beta_2t} \times v_{t-1} + (1 - e^{−\beta_2t}) \times g_t & (8),
\end{align}

where $\beta_{1,2} \in [0, 1[$ control the exponential decay rates of these moving averages. The moving averages themselves are estimates of the first moment (the mean) and the second raw moment (the uncentered variance) of the gradient. They are initialized as (vectors of) $\theta$’s. It should be noted that at $t=0$, we have: $m_t = m_{t−1}$, so the gradient descent doesn’t play a role for the first iteration. On the other hand, for $t = +Inf$. we have: $m_t= \nabla \theta f_t(\theta_{t−1})$, where only the gradient of the function plays a role. The estimate of these moments are given by:

\begin{align}
\hat{m}_t = \frac{m_t }{(1 − \beta_1^t)} & , & \hat{v}_t= \frac{v_t }{(1 − \beta_2^t)} & (9),
\end{align}

where $\beta^t$ is $\beta$ to the power $t$. Then, the updated parameters $\theta$ (equivalent of the weights of a NN) from an epoch to another can be written as function of the estimates of these two moments:

\begin{align}
\theta_t = \theta_{t−1} − \frac{\alpha \times \hat{m}_t }{( \hat{v}_t + \epsilon)} & (10).
\end{align}

It is interesting to note that an optimization based on Adam has an adaptive learning rate, which is deduced from the first (mean) and second (variance) moments of gradients: in Eq. (10), the effective step, proportional to a learning rate, is given by: $\alpha \frac{\hat{m}_t} {\sqrt{\hat{v}_t}}$, which is $t$-dependent and thus adaptive. For most of cases, it has $\alpha$ as upper bound. Generally, the Adam optimization is helpful when the objective function (eg. a loss or cost function) is stochastic: when it is composed of a sum of sub-functions evaluated at different sub-samples of data.

#### Regularization

If one $w$ i is too large, a given node will dominate others. Consequently, the NN, as ensemble of nodes, will stop to learn because a few nodes will dominate the whole process while not allowing the learning through a large enough number of nodes. Therefore, one can introduce weight regularization in the loss function to penalize too large weights $w$:

\begin{align}
L_1 = L + \alpha \times \Sigma_{i,j} |w_{ij}| & , & L_2 = L + \alpha \times \Sigma_{i,j} |w_{ij}|^2 & (11).
\end{align}

Practically, it is penalizing, possibly suppressing, the link between nodes $i$ and $j$. Regularization can stop the training when eg. the $L_2$ norm of the difference of weights between two epochs is smaller than $\epsilon$: $||w_t − w_{t−k}||^2 < \epsilon$; it reduces over-training.

### Activation functions

Forword: In this section $x$ or $z$ is generally the product of weights $w_{ij}$ and input values $x_i$.

Various activation functions are used for different nodes of a NN, their analytical properties matching the varying needs of different nodes and/or serving computational purposes. For some of activation functions, one can have $\frac{\delta L}{\delta w} \sim 0$ for extreme values of $x$. This has the inconvience of leading to a non-learning NN.

Let us consider the case where $g(x)=x$ which, through a simplified version of the NN, illustrates one of its basic capability. In this case, the outputs $y_j$ of the NN (see Eq. (4)) are completely linear, and provided there are enough nodes in the hidden layers versus the number of input variables, the NN would, via a linear combination of the input variables $x_i$, numerically diagonalise them. If the data is such that correlations among discriminating variables are only linear, this aspect of the NN would decorrelate them. If correlations among variables/features are of higher order, the user can decorrelate them before feeding them to the NN.

The purpose of the activation function is to introduce a non-linear functionality in the NN.

* For the first and hidden layers, the most common activation function used is the Rectified Linear Units (ReLU) function (see figure below). For positive values of $x$, ReLU function is simply $x$. This function avoids the $\frac{\delta L}{\delta w} \sim 0$ problem at positive, but not at negative values of $x$. It has another advantage, which is computational: since it doesn’t contain any exponential term, it results in a training time 6 times shorter than with either sigmoid or Tanh.
* The Tanh function (see figure below) has a zero-centered output which leads to an easier learning of weights which are a mixture of positive and negative weights; however, it has also the problem $\frac{\delta L}{\delta w} \sim 0$ for extreme values of $x$.
* The sigmoid function (see figure below) is useful for the output layer of the NN where we want a response in the $[0,1]$ interval. For hidden layers, this function has the $\frac{\delta L}{\delta w} \sim 0$ problem. Furthermore, it is not zero-centered. It is usually used for the output layer of binary NN’s. If the tag of events is as defined for Eq. (1), ie. $z=0/1$ for B/S events, the sigmoid function allows to classify/predict S events for $y>0.5$ and B events for $y<0.5$, this at a single node of the output layer, thus allowing to avoid the use of two nodes at the output layer for a binary NN.
* The Softmax activation function is typically used in the output layer of multi-class NN’s. It first amplifies the raw output scores of the nodes of the previous layer with an exponential function and then converts them into a probability distribution across multiple classes, ensuring that the probabilities for all classes sum up to 1. For example, for a NN with two nodes $(y_1,y_2)$ at its output layer, and where each $y$ is an output like in Eq. (4), the outcome of the first output node is: $\frac{e^{y_1}}{(e^{y_1}+e^{y_2})}$, while the one of the second output node is: $\frac{e^{y_2}}{(e^{y_1}+e^{y_2})}$.

> <img src="../../images/tutorials/nn_in_cms/ActivationFunctions.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 3

Finally, we should mention the initializer, which is a notion often associated with the activation functions. An initializer is a method for initializing the weights of a NN. Its goal is to avoid different nodes learning identical mappings (like Eq. (4)) within the network. This is achieved by taking the initial weights $w_{ij}$ as random numbers from a uniform interval $[-w,+w]$ or from a gaussian distribution with mean value 0 and standard deviation $\sigma$. One of the most famous initializers is the Glorot method, which draws samples from a uniform distribution with limits determined by the number of input and output units in the layer. The He Normal initializer is similar to the Glorot method while being specifically designed for ReLU activation function.

### Number of epochs, batch size

An epoch, as referred to in the subsection "Learning/Training of a NN" is simply an iteration where the full forward- & backward propagation take place, going through all training events once. The batch size $b$ is the number of events taken to update the weights. If $N$ is the number of events to be trained upon, then in 1 epoch, there are $N/b$ updates of the weights. Frequently, batch sizes are chosen as $2^m$. The update through several batches allows multiple parameter updates per epoch. Furthermore, in the case where the partial derivative of Eq. (6) is evaluated as expectation value over a limited number of events as in the batches, the partial derivative can have a larger variance, thus sometimes helping to escape from local minima.

> <img src="../../images/tutorials/nn_in_cms/Batch.png" alt="" style="width: 500px;"/>
> <figcaption>Figure 4

<b>Advice & Possible pitfalls:</b> It is generally a good practice to make b large as to include enough statistics for the training within an update. This can contribute to have a training/validation curve which is more stable, ie. with less fluctuations. On the other hand, if $b$ is so large as to match the size of the training sample $N$, there will be only one update in the training, as all data will be used to train the NN at once: this can be time consuming, and quite inefficient.

### Code snippets

We should first define the basic parameters of the NN: learning- & decay-rate, architecture, and important functions (activation and initialization). In the same shot, we can also set the number of epochs and the size of the batches. NB: please note that in [Keras 3](https://keras.io/api/optimizers/adam/) decay has changed to [learning rate schedules](https://keras.io/api/optimizers/learning_rate_schedules/).

```python
activ = "relu"
LearningRate = 1.e-5
DecayRate = 1.e-2
NodeLayer = "50 50"
architecture=NodeLayer.split()
ini = "he_normal"
n_epochs = 2000
batch_size = 10000
```

We can then pass these parameters to the optimizer, meanwhile defining other parameters of the NN such as the number of epochs and batch size. Please note that we will cover the two latter notions in the next section. Model's compilation arguments, training parameters and optimizer:

```python
trainParams = {'epochs': n_epochs, 'batch_size': batch_size, 'verbose': verbose}
compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ["accuracy"]}
myOpt = Adam(learning_rate=LearningRate, decay=DecayRate)
compileArgs['optimizer'] = myOpt
```

Finally, we build the model, also compiling the arguments provided above. In the example below, we first build the first layer where there are 12 input variables, then the hidden layers which have as many layers/nodes as specified in the argument $architecture$. For both initial and hidden layers, we use ReLU as activation function and he normal as initializer. We finally define the output layer with 1 single node with a sigmoid activation function.

```python
model = Sequential()
# 1st hidden layer: it has as many nodes as provided by architecture[0]
model.add(Dense(int(architecture[0]), input_dim=12, activation=activ, kernel_initializer=ini))
i=1
while i < len(architecture):
    model.add(Dense(int(architecture[i]), activation=activ, kernel_initializer=ini))
    i=i+1
model.add(Dense(1, activation='sigmoid')) # Output layer: 1 node, with sigmoid
model.compile(**compileArgs)
model.summary()
```

Once the model is defined, we train the model:

```python
history = model.fit(xTrn, yTrn, validation_data=(xVal,yVal,weightVal), sample_weight=weightTrn, shuffle=True, callbacks=[checkpoint], **trainParams)
```

When defining the model as we did above, we provided the criterion for saving the best epoch as the one where the weights are such that the validation loss is at its minimum. The method below (called $checkpoint$), which uses the $callbacks$ function, saves such weights, and the it is used in the line above for training:

```python
checkpoint = callbacks.ModelCheckpoint(
    filepath=filepath+"best_weights.h5",
    verbose=1,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True
)
```
