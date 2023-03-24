# Autoencoders

## Introduction
Autoencoders are a powerful tool that has gained popularity in HEP and beyond recently. These types of algorithms are neural networks that learn to decompress data with minimal reconstruction error [Goodfellow, et. al.][1a].

The two main parts of an autoencoder algorithm are the encoder function $f(x)$ and the decoder function $g(x)$. The learning process of an autoencoder is a minimization of a loss function, $L(x,g(f(x)))$, that compares the original data to the output of the decoder, similar to that of a neural network. As such, these algorithms can be trained using the same techniques, like minibatch gradient descent with backpropagation.

### Constrained Autoencoders (Undercomplete and Regularized)

An autoencoder that is able to perfectly reconstruct the original data one-to-one, such that $g(f(x)) = x$, is not very useful for extracting salient information from the data. There are several methods imposed on simple autoencoders to encourage them to extract useful aspects of the data.

One way of avoiding perfect data reconstruction is by constraining the dimension of the encoding function $f(x)$ to be less than the data $x$. These types of autoencoders are called *undercomplete autoencoders*, which force the imperfect copying of the data such that the encoding and decoding networks can prioritize the most useful aspects of the data. 


However, if undercomplete encoders are given too much capacity, they will struggle to learn anything of importance from the data. Similarly, this problem occurs in autoencoders with encoder dimensionality greater than or equal to the data (the overcomplete case). In order to train any architecture of AE successfully, constraints based on the complexity of the target distribution must be imposed, apart from small dimensionality. These *regularized autoencoders* can have constraints on the sparsity, robustness to noise, and robustness to changes in data (the derivative).


Regularized autoencoders (can be overcomplete, nonlinear)
	- sparse
	- denoising
	- penalized


## History

## Variational Autoencoders

## Applications in HEP

## Tensorflow Example

References
--
- [Goodfellow, et. al., 2016, "Deep Learning"][1a]

[1a]: https://www.deeplearningbook.org/contents/generative_models.html