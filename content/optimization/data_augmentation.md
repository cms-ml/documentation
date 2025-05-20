# Data augmentation

## Introduction
*This introduction is based on papers by [Shorten & Khoshgoftaar, 2019][0x] and [Rebuffi et al., 2021][0c] among others*

With the increasing complexity and sizes of neural networks one needs huge amounts of data in order to train a state-of-the-art model. However, generating this data is often very resource and time intensive. Thus, one might either augment the existing data with more descriptive variables or combat the data scarcity problem by artificially increasing the size of the dataset by adding new instances without the resource-heavy generation process. Both processes are known in machine learning (ML) applications as *data augmentation (DA)* methods. 

The first type of these methods is more widely known as *feature generation* or *feature engineering* and is done on instance level. Feature engineering focuses on crafting informative input features for the algorithm, often inspired or derived from first principles specific to the algorithm's application domain.


The second type of method is done on the dataset level. These types of techniques can generally be divided into two main categories: *real data augmentation* (RDA) and *synthetic data augmentation* (SDA). As the name suggests, RDA makes minor changes to the already existing data in order to generate new samples, whereas SDA generates new data from scratch. Examples of RDA include rotating (especially useful if we expect the event to be rotationally symmetric) and zooming, among a plethora of other methods detailed in this overview article. Examples of SDA include traditional sampling methods and more complex generative models like Generative Adversaial Netoworks (GANs) and Variational Autoencoders (VAE). Going further, the generative methods used for synthetic data augmentation could also be used in fast simulation, which is a notable bottleneck in the overall physics analysis workflow.


Dataset augmentation may lead to more successful algorithm outcomes. For example, introducing noise into data to form additional data points improves the learning ability of several models which otherwise performed relatively poorly, as shown by [Freer & Yang, 2020][0a]. This finding implies that this form of DA creates variations that the model may see in the real world. If done [right](#tips-and-tricks), preprocessing the data with DA will result in superior training outcomes. This improvement in performance is due to the fact that DA methods act as a regularizer, reducing overfitting during training. In addition to simulating real-world variations, DA methods can also even out categorical data with [imbalanced classes](#class-imbalance).


| ![Data Augmentation](../images/optimization/data_augmentation.png "Data augmentation") |
|:--:|
| Fig. 1: *Generic pipeline of a heuristic DA (figure taken from [Li, 2020][0b])* |


Before diving more in depth into the various DA methods and applications in HEP, here is a list of the most notable **benefits of using DA** methods in your ML workflow:

- Improvement of model prediction precision
- More training data for the model
- Preventing data scarcity for state-of-the-art models
- Reduction of over overfitting and creation of data variability
- Increased model generalization properties
- Help in resolving class imbalance problems in datasets
- Reduced cost of data collection and labeling
- Enabling rare event prediction

And some words of caution:

- There is no 'one size fits all' in DA. Each dataset and usecase should be considered separately.
- Don't trust the augmented data blindly
- Make sure that the augmented data is representative of the problem at hand, otherwise it will negatively affect the model performance.
- There must be no unnecessary duplication of existing data, only by adding unique information we gain more insights.
- Ensure the validity of the augmented data before using it in ML models.
- If a real dataset contains biases, data augmented from it will contain biases, too. So, identification of optimal data augmentation strategy is important. So, double check your DA strategy.




<!-- 1. -->
## Feature Engineering
*This part is based mostly on [Erdmann et al., 2018][1x]*

Feature engineering (FE) is one of the key components of a machine learning workflow.
This process transforms and augments training data with additional features in order to make the training more effective.

With multi-variate analyeses (MVAs), such boosted decision trees (BDTs) and neural networks, one could start with raw, "low-level" features, like four-momenta, and the algorithm can learn higher level patterns, correlations, metrics, etc. However, using "high-level" variables, in many cases, leads to outcomes superior to the use of low-level variables. As such, features used in MVAs are handcrafted from physics first principles. 

Still, it is shown that a deep neural network (DNN) can perform better if it is trained with both specifically constructed variables and low-level variables. This observation suggests that the network extracts additional information from the training data.

### HEP Application - Lorentz Boosted Network
For the purposeses of FE in HEP, a novel ML architecture called a *Lorentz Boost Network (LBN)* (see Fig. 2) was proposed and implemented by [Erdmann et al., 2018][1x]. It is a multipurpose method that uses Lorentz transformations to exploit and uncover structures in particle collision events. LBN is the first stage of a two-stage neural network (NN) model, that enables a fully autonomous and comprehensive characterization of collision events by exploiting exclusively the four-momenta of the final-state particles. 

Within LBN, particles are combined to create rest frames representions, which enables the formation of further composite particles. These combinations are realized via linear combinations of N input four-vectors to a number of M particles and rest frames. Subsequently these composite particles are then transformed into said rest frames by Lorentz transformations in an efficient and fully vectorized implementation.

The properties of the composite, transformed particles are compiled in the form of characteristic variables like masses, angles, etc. that serve as input for a subsequent network - the second stage, which has to be configured for a specific analysis task, like classification.

The authors observed leading performance with the LBN and demonstrated that LBN forms physically meaningful particle combinations and generates suitable characteristic variables.


The usual ML workflow, employing LBN, is as follows:

```
Step-1: LBN(M, F)

    1.0: Input hyperparameters: number of combinations M; number of features F
    1.0: Choose: number of incoming particles, N, according to the research
         question

    1.1: Combination of input four-vectors to particles and rest frames

    1.2: Lorentz transformations

    1.3 Extraction of suitable high-level objects


Step-2: NN

    2.X: Train some form of a NN using an objective function that depends on
         the analysis / research question.

```


| ![LBN](../images/optimization/LBN_architecture.png "Lorentz Boost Network architecture") |
|:--:|
| Fig. 2: *The Lorentz Boost Network architecture (figure taken from [Erdmann et al., 2018][1x])* |


The [LBN package](https://github.com/riga/LBN) is also pip-installable:

```bash
pip install lbn
```




## RDA Techniques
*This section and the following subsection are based on the papers by [Freer & Yang, 2020][0a], [Dolan & Ore, 2021][1a], [Barnard et al., 2016][1e], and [Bradshaw et al., 2019][1b]*

RDA methods augment the existing dataset by performance some transformation on the existing data points. These transformations could include rotation, flipping, color shift (for an image), Fourier transforming (for signal processing) or some other transformation that preserves the validity of the data point and its corresponding label. As mentioned in [Freer & Yang, 2020][0a], these types of transformations augment the dataset to capture potential variations that the population of data may exhibit, allowing the network to capture a more generalized view of the sampled data.



### HEP Application - Zooming
<!--In order to tag jets over a wide range of transverse momenta of jet masses, one often needs multiple networks trained on a specific narrow range of transverse momenta. However this is undesirable, since  training data for the entire range over which the model will be used is necessary. Furthermore this introduces unnecessary complexity to the tagger. One way to overcome the problem is to train a mass-generalized jet tagger, where with a simple DA strategy we standardize the angular scale of jets with different masses - this kind of strategy is shown to produce a strong generalization by [Dolan & Ore, 2021][1a].-->

In [Barnard et al., 2016][1e], the authors investigate the effect of parton shower modelling in DNN jet taggers using images of hadronically decaying W bosons. They introduce a method known as zooming to study the scale invariance of these networks. This is the RDA strategy used by [Dolan & Ore, 2021][1a]. Zooming is similar to a normalization procedure such that it standardizes features in signal data, but it aims to not create similar features in background. 

After some standard data processing steps, including jet trimming and clustering via the $k_t$ algorithm, and some further processing to remove spatial symmetries, the resulting jet image depicts the leading subjet and subleading subjet directly below. [Barnard et al., 2016][1e] notes that the separation between the leading and subleading subjets varies linearly as $2m/p_T$ where $m$ and $p_T$ are the mass and transverse momentum of the jet. Standardizing this separation, or removing the linear dependence, would allow the DNN tagger to generalize to a wide range of jet $p_T$. To this end, the authors construct a factor, $R/\DeltaR_{act}$, where $R$ is some fixed value and $\DeltaR_{act}$ is the separation between the leading and subleading subjets. To discriminate between signal and background images with this factor, the authors enlarge the jet images by a scaling factor of $\text{max}(R/s,1)$ where $s = 2m_W/p_T$ and $R$ is the original jet clustering size. This process of jet image enlargement by a linear mass and $p_T$ dependent factor to account for the distane between the leading and subleading jet is known as zooming. This process can be thought of as an RDA technique to augment the data in a domain-specific way.

Advantage of using the zooming technique is that it makes the construction of scale invariant taggers easier. Scale invariant searches which are able to interpolate between the boosted and resolved parts of phase space have the advantage of being applicable over a broad range of masses and kinematics, allowing a single search or analysis to be effective where previously more than one may have been necessary.

As predicted the zoomed network outperforms the unzoomed one, particularly at low signal efficiency, where the background rejection rises by around 20%. Zooming has the greatest effect at high pT.

<!--
Given the success of zooming, an alternative approach would be to consider scale transformations as a symmetry of the data and embed this information into the network architecture itself. There already exist work on jet taggers that are equivalent to other symmetries of jets as well as implementations of scaling-equivariant CNNs. Of course, in applications where the relationship between domains is less easily understood, it may not be possioble to identify the appropriate DA procedure.

In conclusion [Dolan & Ore, 2021][1a] found that mass-generalization is not necessarily a byproduct of decorrelation.
When jets are zoomed, all the models they compared behave similarly with far less correlation than the unzoomed baseline, where zooming provides a strong generalization for all models and leads to relatively small dependence on jet mass.
-->

<!-- ### Transformation Adversarial Networks for Data Augmentations (TANDA) -->

### Traditional SDA Techniques
<!-- ## Class imbalance -->
*Text in part based on [He et al., 2010][2k]*

Generally speaking, imbalanced learning occurs whenever some type of data distribution dominates the instance space compared to other data distributions. Methods for handling imbalanced learning problems can be divided into the following five major categories:

- **[Sampling strategies](#sampling)**
- **Synthetic data generation ([SMOTE](#synthetic-minority-over-sampling-technique-smote) & [ADASYN](#adaptive-synthetic-sampling-approach) & DataBoost-IM)**  - aims to overcome the imbalance by artificially generating data samples.
- **Cost-sensitive learning** - uses cost-matrix for different types of errors or instance to facilitate learning from imbalanced data sets. This means that cost-sensitive learning does not modify the imbalanced data distribution directly, but targets this problem by using different cost-matrices that describe the cost for misclassifying any particular data sample.
- **Active learning** - conventionally used to solve problems related to unlabeled data, though recently it has been used in learning imbalanced data sets. Instead of searching the entire training space, this method effectively selects informative instances from a random set of training populations, therefore significantly reducing the computational cost when dealing with large imbalanced data sets.
- **Kernel-based methods** - by integrating the regularized orthogonal weighed least squares (ROWLS) estimator, a kernel classifier construction algorithm is based on orthogonal forward selection (OFS) to optimize the model generalization for learning from two-class imbalanced data sets.


<!--Classification tasks benefit when the class distribution of the response variable is well balanced. A popular method in addition to data augmentation to solve the problem of class imbalance is [sampling](#sampling). This technique is used to adjust the class distribution of the dataset (i.e. the ratio between the different classes represented).-->


### Sampling

When the percentage of the minority class is less than 5%, it can be considered a rare event. When a dataset is imbalanced or when a rare event occurs, it will be difficult to get a meaningful and good predictive model due to lack of information about the rare event [Au et al., 2010][2i]. In these cases, re-sampling techniques can be helpful. The re-sampling techniques are implemented in four different categories: undersampling the majority class, oversampling the minority class, combining over- and undersampling, and ensembling sampling. Oversampling and undersampling are found to work well in improving the classification for the imbalanced dataset. [Yap et al., 2013][2h]


<!--**Random sampling**
I'm not sure why you mention random sampling if you say there are better methods without first describing it/pros + cons.
[[ref]](https://www.aaai.org/Papers/KDD/1998/KDD98-011.pdf)

Today there are more promising techniques that try to improve the disadvantages of random approaches, such as synthetic data augmentation ([SMOTE](#synthetic-minority-over-sampling-technique-smote), [ADASYN](#adaptive-synthetic-sampling-approach-adasyn)) or clustering-based under-sampling techniques ([ENN](https://ieeexplore.ieee.org/document/4309137?TB_iframe=true&width=370.8&height=658.8)).-->


**Stratified sampling (STS)** 
This technique is used in cases where the data can be partitioned into strata (subpopulations), where each strata should be collectively exhaustive and mutually exclusive. The process of dividing the data into homogeneus subgroups before sampling is referred to as *stratification*. The two common strategies of STS are *proportionate allocation (PA)* and *optimum (disproportionate) allocation (OA)*. The former uses a fraction in each of the stata that is proportional to that of the total population. The latter uses the standard deviation of the distribution of the variable as well, so that the larger samples are taken from the strata that has the greatest variability to generate the least possible sampling variance. The advantages of using STS include smaller error in estimation (if measurements within strata have lower standard deviation) and similarity in uncertainties across all strata in case there is high variability in a given strata.

**NOTE:** STS is only useful if the population can be exhaustively partitioned into subgroups. Also in case of unknown class priors (the ratio of strata to the whole population) might have deleterious effects on the classification performance.

**Over- and undersampling**
Oversampling randomly duplicates minority class samples, while undersampling discards majority class samples in order to modify the class distribution. While oversampling might lead to overfitting, since it makes exact copies of the minority samples, undersampling may discard potentially useful majority samples.

Oversampling and undersampling are essentially opposite and roughly equivalent techniques. There are also more complex oversampling techniques, including the creation of artificial data points with algorithms like Synthetic Minority Over-sampling TEchnique (SMOTE). 

It has been shown that the combination of SMOTE and undersampling performs better than only undersampling the majority class. However, over- and undersampling remain popular as it each is much easier to implement alone than in some complex hybrid approach.

**Synthetic Minority Over-sampling Technique (SMOTE)**
*Text mostly based on [Chawla et al., 2002][2j] and in part on [He et al., 2010][2k]*

In case of Synthetic Minority Over-sampling Technique (SMOTE), the minority class is oversampled by creating synthetic examples along the line segments joining any or all of the $k$-nearest neighbours in the minority class.
The synthetic examples cause the classifier to create larger and less specific decision regions, rather than smaller and more specific regions.
More general regions are now learned for the minority class samples rather than those being subsumed by the majority class samples around them.
In this way SMOTE shifts the classifier learning bias toward the minority class and thus has the effect of allowing the model to generalize better.

There also exist extensions of this work like SMOTE-Boost in which the syntetic procedure was integrated with adaptive boosting techniques to change the method of updating weights to better compensate for skewed distributions.

So in general SMOTE proceeds as follows
```
SMOTE(N, X, k)
Input: N - Number of synthetic samples to be generated
       X - Underrepresented data
       k - Hyperparameter of number of nearest neighbours to be chosen

Create an empty list SYNTHETIC_SAMPLES
While N_SYNTHETIC_SAMPLES < N
    1. Randomly choose an entry xRand from X
    2. Find k nearest neighbours from X
    3. Randomly choose an entry xNeighbour from the k nearest neighbours
    4. Take difference dx between the xRand and xNeighbour
    5. Multiply dx by a random number between 0 and 1
    6. Append the result to SYNTHETIC_SAMPLES
Extend X by SYNTHETIC_SAMPLES
```


**Adaptive synthetic sampling approach (ADASYN)**
*Text mostly based on [He et al., 2010][2k]*

Adaptive synthetic sampling approach (ADASYN) is a sampling approach for learning from imbalanced datasets. The main idea is to use a weighted distribution for different minority class examples according to their level of difficulty in learning, where more synthetic data is generated for minority class examples that are harder to learn compared to those minority examples that are easier to learn. Thus, ADASYN improves learning with respect to the data distributions by reducing the bias introduced by the class imbalance and by adaptively shifting the classification boundary toward the difficult examples.

The objectives of ADASYN are reducing bias and learning adaptively. The key idea of this algorithm is to use a density distribution as a criterion to decide the number of synthetic samples that need to be generated for each minority data example. Physically, this density distribution is a distribution of weights for different minority class examples according to their level of difficulty in learning. The resulting dataset after using ADASYN will not only provide a balanced representation of the data distribution (according to the desired balance level defined in the configuration), but it also forces the learning algorithm to focus on those difficult to learn examples. It has been shown [He et al., 2010][2k], that this algorithm improves accuracy for both minority and majority classes and does not sacrifice one class in preference for another.

ADASYN is not limited to only two-class learning, but can also be generalized to multiple-class imbalanced learning problems as well as incremental learning applications.

For more details and comparisons of ADASYN to other algorithms, please see [He et al., 2010][2k].

### Existing implementations
[Imbalanced-learn](https://imbalanced-learn.org/stable/user_guide.html) is an open-source Python library which provides a suite of algorithms for treating the class imbalance problem.

For augmentig image data, one can use of of the following:

- Albumentations
- ImgAug
- Autoaugment
- Augmentor
- DeepAugmnent

But it is also possible to use tools directly implemented by tensorflow, keras etc. For example:

```python
flipped_image = tf.image.flip_left_right(image)
```


<!-- 2. -->
## Deep Learning-based SDA Techniques
In data science, data augmentation techniques are used to increase the amount of data by either synthetically creating data from already existing samples via a GAN or modifying the data at hand with small noise or rotation. ([Rebuffi et al., 2021][0c])

More recently, data augmentation studies have begun to focus on the field of deep learning (DL), more specifically on the ability of generative models, like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), to create artificial data. This synthetic data is then introduced during the classification model training process to improve performance and results.
<!--
I have no idea why this list is here it's also not entirely true - people use rotations, kernel filters, etc. for image processing - like in the zooming paper - in physics analyses
Although there exist many DA methods in classic image processing, like those listed below, usually these methods are not suitable for the tasks at hand in HEP.

- Geometric transformations (flip, crop, rotate, translate, etc.)
- Color space transformations (change RGB color channels, intensify color)
- Kernel filters (sharpen, blur)
- Random erasing (of a part of the image)
- Image mixing

Some common tools used in HEP applications are describen in the following sections.-->

**Generative Adversarial Networks (GANs)**
*The following text is written based on the works by [Musella & Pandolfi, 2018][2a] and [Hashemi et al., 2019][2b] and [Kansal et al., 2022][2c] and [Rehm et al., 2021][2d] and [Choi & Lim, 2021][2e] and [Kansal et al., 2020][2f]*

GANs have been proposed as a fast and accurate way of modeling high energy jet formation ([Paganini et al., 2017a][2o]) and modeling showers throughcalorimeters of high-energy physics experiments ([Paganini et al., 2017][2n] ; [Paganini et al., 2012][2m];  [Erdman et al., 2020][2l]; [Musella & Pandolfi, 2018][2a]) GANs have also been trained to accurately approximate bottlenecks in computationally expensive simulations of particle physics experiments. Applications in the context of present and proposed CERN experiments have demonstrated the potential of these methods for accelerating simulation and/or improving simulation fidelity[ (ATLAS Collaboration, 2018][2p];  [SHiP Collaboration, 2019][2q]).

The generative model approximates the combined response of aparticle detecor simulation and reconstruction algorithms to hadronic jets given the latent space of uniformly distributed noise, auxiliary features and jet image at particle level (jets clustered from the list of stable particles produced by PYTHIA).


In the paper by [Musella & Pandolfi, 2018][2a], the authors apply generative models parametrized by neural networks (GANs in particular) to the simulation of particles-detector response to hadronic jets. They show that this parametrization achieves high-fidelity while increasing the processing speed by several orders of magnitude.

Their model is trained to be capable of predicting the combined effect of particle-detector simulation models and reconstruction algorithms to hadronic jets.

Generative adversarial networks (GANs) are pairs of neural networks, a generative and a discriminative one, that are trained concurrently as players of a minimax game ([Musella & Pandolfi, 2018][2a]). The task of the generative network is to produce, starting from a latent space with a fixed distribution, samples that the discriminative model tries to distinguish from samples drawn from a target dataset. This kind of setup allows the distribution of the target dataset to be learned, provided that both of the networks have high enough capacity.

The input to these networks are hadronic jets, represented as "gray-scale" images of fixed size centered around the jet axis, with the pixel intensity corresponding to the energy fraction in a given cell. The architectures of the networks are based on the **image-to-image** translation. There few differences between this approach and image-to-image translation. Firstly, non-empty pixels are explicitly modelled in the generated images since these are much sparser than the natural ones. Secondly, feature matching and a dedicated adversarial classifier enforce good modelling of the total pixel intensity (energy). Lastly, the generator is conditioned on some auxiliary inputs.


By predicting directly the objects used at analysis level and thus reproducing the output of both detector simulation and reconstruction algorithms, computation time is reduced. This kind of philosophy is very similar to parametrized detector simulations, which are used in HEP for phenomenological studies. The attained accuracies are comparable to the full simulation and reconstruction chain.




<!-- [GANs for generating EFT models](https://arxiv.org/pdf/1809.02612.pdf) -->

### Variational autoencoders (VAEs)
*The following section is partly based on [Otten et al., 2021][2g]*


In contrast to the traditional autoencoder (AE) that outputs a single value for each encoding dimension, variational autoencoders (VAEs) provide a probabilistic interpretation for describing an observation in latent space.

In case of VAEs, the encoder model is sometimes referred to as the recognition model and the decoder model as generative model.

By constructing the encoder model to output a distribution of the values from which we randomly sample to feed into our decoder model, we are enforcing a continuous, smooth latent space representation. Thus we expect our decoder model to be able to accurately reconstruct the input for any sampling of the latent distributions, which then means that values residing close to each other in latent space should have very similar reconstructions.

<!-- need VAE data augmentation example -->




## ML-powered Data Generation for Fast Simulation
*The following text is based on this [Chen et al., 2020][1f]*


We rely on accurate simulation of physics processes, however currently it is very common for LHC physics to be affected by large systematic uncertanties due to the limited amount of simulated data, especially for precise measurements of SM processes for which large datasets are already available. So far the most widely used simulator is GEANT4 that provides state-of-the-art accuracy. But running this is demanding, both in terms of time and resources. Consequently, delivering synthetic data at the pace at which LHC delivers real data is one of the most challenging tasks for computing infrastructures of the LHC experiments. The typical time it takes to simulate one single event is in the ballpark of 100 seconds.

Recently, generative algorithms based on deep learning have been proposed as a possible solution to speed up GEANT4. However, one needs to work beyond the collision-as-image paradigm so that the DL-based simulation accounts for the irregular geometry of a typical detector while delivering a dataset in a format compatible with downstream reconstruction software.

One method to solve this bottleneck was proposed by [Chen et al., 2020][1f]. They adopt a generative DL model to convert an analysis specific representation of collision events at generator level to the corresponding representation at reconstruction level. Thus, this novel, fast-simulation workflow starts from a large amount of generator-level events to deliver large analysis-specific samples.

They trained a neural network to model detector resolution effects as a transfer function acting on an analysis-specific set of relevant features, computed at generator level. However, their model does not sample events from a latent space (like a [GAN](#generative-adversarial-networks-gans) or a plain [VAE](#variational-autoencoders-vaes)). Instead, it works as a fast simulator of a given generator-level event, preserving the correspondence between the reconstructed and the generated event, which allows us to compare event-by-event residual distributions. Furthermore, this model is much simpler than a generative model.

Step one in this workflow is generating events in their full format, which is the most resource heavy task, where, as noted before, generating one event takes roughly 100 seconds. However, with this new proposed method O(1000) events are generated per second. This would save on storage: for the full format O(1) MB/event is needed, where for the DL model only 8 MB was used to store 100000 events. To train the model, they used NVIDIA RTX2080 and it trained for 30 minutes, which in terms of overall production time is negligible. For generating N=1M events and n=10%N, one would save 90% of the CPU resources and 79% of the disk storage. Thus augmenting the centrally produced data is a viable method and could help the HEP community to face the computing challenges of the High-Luminosity LHC.

Another more extreme approach investigated the use of GANs and VAEs for generating physics quantities which are relevant to a specific analysis. In this case, one learns the N-dimensional density function of the event, in a space defined by the quantities of interest for a given analysis. So sampling from this function, one can generate new data. Trade-off between statistical precision (decreases with the increasing amount of generated events) and the systematic uncertainty that could be induced by a non accurate description of the n-dim pdf.

Qualitatively, no accuracy deterioration was observed due to scaling the dataset size for DL. This fact proves the robustness of the proposed methodology and its effectiveness for data augmentation.




## Open challenges in Data Augmentation
*Excerpts are taken from [Li, 2020][0b]*

The limitations of conventional data augmentation approaches reveal huge opportunities for research advances. Below we summarize a few challenges that motivate some of the works in the area of data augmentation.

- From manual to automated search algorithms: As opposed to performing suboptimal manual search, how can we design learnable algorithms to find augmentation strategies that can outperform human-designed heuristics?
- From practical to theoretical understanding: Despite the rapid progress of creating various augmentation approaches pragmatically, understanding their benefits remains a mystery because of a lack of analytic tools. How can we theoretically understand various data augmentations used in practice?
- From coarse-grained to fine-grained model quality assurance: While most existing data augmentation approaches focus on improving the overall performance of a model, it is often imperative to have a finer-grained perspective on critical subpopulations of data. When a model exhibits inconsistent predictions on important subgroups of data, how can we exploit data augmentations to mitigate the performance gap in a prescribed way?


<!-- In deep learning, architecture engineering is the new feature engineering -->

<!--
Advanced models for data augmentation are

Adversarial training/Adversarial machine learning: It generates adversarial examples which disrupt a machine learning model and injects them into a dataset to train.
Generative adversarial networks (GANs): GAN algorithms can learn patterns from input datasets and automatically create new examples which resemble training data.
Neural style transfer: Neural style transfer models can blend content image and style image and separate style from content.
Reinforcement learning: Reinforcement learning models train software agents to attain their goals and make decisions in a virtual environment. -->

<!--
## Appendix

### Planing
*This section is based on the papers by [Chang et al., 2018][1c] and [Oliveira et al., 2017][1d]*

Planing is one of many different approaches to understanding a networks discrimination power and is used for identifying combinations of variables that can discriminate signal from background is done by removing information, where the performance degradation of the new network provides diagnostic value.
Additionally it allows the investigation of the linear versus nonlinear nature of the boundaries between signal and background.

Planing was one of the first methods for mass decorrelation that was explored in ML studies of jet physics. The planing procedure introduces reweighting of the data to smooth away the features in a given variable as shown in (1), which in practice corresponds to binning the variable and inverting it. However, doing this produces still some finite bin effects. This weighing results in having uniform distributions in signal and background such that the jet mass no longer provides discrimination. New networks trained on the modified data. 


| ![Planing weights](../images/optimization/planing.png "Planing weights") |
|:--:|
| *Planing weights (Eq. 1)* |

By iteratively planing training data, it is possible to remove the machine's ability to classify. As a by-product, the planed variables determine combinations of input variables that explain the machine's discriminating power.

Another method of inferring the discrimination power of variables is *saturation*. It compares a network trained on only low level inputs with networks trained after adding higher-level variables. Saturation provides a tool to ensure that our networks are sufficiently deep, by checking that the new network's performance does not improve by much.

Yet another method would be to train networks using only the high-level variable(s) of interest as inputs, where in contrast to the saturation technique, no low level information is being provided to the network. The diagnostic test would be to compute if the resulting network can achieve performance similar to that of a deep network that has been trained on only the low level inputs.

However planing has two advantages over the previously described methods. First, the number of input parameters would typically change when going from only low level to only high level variables. Unlike planing this requires altering the network architecture. This in turn can impact the [optimization of hyperparameters](./model_optimization.md), thereby complicating the comparison. Furthermore this method suffers the same issue as saturation in that as the limit towards ideal performance is achieved, one is forced to take seriously small variations in the metrics. If there are not enough training trials to adequately determine the errors, these small variations could be incorrectly interpreted as consistent with zero. This can again be contrasted with planing in that our approach yields a qualitative drop in performance and is more straightforward to interpret.
-->



References
-----

- [Shorten & Khoshgoftaar, 2019, "A survey on Image Data Augmentationfor Deep Learning"][0x]
- [Freer & Yang, 2020, "Data augmentation for self-paced motor imagery classification with C-LSTM"][0a]
- [Li, 2020, "Automating Data Augmentation: Practice, Theory and New Direction"][0b]
- [Rebuffi et al., 2021, "Data Augmentation Can Improve Robustness"][0c]
<!--  -->
- [Erdmann et al., 2018, "Lorentz Boost Networks: Autonomous Physics-Inspired Feature Engineering"][1x]
- [Dolan & Ore, 2021, "Meta-learning and data augmentation for mass-generalised jet taggers"][1a]
- [Bradshaw et al., 2019, "Mass agnostic jet taggers"][1b]
- [Chang et al., 2018, "What is the Machine Learning?"][1c]
- [Oliveira et al. 2017, "Jet-Images – Deep Learning Edition"][1d]
- [Barnard et al., 2016, "Parton Shower Uncertainties in Jet Substructure Analyses with Deep Neural Networks"][1e]
- [Chen et al., 2020, "Data augmentation at the LHC through analysis-specific fast simulation with deep learning"][1f]
<!--  -->
- [Musella & Pandolfi, 2018, "Fast and accurate simulation of particle detectors using generative adversarial networks"][2a]
- [Hashemi et al., 2019, "LHC analysis-specific datasets with Generative Adversarial Networks"][2b]
- [Kansal et al., 2022, "Particle Cloud Generation with Message Passing Generative Adversarial Networks"][2c]
- [Rehm et al., 2021, "Reduced Precision Strategies for Deep Learning: A High Energy Physics Generative Adversarial Network Use Case"][2d]
- [Choi & Lim, 2021, "A Data-driven Event Generator for Hadron Colliders using Wasserstein Generative Adversarial Network"][2e]
- [Kansal et al., 2020, "Graph Generative Adversarial Networks for Sparse Data Generation in High Energy Physics"][2f]
- [Otten et al., 2021, "Event Generation and Statistical Sampling for Physics with Deep Generative Models and a Density Information Buffer"][2g]
- [Yap et al., 2013, "An Application of Oversampling, Undersampling, Bagging and Boosting in Handling Imbalanced Datasets"][2h]
- [Au et al., 2010, "Mining Rare Events Data by Sampling and Boosting: A Case Study"][2i]
- [Chawla et al., 2002, "SMOTE: Synthetic Minority Over-sampling Technique"][2j]
- [He et al., 2010, "ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"][2k]
- [Erdman et al., 2020, "Precise simulation of electromagnetic calorimeter showers using a Wasserstein Generative Adversarial Network"][2l]
- [Paganini et al., 2012, "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks"][2m]
- [Paganini et al., 2017, "Accelerating Science with Generative Adversarial Networks: An Application to 3D Particle Showers in Multi-Layer Calorimeters"][2n]
- [Paganini et al., 2017, "Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis"][2o]
- [ATLAS Collaboration, 2018, "Deep generative models for fast shower simulation in ATLAS"][2p]
- [SHiP Collaboration, 2019, "Fast simulation of muons produced at the SHiP experiment using Generative Adversarial Networks"][2q]

[0x]: https://journalofbigdata.springeropen.com/track/pdf/10.1186/s40537-019-0197-0.pdf
[0a]: https://iopscience.iop.org/article/10.1088/1741-2552/ab57c0
[0b]: https://ai.stanford.edu/blog/data-augmentation/
[0c]: https://arxiv.org/pdf/2111.05328.pdf
<!--  -->
[1a]: https://arxiv.org/pdf/2111.06047.pdf
[1b]: https://arxiv.org/pdf/1908.08959.pdf
[1c]: https://arxiv.org/pdf/1709.10106.pdf
[1d]: https://arxiv.org/pdf/1511.05190.pdf
[1e]: https://arxiv.org/pdf/1609.00607.pdf
[1f]: https://arxiv.org/pdf/2010.01835.pdf
[1x]: https://arxiv.org/abs/1812.09722
<!--  -->
[2a]: https://arxiv.org/pdf/1805.00850.pdf
[2b]: https://arxiv.org/abs/1901.05282
[2c]: https://arxiv.org/pdf/2106.11535.pdf
[2d]: https://arxiv.org/pdf/2103.10142.pdf
[2e]: https://arxiv.org/pdf/2102.11524.pdf
[2f]: https://arxiv.org/pdf/2012.00173.pdf
[2g]: https://arxiv.org/pdf/1901.00875.pdf
[2h]: https://link.springer.com/chapter/10.1007/978-981-4585-18-7_2
[2i]: https://link.springer.com/chapter/10.1007/978-3-642-12035-0_38
[2j]: https://www.jair.org/index.php/jair/article/view/10302/24590
[2k]: http://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf
[2l]: https://arxiv.org/pdf/1807.01954.pdf
[2m]: https://arxiv.org/pdf/1712.10321.pdf
[2n]: https://arxiv.org/pdf/1705.02355.pdf
[2o]: https://arxiv.org/pdf/1701.05927.pdf
[2p]: https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PUBNOTES/ATL-SOFT-PUB-2018-001/
[2q]: https://arxiv.org/abs/1909.04451.pdf
<!-- [CLAMP: Class-conditional Learned Augmentations for Model Patching] -->



---


Content may be edited and published elsewhere by the author.

Page author: Laurits Tani, 2022
