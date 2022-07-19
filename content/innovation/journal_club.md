# CMS Machine Learning Journal Club

Welcome to the CMS Machine Learning Journal Club (JC)! Here we read an discuss new cutting edge ML papers, with an emphasis on how these can be used within the collaboration. Below you can find a summary of each JC as well as some code examples demonstrating how to use the tools or methods introduced.

To vote for or to propose new papers for discussion, go to [https://cms-ml-journalclub.web.cern.ch/](https://cms-ml-journalclub.web.cern.ch/).

Below follows a complete list of all the previous CMS ML JHournal clubs, together with relevant documentation and code examples.

## Dealing with Nuisance Parameters using Machine Learning in High Energy Physics: a Review 
Tommaso Dorigo, Pablo de Castro

*Abstract: In this work we discuss the impact of nuisance parameters on the effectiveness of machine learning in high-energy physics problems, and provide a review of techniques that allow to include their effect and reduce their impact in the search for optimal selection criteria and variable transformations. The introduction of nuisance parameters complicates the supervised learning task and its correspondence with the data analysis goal, due to their contribution degrading the model performances in real data, and the necessary addition of uncertainties in the resulting statistical inference. The approaches discussed include nuisance-parameterized models, modified or adversary losses, semi-supervised learning approaches, and inference-aware techniques.*

- [Indico](https://indico.cern.ch/event/1001041/)
- [Paper](https://arxiv.org/abs/2007.09121)
       
## Mapping Machine-Learned Physics into a Human-Readable Space
Taylor Faucett, Jesse Thaler, Daniel Whiteson

*Abstract: We present a technique for translating a black-box machine-learned classifier operating on a high-dimensional input space into a small set of human-interpretable observables that can be combined to make the same classification decisions. We iteratively select these observables from a large space of high-level discriminants by finding those with the highest decision similarity relative to the black box, quantified via a metric we introduce that evaluates the relative ordering of pairs of inputs. Successive iterations focus only on the subset of input pairs that are misordered by the current set of observables. This method enables simplification of the machine-learning strategy, interpretation of the results in terms of well-understood physical concepts, validation of the physical model, and the potential for new insights into the nature of the problem itself. As a demonstration, we apply our approach to the benchmark task of jet classification in collider physics, where a convolutional neural network acting on calorimeter jet images outperforms a set of six well-known jet substructure observables. Our method maps the convolutional neural network into a set of observables called energy flow polynomials, and it closes the performance gap by identifying a class of observables with an interesting physical interpretation that has been previously overlooked in the jet substructure literature.*
- [Indico](https://indico.cern.ch/event/1020353/)
- [Paper](https://arxiv.org/abs/2010.11998)
  
## Model Interpretability (2 papers):
- [Indico](https://indico.cern.ch/event/1040324/)
### Identifying the relevant dependencies of the neural network response on characteristics of the input space
Stefan Wunsch, Raphael Friese, Roger Wolf, Günter Quast

*Abstract: The relation between the input and output spaces of neural networks (NNs) is investigated to identify those characteristics of the input space that have a large influence on the output for a given task. For this purpose, the NN function is decomposed into a Taylor expansion in each element of the input space. The Taylor coefficients contain information about the sensitivity of the NN response to the inputs. A metric is introduced that allows for the identification of the characteristics that mostly determine the performance of the NN in solving a given task. Finally, the capability of this metric to analyze the performance of the NN is evaluated based on a task common to data analyses in high-energy particle physics experiments.*

  - [Paper](https://arxiv.org/abs/1803.08782)
  - 
### iNNvestigate neural networks!
Maximilian Alber, Sebastian Lapuschkin, Philipp Seegerer, Miriam Hägele, Kristof T. Schütt, Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller, Sven Dähne, Pieter-Jan Kindermans

*In recent years, deep neural networks have revolutionized many application domains of machine learning and are key components of many critical decision or predictive processes. Therefore, it is crucial that domain specialists can understand and analyze actions and pre- dictions, even of the most complex neural network architectures. Despite these arguments neural networks are often treated as black boxes. In the attempt to alleviate this short- coming many analysis methods were proposed, yet the lack of reference implementations often makes a systematic comparison between the methods a major effort. The presented library iNNvestigate addresses this by providing a common interface and out-of-the- box implementation for many analysis methods, including the reference implementation for PatternNet and PatternAttribution as well as for LRP-methods. To demonstrate the versatility of iNNvestigate, we provide an analysis of image classifications for variety of state-of-the-art neural network architectures.*

- [Paper](https://arxiv.org/abs/1808.04260)
- [Code](https://github.com/albermax/innvestigate)
                 
## Simulation-based inference in particle physics and beyond (and beyond)
Johann Brehmer, Kyle Cranmer

*Abstract: Our predictions for particle physics processes are realized in a chain of complex simulators. They allow us to generate high-fidelity simulated data, but they are not well-suited for inference on the theory parameters with observed data. We explain why the likelihood function of high-dimensional LHC data cannot be explicitly evaluated, why this matters for data analysis, and reframe what the field has traditionally done to circumvent this problem. We then review new simulation-based inference methods that let us directly analyze high-dimensional data by combining machine learning techniques and information from the simulator. Initial studies indicate that these techniques have the potential to substantially improve the precision of LHC measurements. Finally, we discuss probabilistic programming, an emerging paradigm that lets us extend inference to the latent process of the simulator.*

- [Indico](https://indico.cern.ch/event/1084780/)
- [Paper](https://arxiv.org/abs/2010.06439)
- [Code](https://github.com/madminer-tool/madminer)
       
## Efficiency Parameterization with Neural Networks
C. Badiali, F.A. Di Bello, G. Frattari, E. Gross, V. Ippolito, M. Kado, J. Shlomi

*Abstract: Multidimensional efficiency maps are commonly used in high energy physics experiments to mitigate the limitations in the generation of large samples of simulated events. Binned multidimensional efficiency maps are however strongly limited by statistics. We propose a neural network approach to learn ratios of local densities to estimate in an optimal fashion efficiencies as a function of a set of parameters. Graph neural network techniques are used to account for the high dimensional correlations between different physics objects in the event. We show in a specific toy model how this method is applicable to produce accurate multidimensional efficiency maps for heavy flavor tagging classifiers in HEP experiments, including for processes on which it was not trained.*
- [Indico](https://indico.cern.ch/event/1143474/)
- [Paper](https://arxiv.org/abs/2004.02665)
- [Code](https://gitlab.cern.ch/nkakati/tt-with-gnn)
       
## A General Framework for Uncertainty Estimation in Deep Learning
Antonio Loquercio, Mattia Segù, Davide Scaramuzza

*Neural networks predictions are unreliable when the input sample is out of the training distribution or corrupted by noise. Being able to detect such failures automatically is fundamental to integrate deep learning algorithms into robotics. Current approaches for uncertainty estimation of neural networks require changes to the network and optimization process, typically ignore prior knowledge about the data, and tend to make over-simplifying assumptions which underestimate uncertainty. To address these limitations, we propose a novel framework for uncertainty estimation. Based on Bayesian belief networks and Monte-Carlo sampling, our framework not only fully models the different sources of prediction uncertainty, but also incorporates prior data information, e.g. sensor noise. We show theoretically that this gives us the ability to capture uncertainty better than existing methods. In addition, our framework has several desirable properties: (i) it is agnostic to the network architecture and task; (ii) it does not require changes in the optimization process; (iii) it can be applied to already trained architectures. We thoroughly validate the proposed framework through extensive experiments on both computer vision and control tasks, where we outperform previous methods by up to 23% in accuracy.*

- [Indico](https://indico.cern.ch/event/1163742/)
- [Paper](https://arxiv.org/abs/1907.06890)
- [Code](https://github.com/mattiasegu/uncertainty_estimation_deep_learning)
