# CMS-ML Dataset Tab
## Introduction
Welcome to CMS-ML Dataset tab! Our tab is designed to provide accurate,
 up-to-date, and relevant data across various purposes. We strive to make this tab
resourceful for your analysis and decision-making needs. We are working on benchmarking
more dataset and presenting them in a user-friendly format.
This tab will be continuously updated
     to reflect the latest developments. Explore, analyze, and derive insights
       with ease!


## 1. JetNet

#### Links
[Github Repository](https://github.com/jet-net/JetNet)

[Zenodo](https://zenodo.org/records/6975118)

#### Description
 JetNet is a project aimed at enhancing accessibility and
  reproducibility in jet-based machine learning. It offers easy-to-access
   and standardized interfaces for several datasets, including JetNet,
   TopTagging, and QuarkGluon. Additionally, JetNet provides standard
    implementations of various generative evaluation metrics such as Fréchet
     Physics Distance (FPD), Kernel Physics Distance (KPD), Wasserstein-1 (W1),
      Fréchet ParticleNet Distance (FPND), coverage, and Minimum Matching Distance (MMD).
      Beyond these, it includes a differentiable implementation of the energy mover's distance and other general jet utilities, making it a comprehensive resource for researchers and practitioners in the field.

#### Nature of Objects
- Objects: Gluon (g), Top Quark (t), Light Quark (q), W boson (w), and Z boson (z) jets of ~1 TeV transverse momentum (\(p_T\))
- Number of Objects: N = 177252, 177945, 170679, 177172, 176952 for g, t, q, w, z jets respectively.


#### Format of Dataset
 - File Type: HDF5
 - Structure: Each file has particle_features; and jet_features; arrays, containing the list of particles' features per jet and the corresponding jet's features, respectively. Particle_features is of shape [N, 30, 4], where N is the total number of jets,
30 is the max number of particles per jet, and
4 is the number of particle features, in order: []\eta, \varphi, \p_T, mask]. See Zenodo for definitions of these.
jet_features is of shape [N, 4], where
4 is the number of jet features, in order: [\(p_T\), \(\eta\), mass, # of particles].

#### Related Projects
 - [Top tagging benchmark](https://arxiv.org/abs/1707.08966)
 - [Particle Cloud Generation with Message Passing Generative Adversarial Networks](https://arxiv.org/abs/2106.11535)


## 2. Top Tagging Benchmark Dataset

#### Links

[Zenodo](https://zenodo.org/records/2603256)

#### Description
A set of MC simulated training/testing events for the evaluation of top quark tagging architectures.
- 14 TeV, hadronic tops for signal, qcd diets background, Delphes ATLAS detector card with Pythia8
- No MPI/pile-up included
- Clustering of  particle-flow entries (produced by Delphes E-flow) into anti-kT 0.8 jets in the pT range [550,650] GeV
- All top jets are matched to a parton-level top within ∆R = 0.8, and to all top decay partons within 0.8
- Jets are required to have |eta| < 2
- The leading 200 jet constituent four-momenta are stored, with zero-padding for jets with fewer than 200
- Constituents are sorted by pT, with the highest pT one first
- The truth top four-momentum is stored as truth_px etc.
- A flag (1 for top, 0 for QCD) is kept for each jet. It is called is_signal_new
- The variable "ttv" (= test/train/validation) is kept for each jet. It indicates to which dataset the jet belongs. It is redundant as the different sets are already distributed as different files.

#### Nature of Objects
- Objects: 14 TeV, hadronic tops for signal, qcd diets background, Delphes ATLAS detector card with Pythia8
- Number of Objects: In total 1.2M training events, 400k validation events and 400k test events.

#### Format of Dataset
- File Type: HDF5
- Structure: Use “train” for training, “val” for validation during the training and “test” for final testing and reporting results. For details, see the Zenodo link

#### Related Projects
- Butter, Anja; Kasieczka, Gregor; Plehn, Tilman and Russell, Michael (2017). Based on data from 10.21468/SciPostPhys.5.3.028 (1707.08966)
- Kasieczka, Gregor et al (2019). Dataset used for arXiv:1902.09914 (The Machine Learning Landscape of Top Taggers)

## More dataset coming in!
Have any questions? Want your dataset shown on this page? Contact [CMS ML dataset benchmarking team](cms-conveners-ml-knowledge@cern.ch)!



<!--
Template

## JetNet

#### Links
[Github Repository]()

[Zenodo]()

#### Description

#### Nature of Objects
- Objects:
- Number of Objects

#### Format of Dataset
- File Type:
- Structure:

#### Related Projects
- []()

-->
