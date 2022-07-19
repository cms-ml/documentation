# CMS Machine Learning Hackathons

Welcome to the CMS ML Hackathons! Here we encourage the exploration of cutting edge ML methods to particle physics problem through multi-day focused work. Form hackathon teams and work together with the ML Innovation group to get support with organization and announcements, hardware/software infrastructure, follow-up meetings and ML-related technical advise.

If you are interested in proposing a hackathon, please send an e-mail to the [CMS ML Innovation conveners](cms-conveners-ml-innovation@cern.ch) with a potential topic and we will get in touch!

Below follows a list of previous successful hackathons.

## HGCAL TICL reconstruction

20 Jun 2022 - 24 Jun 2022 <br>
[https://indico.cern.ch/e/ticlhack](https://indico.cern.ch/e/ticlhack)

Abstract: The HGCAL reconstruction relies on “The Iterative CLustering” (TICL) framework. It follows an iterative approach, first clusters energy deposits in the same layer (layer clusters) and then connect these layer clusters to reconstruct the particle shower by forming 3-D objects, the “tracksters”. There are multiple areas that could benefit from advanced ML techniques to further improve the reconstruction performance.

In this project we plan to tackle the following topics using ML:

- trackster identification (ie, identification of the type of particle initiating the shower) and energy regression
linking of tracksters stemming from the same particle to reconstruct the full shower and/or use a high-purity trackster as a seed and collect 2D (ie. layer clusters) and/or 3D (ie, tracksters) energy deposits in the vicinity of the seed trackster to fully reconstruct the particle shower
- tuning of the existing pattern recognition algorithms
- reconstruction under HL-LHC pile-up scenarios (eg., PU=150-200)
- trackster characterization, ie. predict if a trackster is a sound object in itself or determine if it is more likely to be a composite one.

A CodiMD document has been created with an overview of the topics and to keep track of the activities during the hackathon:

https://codimd.web.cern.ch/s/hMd74Yi7J

## Jet tagging
8 Nov 2021 - 11 Nov 2021 <br>
[https://indico.cern.ch/e/jethack](https://indico.cern.ch/e/jethack)

Abstract: The identification of the initial particle (quark, gluon, W/Z boson, etc..) responsible for the formation of the jet, also known as jet tagging, provides a powerful handle in both standard model (SM) measurements and searches for physics beyond the SM (BSM). In this project we propose the development of jet tagging algorithms both for small-radius (i.e. AK4) and large-radius (i.e., AK8) jets using as inputs the PF candidates.

Two main projects are covered:

- Jet tagging for scouting
- Jet tagging for Level-1

### Jet tagging for scouting
Using as inputs the PF candidates and local pixel tracks reconstructed in the scouting streams, the main goals of this project are the following:
 

Develop a jet-tagging baseline for scouting and compare the performance with the offline reconstruction
Understand the importance of the different input variables and the impact of -various configurations (e.g., on pixel track reconstruction) in the performance
Compare different jet tagging approaches with mind performance as well as inference time.
Proof of concept: ggF H->bb, ggF HH->4b, VBF HH->4b

### Jet tagging for Level-1
Using as input the newly developed particle flow candidates of Seeded Cone jets in the Level1 Correlator trigger, the following tasks will be worked on:

Developing a quark, gluon, b, pileup jet classifier for Seeded Cone R=0.4 jets using a combination of tt,VBF(H) and Drell-Yan Level1 samples
Develop tools to demonstrate the gain of such a jet tagging algorithm on a signal sample (like q vs g on VBF jets)
Study tagging performance as a function of the number of jet constituents
Study tagging performance for a "real" input vector (zero-paddes, perhaps unsorted)
Optimise jet constituent list of SeededCone Jets (N constituents, zero-removal, sorting etc)
Develop q/g/W/Z/t/H classifier for Seeded Cone R=0.8 jets

## GNN-4-tracking

27 Sept 2021 - 1 Oct 2021

https://indico.cern.ch/e/gnn4tracks

Abstract: The aim of this hackathon is to integrate graph neural nets (GNNs) for particle tracking into CMSSW.

The hackathon will make use of a GNN model reported by the paper Charged particle tracking via edge-classifying interaction networks by Gage DeZoort, Savannah Thais, et.al. They used a GNN to predict connections between detector pixel hits, and achieved accurate track building. They did this with the TrackML dataset, which uses a generic detector designed to be similar to CMS or ATLAS. Work is ongoing to apply this GNN approach to CMS data.

Tasks:
The hackathon aims to create a workflow that allows graph building and GNN inference within the framework of CMSSW. This would enable accurate testing of future GNN models and comparison to existing CMSSW track building methods. The hackathon will be divided into the following subtasks:

Task 1: Create a package for extracting graph features and building graphs in CMSSW 

Code will be provided from an EDAnalyzer that, given a miniAOD sample, extracts relevant features as flat ntuples to be used in graph building. There will be two subtasks; 
Re-factorise the “feature-extraction” logic in an existing EDAnalyzer to a CMSSW class interface 
Implement graph building in CMSSW
Code for the graph building will be provided in both Python and C++. 

The intention of 1.1 is to create an interface that can be used in both the existing EDAnalyzer and directly with SONIC (see Task 2). Task 1.2 will enable graphs to be built within CMSSW which can then be sent to Sonic. 
 
Task 2. GNN inference on Sonic servers 

The GNN uses Pytorch, which is not yet supported in CMSSW. We will therefore use SONIC (Services for Optimized Network Inference on Coprocessors). Pre-trained GNN models will be provided, which can be placed on the SONIC server. This task will aim to set up a workflow that sends data to SONIC, performs inference, and sends the predictions back into CMSSW. 
 
Task 3: Track fitting after GNN track building 

The GNN does pattern recognition to reconstruct particle tracks using detector hits. In order to reconstruct track parameters, one needs to do the final track fitting with e.g. a Kalman filter. The track segments created by the GNN needs to be converted to a TrackCandidate that can be fed to the track fitter. There is already code in CMSSW for this that can be repurposed. 
 
Task 4. Performance evaluation for the new track collection 

The fitted tracks will be compared to that of existing track methods. The commonly used MultiTrackValidator will be used for accessing the tracking physics performance. In addition, the workflow for evaluating the computing performance will be defined. This task will be the step for having a standard workflow for the comparison of the performance of new approaches, which will be developed in the future.
Participants can put their name below a subtask at this Google Doc.

### Material:
Code is provided at [this GitHub organisation](https://github.com/CMS-GNN-Tracking-Hackathon-2021).

## Anomaly detection

In this four day Machine Learning Hackathon, we will develop new anomaly detection algorithms for New Physics detection, intended for deployment in the two main stages of the CMS data aquisition system: The Level-1 trigger and the High Level Trigger.

There are two main projects:

### Event-based anomaly detection algorithms for the Level-1 Trigger
### Jet-based anomaly detection algorithms for the High Level Trigger, specifically targeting Run 3 scouting

### Materials
A list of  projects can be found [in this document](https://docs.google.com/document/d/15bx5mRpoO8wY_DmhuEC3gWFzadNu_wHjfHbe78vYgGc/edit?usp=sharing). 
Instructions for fetching the data and example code for the two projects can be found at

Level-1 Anomaly Detection: github.com/anomalyHackathon
