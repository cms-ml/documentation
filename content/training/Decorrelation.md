# Decorrelation
When preparing to train a machine learning algorithm, it is important to think about the correlations of the output and their impact on how the trained model is used.
Generally, the goal of any training is to maximize correlations with variables of interests.
For example, a classifier is trained specifically to be highly correlated with the classification categories. 
However, there is often another set of variables that high correlation with the ML algorithm's output is not desirable and could make the ML algorithm useless regardless of its overall performance.

There are numerous methods that achieve the goal of minimizing correlations of ML algorithms.
Choosing the correct decorrelation method depends on the situation, e.g., which ML algorithm is being used and the type of the undesirable variables. 
Below, we detail various methods for common scenarios focusing on BDT (boosted decision tree) and neural network algorithms. 


## Impartial Training Data
Generally, the best method for making a neural network or BDT's output independent of some known variable is to remove any bias in the training dataset, which is commonly done by adding or removing information.

### Adding Information
1. Training on a mix of signals with different masses can help prevent the BDT from learning the mass.

### Removing Information
1. If you have any input variables that are highly correlated with the mass, you may want to omit them. There may be a loss of raw discrimination power with this approach, but the underlying interpretation will be more sound.


## Reweighting
1. "Reweighting all the input samples to match a reference kinematic distribution. This can be the mass directly, or an ingredient in the invariant mass, such as the pt. I describe this as reweighting to a reference distribution rather than flattening, because a completely flat distribution can require very large weights that skew the training. Making all the samples’ distributions match is usually enough to prevent the BDT from sculpting the background mass. This technique was ultimately what we used in EXO-19-020 (see AN-19-061 for more details)." - Kevin Pedro
    1. This is what is done for the ImageTop tagger and ParticleNet group of taggers.


## Adversarial Approach
1. Modify the loss function to enforce uniformity in the variable of interest (i.e. the mass). Check out these links ([1](https://arxiv.org/abs/1410.4140), [2](https://github.com/arogozhnikov/hep_ml), [3](https://github.com/arogozhnikov/hep_ml/blob/master/notebooks/BoostingToUniformity.ipynb)). "I’ve personally found that the “flatness loss function” (eq. 2.9, corresponding to BinFlatnessLossFunction in the python package) works best here." - Kevin Pedro
    1. This is what is done for the DeepAK8-MD taggers.

1. Add a distance correlation function to the loss function that calculates the non-linear correlation between the NN output and some variables that you care about, e.g. jet mass, that you can force the network to minimize which decorrelates the two variables. ([Disco](https://arxiv.org/pdf/2001.05310.pdf))


## Notes to delete later
Based on a question we received on [CMSTalk](https://cms-talk.web.cern.ch/t/bdt-score-mass-decorrelation/13184) we should create a new document covering decorrelation methods, an example of which would be decorrelating w.r.t the mass distribution.

1. Follow the [prescription](https://arxiv.org/abs/1603.00027) published by the JetMET group, which talks about designing V taggers. Basically changes the cut on the discriminant as a smooth analytical function of the variable of interest, say M_{X} (hereon referred to as MX).
    1. You can use a variant of DDT that defines a numerical function for every value of (MX,pt) (or MX,MY) plane; this function is smoothed (which is important, since non-smoothing may result in an artificial bump in data). You can read a lot about this in the documentation supporting EXO-17-001. This is usually called a “DDT map”.

1. Methods used by the "mass parameterized BDT" used in the HH Multi-lepton search (HIG-21-002). This was talked about in an [ML forum presentation](https://indico.cern.ch/event/818774/contributions/3420466/attachments/1840727/3019731/ML_Forum_talk_May8_2019.pdf).

    1. **Randomization method**: 
    Assign a mass value to the background event, chosen randomly out of the array of available signal mass values for which we have signal MC. This method is fast, but you are effectively reducing the available background per signal mass by the number of mass points.

    1. **Oversampling method**: 
    Make N copies of the background event, where N stands for the total number of signal masses for which we have signal MC, and assign each copy one of the mass values. This method is slower than the *randomization method*, but the stats per mass bin are the same.

    1. **Simplified method**: 
    Profile the input variables vs the signal mass in all of the signal MC and then do a polynomial fit to the distribution. Then correct/reweight the BDT input variables so that the input variable correlation with signal mass is flat.

1. Konstantin Matchev et. al. covered this in ([1911.12299](https://link.springer.com/article/10.1007/JHEP03(2021)291)). The example used in the question is covered in section 4 of the paper, with figure 4 being an example of the achievable decorrelation. The authors also produced a Python module called [ThickBrick](https://prasanthcakewalk.gitlab.io/thickbrick/) which implements their prescription. A demo of ThickBrick was given at [PyHEP2020](https://www.youtube.com/watch?v=rM19CoMNkfA).

[*] According to CMS-DP-2020-002 and the ParticleNet authors, (3) worked better than (4) and both are preferable to the DDT methods in (5). You can also do (3) and then apply DDT (5) on top of that to take care of any residual differences.