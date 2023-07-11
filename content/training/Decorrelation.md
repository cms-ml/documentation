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

## Parametric Cut
When designing jet taggers, variables of interest for discriminators include N-subjettiness derived quantities. Often, these quantities will be correlated with, for example, the $p_T$ of the jet. One example of this type of correlation is called "mass scuplting" and happens when the distribution of the discriminating variable in background begins to exhibit a shape similar to that of the signal with successive cuts. This correlation can have confounding effects in the tagger and one way to remove these effects is to parametrically cut on the discriminant. 

One such prescription to remove these correlations is described [here](https://arxiv.org/abs/1603.00027) and focuses on removing the $p_T$ dependence in the soft-drop mass variable $\rho$. The authors note that there is a $p_T$ dependence in the N-subjettiness ratio $\tau_2/\tau_1$ as a function of the QCD jet scaling (soft-drop) variable, defined as $\rho = log(m^2)(p_T^2)$, which leads to mass sculpting. In order to alleviate this issue, the authors introduce a modified version of the soft-drop variable, $\rho' = \rho + log(p_T/\mu)$ where $\mu$ is chosen to be 1 GeV. It can also be noted that there is a linear depedence between $\tau_2/\tau_1$ and $\rho'$. Here, the authors remedy this by modelling the linear depedence with $\tau_{21}' + \tau_2/\tau_1 - M \times \rho'$ where $\M$ is fit from the data. Applying both these transformations flattens out the relationship between the ratio and the soft-drop variable and removes the mass sculpting effects. It is imperative that the transformation between variables are smooth, as discontinuous functions may lead to artificial features in the data.


## Methods for mass parameterized BDTs
A dihiggs multilepton search (HIG-21-002) made use of a BDT for signal discrimination, parameterized by the dihiggs invariant mass. In order to avoid correlations in the BDT output and invariant mass of the dihiggs system, they looked at three different methods of decorrelation: randomization, oversampling, and variable profiling. All of the methods utilized a (more or less) 50/50 dataset train/test split where one iteration of the BDT was trained on "even" numbered events and the datacards were produced with the "odd" numbered events. This procedure was repeated for the opposite configuration. Then, to deteremine if the BDT was correctly interpolating the signal masses, one mass point was omitted from training and the results of this BDT were compared to a BDT trained on only this single, omitted mass point. For each train/test configuration (even/odd or odd/even), the BDT's performance gain, as well as loss, were evaluated with ROC curves with two ommitted mass points (done separately).

In the randomization method, a generator-level dihiggs invariant mass was randomly assigned to each background event the BDT was trained on. For the oversampling method, every signal mass point was assigned to a duplicate of each background event. Obviously the oversampling method leads to slower execution but the same effective statistics for all backgrounds and each signal mass. Conversely, the randomization approach is quicker, but leads to reduced effective statistics. Lastly, to improve performance over lower signal masses, each BDT input variable was profiled as a function of $m_{HH}$. This profile was fit with a polynomial function, and then each point in the input distribution is divided by the fit function value. This corrected ratio is used as the new input to the BDT. The authors also found that splitting the BDT training into high and low mass regions helped.

In the end, oversampling, especially when combined with input variable corrections, provided a sizable performance gain (5.6%) over the randomization method. This gain is determined from ROC curves made for each training iteration (even/odd or odd/event) and each method. The performance loss is also a 5% improvement over the randomization method. 

For more information on these methods, see the HIG-21-002 [ML forum presentation](https://indico.cern.ch/event/818774/contributions/3420466/attachments/1840727/3019731/ML_Forum_talk_May8_2019.pdf)).




## Notes to delete later
Based on a question we received on [CMSTalk](https://cms-talk.web.cern.ch/t/bdt-score-mass-decorrelation/13184) we should create a new document covering decorrelation methods, an example of which would be decorrelating w.r.t the mass distribution.

1. Methods used by the "mass parameterized BDT" used in the HH Multi-lepton search (HIG-21-002). This was talked about in an [ML forum presentation](https://indico.cern.ch/event/818774/contributions/3420466/attachments/1840727/3019731/ML_Forum_talk_May8_2019.pdf).

    1. **Randomization method**: 
    Assign a mass value to the background event, chosen randomly out of the array of available signal mass values for which we have signal MC. This method is fast, but you are effectively reducing the available background per signal mass by the number of mass points.

    2. **Oversampling method**: 
    Make N copies of the background event, where N stands for the total number of signal masses for which we have signal MC, and assign each copy one of the mass values. This method is slower than the *randomization method*, but the stats per mass bin are the same.

    3. **Simplified method**: 
    Profile the input variables vs the signal mass in all of the signal MC and then do a polynomial fit to the distribution. Then correct/reweight the BDT input variables so that the input variable correlation with signal mass is flat.

1. Konstantin Matchev et. al. covered this in ([1911.12299](https://link.springer.com/article/10.1007/JHEP03(2021)291)). The example used in the question is covered in section 4 of the paper, with figure 4 being an example of the achievable decorrelation. The authors also produced a Python module called [ThickBrick](https://prasanthcakewalk.gitlab.io/thickbrick/) which implements their prescription. A demo of ThickBrick was given at [PyHEP2020](https://www.youtube.com/watch?v=rM19CoMNkfA).

[*] According to CMS-DP-2020-002 and the ParticleNet authors, (3) worked better than (4) and both are preferable to the DDT methods in (5). You can also do (3) and then apply DDT (5) on top of that to take care of any residual differences.