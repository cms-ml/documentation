[THIS IS A DRAFT VERSION]

- Updates for 12 July
    - add a detailed description for ParticleNet code implementation in Sec. 2.
    - make the skeleton for Sec. 3.
    - new feature in `Weaver`: now interact with `tensorboard` for better monitoring.


# ParticleNet

ParticleNet [[arXiv:1902.08570](https://arxiv.org/abs/1902.08570)] is a graph neural network architecture that has many applications in CMS, including heavy flavour jet tagging, jet mass regression, analysis-based applications, etc. The network is fed by various low-level point-like objects as input, e.g., the particle-flow candidates, to predict a feature of a jet or an event.

(TODO: A figure of the ParticleNet model)

With the triumph of ParticleNet in various application schemes in the CMS community, we introduce several user-specific aspects of this model. We cover the following items in four sections:

1. **[An introduction to ParticleNet](#introduction-to-particlenet)**. We step through
    - a general description of ParticleNet;
    - the advantages brought from the architecture by concept;
    - other advanced NN models seeking outstanding performance in the CMS community;
    - a summary of finished/ongoing work in/outside CMS using the ParticleNet(-like) model.
  
2. **[An introduction to `Weaver` and model implementations](#introduction-to-weaver-and-model-implementations)**. We showcase the ML R&D framework `Weaver` and the realization of the ParticleNet model in a step-by-step manner:
    - introducing the key elements and syntax and showing example cards working with a benchmark top tagging dataset;
    - training the three example networks out-of-the-box – from a simple MLP, to a DeepAK8 model (1D CNN), then to the ParticleNet structure (DGCNN) – with `Weaver` on the benchmark task;
    - checking the loss/accuracy curve and making a ROC plot.

    ==This section will facilitate users to design a custom ParticleNet model in their own application schemes.==

3. **[Tuning the ParticleNet model](#tuning-the-particlenet-model)**, including
    - some details that users should be aware of when using/modifying the ParticleNet model, in order to achieve better performance.

4. **[Inference of ParticleNet](#inference-of-particlenet-in-cmssw)**, introducing
    - how the ParticleNet model is inferenced using ONNX Runtime in `cmssw`.

-----
Corresponding persons:

 - Huilin Qu, Loukas Gouskos (original developers of ParticleNet)
 - Congqiao Li (author of the page)

-----

## Introduction to ParticleNet

### 1. General discription

ParticleNet is intrinsically a graph neural net (GNN) model, harnessing the [Dynamic Graph CNN](https://arxiv.org/abs/1801.07829) (DGCNN) as its substructure. 

Intuitively, it treats all candidates inside an object as a permutational-invariant set of points (e.g. a set of PF candidates), each carrying a feature vector (*η*, *φ*, *p*<sub>T</sub>, charge, etc.). The DGCNN uses EdgeConv operation is defined to exploit their spatial correlations (two-dimensional on the *η*-*φ* plain) by finding the *k*-nearest neighbours of each point, and generate a new latent graph layer where points are scattered on a high-dimensional latent space. This is a graph-type analogue of the classical 2D convolution operation, which acts on a fixed 2D grid (e.g., a picture) using a 3×3 local patch to explore the relations of a single-pixel with its 8 nearest pixels, then generates a new 2D grid. 

(TODO: A figure illustrating DGCNN)

The EdgeConv operation then repeats on the new graph to generate the third graph. As the procedure goes on, the spatial dimension where our graph lives in as well as the dimension of the feature vector a node carries gradually increase. A fully connected layer eventually consumes all high-level features and output our target score.

For its applications to CMS, see more in [[CMS-DP-2020-002](https://cds.cern.ch/record/2707946/files/DP2020_002.pdf)], (TODO)

### 2 Advantage

By concept, the advantage of the network may come from exploiting the permutational-invariant symmetry of the points, which is intrinsic to our physics objects. By using the 1D CNN, the points (PF candidates) are more often ordered by *p*<sub>T</sub> to fix on the 1D grid. Only correlations with neighbouring points with similar *p*<sub>T</sub> are grabbed by the network with a convolution operation. The Long Short-Term Memory (LSTM) type recurrent neural network (RNN) provides the flexibility to feed in a variant-length sequence and has a "memory" mechanism to cooperate the information it learns from an early node to the latest node. The concern is that such ordering of the sequence is still artificial, and not an underlying property that an NN must learn to accomplish its task. As a comparison, in the task of the natural language processing where LSTM has a huge advantage, the order of words are important characteristics of a language itself (reflects the "grammar" in some circumstances) and is a feature the NN must learn to master the language. The GNN architecture, on the other hand, provides good handling of both variational length of input as well as maintaining the permutational symmetry.

### 3. Other advanced development

At the time of writing, some other type of GNN-based architecture also shows advantages and have applications in the CMS community. These networks also maintain the permutation symmetry of their low-level inputs. Here we summarize some useful link for your reference.

 - (CMS internal) An overview of GNN applications to CMS [[CMS ML forum](https://indico.cern.ch/event/952419/)]
 - Several new ML architectures (to be given in)  [[ML4Jets2021](https://indico.cern.ch/event/980214/timetable/?view=standard)]
   - (TODO: to list some specific talks related to our topic)

(TODO: NEED to supplement more work after digging deeper)


## Introduction to `Weaver` and model implementations

[`Weaver`](https://github.com/hqucms/weaver) is a machine learning R&D framework for high energy physics (HEP) applications. It trains the neural net with PyTorch and is capable of exporting the model to the ONNX format for fast inference. For the spirit of its design, please see more in README.

Now we'll walk through several examples of NN training and predicting on `Weaver`. We are targeting on a benchmark top tagging task [[arXiv:1707.08966](https://arxiv.org/abs/1707.08966)], please read the "top tagging" section in the [IML public datasets webpage](https://iml.web.cern.ch/public-datasets) (the [gDoc](https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit)) for details. 

Our goal is to do some warm-up with `Weaver`, more specifically, to explore the neural net architecture of a simple multi-layer perceptron (MLP) model, then a DeepAK8 tagger model (the CMS nomenclature) based on 1D CNN and ResNet, and eventually to the ParticleNet model which is based on DGCNN. We'll dig deeper into their implementations in `Weaver` and try to illustrate as many details as possible. After the training and predicting complete, we'll make a comparison between their performance, and hopefully breaking the record for this benchmark with your own hand (and your machine :-)).


### 1. Key elements and syntax

While implementing a new training scheme in `Weaver`, two key elements are crucial from a users perspective: the model and the data configuration file. The model configuration file includes a `get_model` function that returns a `torch.nn.Module` type model plus the model info, the data configuration is a YAML format file describing how to process the input data. Please see the `Weaver` README for details.

Before moving on, we need a preprocessing of the benchmark datasets. The sample is an H5 format file including branches like energy `E_i` and 3-momenta `PX_i`, `PY_i`, `PZ_i` for each jet constituent *i* (*i*=0, ..., 199) inside a jet. All branches are in the 1D flat structure. We will reconstruct the data, converting them to 2D vectors (e.g., in the `vector<float>` format): `E`, `PX`, `PY`, `PZ` which is more commonly seen in CMS. Each vector now has a varied length and corresponds to the number of constituents for each event.

To preprocess on the original dataset, enter the `Weaver` base directory and run
```python
python utils/convert_top_bmk_datasets.py -i <your-sample-dir>
```
This will convert the `.h5` file to `.awkd` file and create some new variables for each jet, including the relative *η* and *φ* value w.r.t. main axis of the jet of each jet constituent.

(TODO: script not in `Weaver` yet)

The converted dataset (including the training, validation, and testing dataset) is used for input. Then, we show three NN model configurations below with some explanation to the code.

=== "A simple MLP"

    As proof of the concept, we explore an MLP model with two hidden layers. All layers are 1D vectors. The model configuration card is shown in `network/benchmark/mlp.py`.

    ```python linenums="1"
    some code here
    ```

    (TODO: some explanation on the model)

    The data card is shown in `data/benchmark/top_bmk_mlp.yaml`. (TODO: some explanation on the data config)
    
    In the following two models we will use similar data cards, the change will only be the way we present the input group(s).

=== "DeepAK8 (1D CNN)"

    !!! note
        The DeepAK8 tagger is a widely used highly-boosted jet tagging in the CMS community. The design of the model can be found in the CMS paper [[arXiv:2004.08262](https://arxiv.org/abs/2004.08262)]. The original model is trained on MXNet and its configuration can be found [here](https://github.com/hqucms/NNTools/blob/master/training/symbols/sym_ak8_pfcand_sv_resnet_v1.py). 
        
        We now migrate the model architecture to `Weaver` and train it on Pytorch. Also, we narrow the multi-class output score to the binary output to adapt our binary classification task (top vs. QCD jet). The model architecture is also re-optimised based on the benchmark datasets - we will cover some optimisation tips in the next section.

    The model card is given in `network/benchmark/DeepAK8.py`

    ```python linenums="1"
    some code here
    ```

    (TODO: some explanation on the model)

    The data card is shown in `data/benchmark/top_bmk_DeepAK8.yaml`, given in the similar way as in the MLP example.
    
    (TODO: some explanation on the data config)
    

=== "ParticleNet (DGCNN)"

    !!! note
        The ParticleNet model applied to the CMS analysis is provided in `network/particle_net_pf_sv.py`, and the data card in `data/ak15_points_pf_sv.yaml`. Here again, we modify the model and re-optimise the architecture to adapt the benchmark classification task.
    
    The model card is given in `network/benchmark/particle_net.py`, shown as follows. Below defines the ParticleNet model. The main definition is in `ParticleNetTagger1Path`. 

    ???+ hint "ParticleNet model config"
        ```python linenums="1"
        import torch
        import torch.nn as nn
        from utils.nn.model.ParticleNet import ParticleNet, FeatureConv


        class ParticleNetTagger1Path(nn.Module):

            def __init__(self,
                        pf_features_dims,
                        num_classes,
                        conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                        fc_params=[(128, 0.1)],
                        use_fusion=True,
                        use_fts_bn=True,
                        use_counts=True,
                        pf_input_dropout=None,
                        for_inference=False,
                        **kwargs):
                super(ParticleNetTagger1Path, self).__init__(**kwargs)
                self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
                self.pf_conv = FeatureConv(pf_features_dims, 32)
                self.pn = ParticleNet(input_dims=32,
                                    num_classes=num_classes,
                                    conv_params=conv_params,
                                    fc_params=fc_params,
                                    use_fusion=use_fusion,
                                    use_fts_bn=use_fts_bn,
                                    use_counts=use_counts,
                                    for_inference=for_inference)

            def forward(self, pf_points, pf_features, pf_mask):
                if self.pf_input_dropout:
                    pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
                    pf_points *= pf_mask
                    pf_features *= pf_mask

                return self.pn(pf_points, self.pf_conv(pf_features * pf_mask) * pf_mask, pf_mask)


        def get_model(data_config, **kwargs):
            conv_params = [
                (16, (64, 64, 64)),
                (16, (128, 128, 128)),
                (16, (256, 256, 256)),
                ]
            fc_params = [(256, 0.1)]
            use_fusion = True

            pf_features_dims = len(data_config.input_dicts['pf_features'])
            num_classes = len(data_config.label_value)
            model = ParticleNetTagger1Path(pf_features_dims, num_classes,
                                    conv_params, fc_params,
                                    use_fusion=use_fusion,
                                    use_fts_bn=kwargs.get('use_fts_bn', False),
                                    use_counts=kwargs.get('use_counts', True),
                                    pf_input_dropout=kwargs.get('pf_input_dropout', None),
                                    for_inference=kwargs.get('for_inference', False)
                                    )
            model_info = {
                'input_names':list(data_config.input_names),
                'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
                'output_names':['softmax'],
                'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
                }

            print(model, model_info)
            print(data_config.input_shapes)
            return model, model_info


        def get_loss(data_config, **kwargs):
            return torch.nn.CrossEntropyLoss()
        ```
    
    `ParticleNetTagger1Path` is an integrated model that contains a `ParticleNet` prototype as the right plots shows. 
    
    (TODO: the architecture plot)
    
    The input includes a set of points with two classes of features: the "coordinates" and the "features". In our case, the points represent the particle candidates inside a jet. The "coordinates" only includes the relative *η* and *φ* value of each particle. The "features" include the relative *η*, *φ*, the log of pT, energy, and perhaps with other features like the charge or multiple bool/integers representing the particle ID class. The feature vector first passes an early 1D convolution layer to extend the feature dimension to 32.

    The building block of the ParticleNet model is the EdgeConv block, see the left plot above. In each EdgeConv block, an analogues convolution method is defined on a graph. The input "coordinates" provide a view of spatial relations of the points in the Euclidean space. It determines the *k*-nearest neighbouring points for each point that will guide the update of the feature vector of a point. For each point, the updated feature vectors are based on the current state of the point and its *k* neighbours. Then the passes several 1D CNN layers with shortcuts (adopting from the ResNet structure).

    The shape of an EdgeConv layer is controlled by the tuple
    ```
    (16, (64, 64, 64))
    ```
    where the first number is *k* which decides how many neighbours we exploit to define the EdgeConv operation. The following `(64, 64, 64)` is the feature vector dimension of the three successive 1D CNN layer. 

    As an output of the EdgeConv layer, each particle has a high-dimensional output feature vector. These vectors are then also viewed as new sets of "coordinates" of these particles, establishing particle relations in a high-dimensional spacial space. This ensures the stackability of the EdgeConv block, as one can further use the new "coordinates" (thus relations) of particles to define the next EdgeConv convolution. This is the core to the Dynamic Graph CNN, as the model can dynamically change the correlations of each point based on learnable features.

    After passing several EdgeConv layers, all EdgeConv output vectors are concatenated in a dense layer as inspired by the ResNet. A dropout layer of p=0.1 is used to prevent overfitting. The full network output two scores in our case after soft-max that represent the one-hot encoding of the top vs. QCD class.

    A full structure printed from PyTorch is shown below:

    ??? hint "ParticleNet full-scale structure"
        ```
        ParticleNetTagger1Path(
            |0.577 M, 100.000% Params, 0.441 GMac, 100.000% MACs|
            (pf_conv): FeatureConv(
                |0.0 M, 0.035% Params, 0.0 GMac, 0.005% MACs|
                (conv): Sequential(
                |0.0 M, 0.035% Params, 0.0 GMac, 0.005% MACs|
                (0): BatchNorm1d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs|)
                (1): Conv1d(4, 32, kernel_size=(1,), stride=(1,), bias=False, |0.0 M, 0.022% Params, 0.0 GMac, 0.003% MACs|)
                (2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.011% Params, 0.0 GMac, 0.001% MACs|)
                (3): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs|)
                )
            )
            (pn): ParticleNet(
                |0.577 M, 99.965% Params, 0.441 GMac, 99.995% MACs|
                (edge_convs): ModuleList(
                |0.305 M, 52.823% Params, 0.424 GMac, 96.047% MACs|
                (0): EdgeConvBlock(
                    |0.015 M, 2.575% Params, 0.021 GMac, 4.716% MACs|
                    (convs): ModuleList(
                    |0.012 M, 2.131% Params, 0.02 GMac, 4.456% MACs|
                    (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.004 M, 0.710% Params, 0.007 GMac, 1.485% MACs|)
                    (1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.004 M, 0.710% Params, 0.007 GMac, 1.485% MACs|)
                    (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.004 M, 0.710% Params, 0.007 GMac, 1.485% MACs|)
                    )
                    (bns): ModuleList(
                    |0.0 M, 0.067% Params, 0.001 GMac, 0.139% MACs|
                    (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.022% Params, 0.0 GMac, 0.046% MACs|)
                    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.022% Params, 0.0 GMac, 0.046% MACs|)
                    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.022% Params, 0.0 GMac, 0.046% MACs|)
                    )
                    (acts): ModuleList(
                    |0.0 M, 0.000% Params, 0.0 GMac, 0.070% MACs|
                    (0): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.023% MACs|)
                    (1): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.023% MACs|)
                    (2): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.023% MACs|)
                    )
                    (sc): Conv1d(32, 64, kernel_size=(1,), stride=(1,), bias=False, |0.002 M, 0.355% Params, 0.0 GMac, 0.046% MACs|)
                    (sc_bn): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.022% Params, 0.0 GMac, 0.003% MACs|)
                    (sc_act): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs|)
                )
                (1): EdgeConvBlock(
                    |0.058 M, 10.121% Params, 0.081 GMac, 18.437% MACs|
                    (convs): ModuleList(
                    |0.049 M, 8.523% Params, 0.079 GMac, 17.825% MACs|
                    (0): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.016 M, 2.841% Params, 0.026 GMac, 5.942% MACs|)
                    (1): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.016 M, 2.841% Params, 0.026 GMac, 5.942% MACs|)
                    (2): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.016 M, 2.841% Params, 0.026 GMac, 5.942% MACs|)
                    )
                    (bns): ModuleList(
                    |0.001 M, 0.133% Params, 0.001 GMac, 0.279% MACs|
                    (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.044% Params, 0.0 GMac, 0.093% MACs|)
                    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.044% Params, 0.0 GMac, 0.093% MACs|)
                    (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.044% Params, 0.0 GMac, 0.093% MACs|)
                    )
                    (acts): ModuleList(
                    |0.0 M, 0.000% Params, 0.001 GMac, 0.139% MACs|
                    (0): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.046% MACs|)
                    (1): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.046% MACs|)
                    (2): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.046% MACs|)
                    )
                    (sc): Conv1d(64, 128, kernel_size=(1,), stride=(1,), bias=False, |0.008 M, 1.420% Params, 0.001 GMac, 0.186% MACs|)
                    (sc_bn): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.0 M, 0.044% Params, 0.0 GMac, 0.006% MACs|)
                    (sc_act): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs|)
                )
                (2): EdgeConvBlock(
                    |0.231 M, 40.128% Params, 0.322 GMac, 72.894% MACs|
                    (convs): ModuleList(
                    |0.197 M, 34.091% Params, 0.315 GMac, 71.299% MACs|
                    (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.066 M, 11.364% Params, 0.105 GMac, 23.766% MACs|)
                    (1): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.066 M, 11.364% Params, 0.105 GMac, 23.766% MACs|)
                    (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False, |0.066 M, 11.364% Params, 0.105 GMac, 23.766% MACs|)
                    )
                    (bns): ModuleList(
                    |0.002 M, 0.266% Params, 0.002 GMac, 0.557% MACs|
                    (0): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.001 M, 0.089% Params, 0.001 GMac, 0.186% MACs|)
                    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.001 M, 0.089% Params, 0.001 GMac, 0.186% MACs|)
                    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.001 M, 0.089% Params, 0.001 GMac, 0.186% MACs|)
                    )
                    (acts): ModuleList(
                    |0.0 M, 0.000% Params, 0.001 GMac, 0.279% MACs|
                    (0): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.093% MACs|)
                    (1): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.093% MACs|)
                    (2): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.093% MACs|)
                    )
                    (sc): Conv1d(128, 256, kernel_size=(1,), stride=(1,), bias=False, |0.033 M, 5.682% Params, 0.003 GMac, 0.743% MACs|)
                    (sc_bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.001 M, 0.089% Params, 0.0 GMac, 0.012% MACs|)
                    (sc_act): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs|)
                )
                )
                (fusion_block): Sequential(
                |0.173 M, 29.963% Params, 0.017 GMac, 3.925% MACs|
                (0): Conv1d(448, 384, kernel_size=(1,), stride=(1,), bias=False, |0.172 M, 29.830% Params, 0.017 GMac, 3.899% MACs|)
                (1): BatchNorm1d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, |0.001 M, 0.133% Params, 0.0 GMac, 0.017% MACs|)
                (2): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs|)
                )
                (fc): Sequential(
                |0.099 M, 17.179% Params, 0.0 GMac, 0.023% MACs|
                (0): Sequential(
                    |0.099 M, 17.090% Params, 0.0 GMac, 0.022% MACs|
                    (0): Linear(in_features=384, out_features=256, bias=True, |0.099 M, 17.090% Params, 0.0 GMac, 0.022% MACs|)
                    (1): ReLU(|0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs|)
                    (2): Dropout(p=0.1, inplace=False, |0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs|)
                )
                (1): Linear(in_features=256, out_features=2, bias=True, |0.001 M, 0.089% Params, 0.0 GMac, 0.000% MACs|)
                )
            )
        )
        ```
    

    The data card is shown in `data/benchmark/top_bmk_DeepAK8.yaml`, given in a similar way as in the MLP example. Here we group the inputs into three classes: `pf_points`, `pf_features` and `pf_masks`. They correspond to the `forward(self, pf_points, pf_features, pf_mask)` prototype of our `nn.Module` model, and will send in these 2D vectors in the mini-batch size for each iteration of training/predicting.
       
    ???+ hint "ParticleNet data config"
        ```yaml linenums="1"
        selection:
        ### use `&`, `|`, `~` for logical operations on numpy arrays
        ### can use functions from `math`, `np` (numpy), and `awkward` in the expression

        new_variables:
        ### [format] name: formula
        ### can use functions from `math`, `np` (numpy), and `awkward` in the expression
        pf_mask: awkward.JaggedArray.ones_like(E)
        is_bkg: np.logical_not(is_signal_new)

        preprocess:
        ### method: [manual, auto] - whether to use manually specified parameters for variable standardization
        method: manual
        ### data_fraction: fraction of events to use when calculating the mean/scale for the standardization
        data_fraction: 

        inputs:
        pf_points:
            length: 100
            vars: 
                - [PX, 0, 0.05]
                - [PY, 0, 0.05]
                - [PZ, 0, 0.05]
        pf_features:
            length: 100
            vars: 
            ### [format 1]: var_name (no transformation)
            ### [format 2]: [var_name, 
            ###              subtract_by(optional, default=None, no transf. if preprocess.method=manual, auto transf. if preprocess.method=auto), 
            ###              multiply_by(optional, default=1), 
            ###              clip_min(optional, default=-5), 
            ###              clip_max(optional, default=5), 
            ###              pad_value(optional, default=0)]
                - [PX, 0, 0.05]
                - [PY, 0, 0.05]
                - [PZ, 0, 0.05]
                - [E_log, 2, 1]
        pf_mask:
            length: 100
            vars: 
                - pf_mask

        labels:
        ### type can be `simple`, `custom`
        ### [option 1] use `simple` for binary/multi-class classification, then `value` is a list of 0-1 labels
        type: simple
        value: [
            is_signal_new, is_bkg
            ]
        ### [option 2] otherwise use `custom` to define the label, then `value` is a map
        # type: custom
        # value: 
            # target_mass: np.where(fj_isQCD, fj_genjet_sdmass, fj_gen_mass) 

        observers:
        - origIdx
        - idx
        - E_tot
        - PX_tot
        - PY_tot
        - PZ_tot
        - P_tot
        - Eta_tot
        - Phi_tot

        # weights:
        ### [option 1] use precomputed weights stored in the input files
        # use_precomputed_weights: true
        # weight_branches: [weight, class_weight]
        ### [option 2] compute weights on-the-fly using reweighting histograms
        ```

We summary the three network architecture in the following plot.

(TODO: add plots)

The number of parameters, FLOPS and input features are summarised in the table.

(TODO: a table)

### 2. Start training!

Now we can train the three neural networks on the provided model and data configurations. Here we present two ways of training. If you have a local machine with CUDA GPUs (or you simply like to try them out on CPUs), please refer to the local training. We may also use the public GPU resource on HTCondor (please do not occupy too many resources) which is our alternative option to have fun with training.

=== "Train on local GPUs"

    The three networks can be trained with a universal script. Note that `${data_config}`, `${model_config}`, and `${prefix}` refers to the value in the above table for each example; and the fake path should be replaced with the correct one.

    ```python
    python train.py \
    --data-train '<path-to-samples>/prep/top_train_*.awkd' '<path-to-samples>/prep/top_val_*.awkd' \
    --data-val '<path-to-samples>/prep/top_val_*.awkd' \
    --feed-separate-train-val --fetch-by-file --fetch-step 1 \
    --num-workers 3 \
    --data-config data/benchmark/${data_config} \
    --network-config networks/benchmark/${model_config} \
    --model-prefix output/${prefix} \
    --gpus 0,1 --batch-size 1024 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
    --log output/${prefix}.train.log
    ```

    !!! note
        The above script will runs on two GPUs. For running on CPUs specify `--gpu ''`.

    Then, predict the score on the test datasets using the best model:

    ```python
    python train.py --predict \
    --data-test '<path-to-samples>/prep/top_test_*.awkd' \
    --num-workers 3 \
    --data-config data/benchmark/${data_config} \
    --network-config networks/benchmark/${model_config} \
    --model-prefix output/${prefix}_best_epoch_state.pt \
    --gpus 0,1 --batch-size 1024 \
    --predict-output output/${prefix}_predict.root
    ```

    (TODO: some features are to be pushed to the repo!)

=== "Use GPUs on lxplus HTCondor"

    On lxplus HTCondor, the GPU(s) can be booked via the arguments `request_gpus`. To get familiar with the GPU service, please refer to the documentation [here](https://batchdocs.web.cern.ch/tutorial/exercise10.html).

    While it is not possible to test the script locally, you can try out the `condor_ssh_to_job` command to connect to the remote condor machine that runs the jobs. This interesting feature will help you with debugging or monitoring the condor job.
    
    Here we provide the example executed script and the condor submitted file for the training and predicting task. Create the following two files:

    ???+ hint "The executable: `run.sh`"
        Still, please remember to specify `${data_config}`, `${model_config}`, and `${prefix}` as shown in the above table, and replace the fake path to the correct one.
        ```shell linenums="1"
        #!/bin/bash

        WORKDIR=`pwd`

        # Download miniconda
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_install.sh
        bash miniconda_install.sh -b -p ${WORKDIR}/miniconda
        export PATH=$WORKDIR/miniconda/bin:$PATH
        pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
        pip install uproot3 awkward0 lz4 xxhash
        pip install tables
        pip install onnxruntime-gpu
        pip install torch

        # CUDA environment setup
        export PATH=$PATH:/usr/local/cuda-10.2/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
        export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64

        # Clone weaver
        git clone https://github.com/hqucms/weaver.git
        cd weaver/
        mkdir output

        # Training
        python train.py \
        --data-train '<path-to-samples>/prep/top_train_*.awkd' '<path-to-samples>/prep/top_val_*.awkd' \
        --data-val '<path-to-samples>/prep/top_val_*.awkd' \
        --feed-separate-train-val --fetch-by-file --fetch-step 1 \
        --num-workers 3 \
        --data-config data/benchmark/${data_config} \
        --network-config networks/benchmark/${model_config} \
        --model-prefix output/${prefix} \
        --gpus 0,1,2,3 --batch-size 1024 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
        --log output/${prefix}.train.log

        # Predicting score
        python train.py --predict \
        --data-test '<path-to-samples>/prep/top_test_*.awkd' \
        --num-workers 3 \
        --data-config data/benchmark/${data_config} \
        --network-config networks/benchmark/${model_config} \
        --model-prefix output/${prefix}_best_epoch_state.pt \
        --gpus 0,1 --batch-size 1024 \
        --predict-output output/${prefix}_predict.root
        ```

    ???+ hint "HTCondor submitted file: `submit.sub`"
        ```linenums="1"
        Universe                = vanilla
        executable              = run.sh
        arguments               = 
        output                  = logs/$(ClusterId).$(ProcId).out
        error                   = logs/$(ClusterId).$(ProcId).err
        log                     = logs/$(ClusterId).log
        should_transfer_files   = YES
        when_to_transfer_output = ON_EXIT_OR_EVICT
        transfer_output_files   = weaver/output
        transfer_output_remaps  = "output = output.$(ClusterId).$(ProcId)"
        request_GPUs = 1
        request_CPUs = 2
        +MaxRuntime = 604800
        queue
        ```
    Make the `run.sh` script an executable, then submit the job.
    ```shell
    chmod +x run.sh
    condor_submit submit.sh
    ```
    The `weaver/output` directory will be transferred back.

    On the HTCondor GPU node it is also possible to launch a Jupyter service, then establish interactive connection to the remote GPU nodes. To do so, ...

=== "Use GPUs on CMS Connect"

    CMS Connect provides several GPU nodes. ...

### 3. Making plots

In the `output` folder, we find the trained PyTorch models after every epoch as well as the log file that records the loss and accuracy in the runtime. We use the script `TODO` to make the loss and accuracy curve. Here is the result of my training. 

(TODO: a plot here)

The predict step also produces a predicted root file in the `output` folder, including the truth label, the predicted store, and several observer variables we provided in the data card. With the predicted root file, we make the ROC curve comparing the performance of the three trained models.

(TODO: a plot here)

To calculate AUC, accuracy, and 1/*e*<sub>B</sub> required by the benchmark table, run
```python
...
```

Here is the result from my training:

| Approach            | AUC  | Accuracy | 1/*e*<sub>B</sub> (@*e*<sub>S</sub>=0.3) |
| ------------------- | ---- | -------- | -------------- |
| 2-layer MLP         |      |          |                |
| DeepAK8 (1D CNN)    |      |          |                |
| ParticleNet (DGCNN) |      |          |                |

The ParticleNet model shows an outstanding performance in this classification task, compared with other approaches and also items  

## Tuning the ParticleNet model

When it comes to the real application of any DNN model, tunning the hyperparameters of our model is an important path leading to the best performance. In this section, we illustrate how to tune the ParticleNet model. For a more detailed discussion on tunning, please refer to [].

### Choice on the optimizer and the learning rate

The optimizer decides how our neural network update all its parameters, and the learning rate means how fast the parameters changes in one training iteration.

Here we quote from a suggested strategy: if you only have the opportunity to optimize one hyperparameter, choose the learning rate. The optimizer is also important because a wiser strategy usually means avoid the zig-zagging updating route, avoid falling into the local minima and even adapting different strategies for the fast-changing parameters and the slow ones. Adam (and its several variations) is a widely used optimizer. Another recently developed "advanced" optimizer is Ranger that combines RAdam and LookAhead. However, the few percent level improvement by using different optimizers is likely to be smeared by an unoptimized learning rate. So we should still pay more attention to the latter.

In the following test, we randomly search for the best learning rate in the three optimizer choices: Adam, AdamW, and Ranger. We use the standard ParticleNet model and train for 5 epochs, for a collection of 50 random choices.


The accuracy and the loss shows...

(TODO: put results here)

### Optimize the model

Since all the EdgeConv layers of ParticleNet are tunable, it is important to adopt a proper set of model hyperparameters. The decision is made mainly based on our data size for training and the number of "points" per each input.

As you may see in our later test, an oversize of model parameters (usually results from large feature dimension and layer numbers) may easily cause overfitting, striking a significant difference in the training and validation performance; while too few parameters will result in a stable training model, but unsatisfactory performance.

(TODO: put results here)

## Inference of ParticleNet in `cmssw`

ParticleNet is now integrated to `cmssw`. Its inference is based on ONNX Runtime during the MiniAOD step. For a detailed description of ONNX Runtime interface in `cmssw`, please refer to []. Below we illustrate briefly the execution flow in `cmssw` for the ParticleNet model inference.
