# PyTorch Geometric 

Geometric deep learning (GDL) is an emerging field focused on applying machine learning (ML) techniques to non-Euclidean domains such as graphs, point clouds, and manifolds. The PyTorch Geometric (PyG) library extends PyTorch to include GDL functionality, for example classes necessary to handle data with irregular structure. PyG is introduced at a high level in [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428) and in detail in the [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/). 

## GDL with PyG 
A complete reveiw of GDL is available in the following recently-published (and freely-available) textbook: [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478). The authors specify several key GDL architectures including convolutional neural networks (CNNs) operating on grids, Deep Sets architectures operating on sets, and graph neural networks (GNNs) operating on graphs, collections of nodes connected by edges. PyG is focused in particular on graph-structured data, which naturally encompases set-structured data. In fact, many state-of-the-art GNN architectures are implemented in PyG (see [the docs](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html))! A review of the landscape of GNN architectures is available in [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434). 


### The *Data* Class: PyG Graphs 
Graphs are data structures involving nodes $\nu_i$ connected by edges $e_{i,j}$. In a fully connected graph, every node is connected by an edge, so that $N_{edges}=N_{nodes}(N_{nodes}-1)/2$. However, graphs are not generally fully-connected; edge connectivity defines the local structure of the graph, specifying the relationship between different nodes. In general, nodes can have positions $\{pos_i\}_{i=1}^{N_{nodes}}$, $pos_i\in\mathcal{R}^{N_{dim}}$, and features (attributes) $\{x_i\}_{i=1}^{N_{nodes}}$, $x_i\in\mathcal{R}^{N_{node\_features}}$.

Edges, too, can have features $\{a_i\}_{i=1}^{N_{edges}}$, $a_i\in\mathcal{R}^{N_{edge\_features}}$,, but do not have positions; instead, edges are defined by the nodes they connect, and may therefore be represented by two node indices (i.e. $e_{i,j}=[i,j]$ connects $\nu_i$ and $\nu_j$). In PyG, graphs are stored as instances of the `data` class, whose fields fully specify the graph:

- `data.x`: node feature matrix, $X\in\mathcal{R}^{N_{nodes}\times N_{node\_features}}$
- `data.edge_index`: node indices at each end of each edge, $E\in\mathcal{R}^{2\times N_{edges}}$ 
- `data.edge_attr`: edge feature matrix, $A\in\mathcal{R}^{N_{edges}\times N_{edge\_features}}$ 
- `data.y`: training target with arbitary shape ($y\in\mathcal{R}^{N_{nodes}\times N_*}$ for node-level targets, $y\in\mathcal{R}^{N_{edges}\times N_*}$ for edge-level targets or $y\in\mathcal{R}^{1\times N_*}$ for node-level targets). 
- `data.pos`: Node position matrix, $P\in\mathcal{R}^{N_{nodes}\times N_{dim}}$
 
 
The PyG [Introduction By Example](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html) tutorial covers the basics of graph creation, batching, transformation, and inference using this `data` class. 

As an example, consider the [ZINC chemical compounds dataset](https://paperswithcode.com/dataset/zinc), which [available](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/zinc.html) as a built-in dataset in PyG:

```python
from torch_geometric.datasets import ZINC
train_dataset = ZINC(root='/tmp/ZINC', subset=True, split='train')
test_dataset =  ZINC(root='/tmp/ZINC', subset=True, split='test')
len(train_dataset)
>>> 10000
len(test_dataset)
>>> 1000   
```
Each graph in the dataset is a chemical compound; nodes are atoms and edges are chemical bonds. The node features `x` are categorical atom labels and the edge features `edge_attr` are categorical bond labels. The `edge_index` matrix lists all bonds present in the compound in COO format. The truth labels `y` indicate a synthetic computed property called constrained solubility; given a set of molecules represented as graphs, the task is to regress the constrained solubility. Therefore, this dataset is suitable for graph-level regression. Let's take a look at one molecule: 

```python
data = train_dataset[27]
data.x # node features
>>> tensor([[0], [0], [1], [2], [0], 
            [0], [2], [0], [1], [2],
            [4], [0], [0], [0], [0],
            [4], [0], [0], [0], [0]])

data.pos # node positions 
>>> None

data.edge_index # COO edge indices
>>> tensor([[ 0,  1,  1,  1,  2,  3,  3,  4,  4,  
              5,  5,  6,  6,  7,  7,  7,  8,  9, 
              9, 10, 10, 10, 11, 11, 12, 12, 13, 
              13, 14, 14, 15, 15, 15, 16, 16, 16,
              16, 17, 18, 19], # node indices w/ outgoing edges
            [ 1,  0,  2,  3,  1,  1,  4,  3,  5,  
              4,  6,  5,  7,  6,  8,  9,  7,  7,
              10,  9, 11, 15, 10, 12, 11, 13, 12, 
              14, 13, 15, 10, 14, 16, 15, 17, 18,
              19, 16, 16, 16]]) # node indices w/ incoming edges

data.edge_attr # edge features
>>> tensor([1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
            1, 1, 1, 1])

data.y # truth labels
>>> tensor([-0.0972])

data.num_nodes
>>> 20

data.num_edges
>>> 40

data.num_node_features
>>> 1 
```

We can load the full set of graphs onto an available GPU and create PyG dataloaders as follows:
```python
import torch
from torch_geometric.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
test_dataset = [d.to(device) for d in test_dataset]
train_dataset = [d.to(device) for d in train_dataset]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
```

### The *Message Passing* Base Class: PyG GNNs 
The 2017 paper [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) presents a unified framework for a swath of GNN architectures known as *message passing neural networks* (MPNNs). MPNNs are GNNs whose feature updates are given by:

$\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)},\mathbf{e}_{j,i}\right) \right)$

Here, $\gamma$ and $\phi$ are learnable functions (which we can approximate as multilayer perceptrons), $\square$ is a permutation-invariant function (e.g. mean, max, add), and $\mathcal{N}(i)$ is the neighborhood of node $i$. In PyG, you'd write your own MPNN by using the `MessagePassing` base class, implementing each of the above mathematical objects as an explicit function. 

- `MessagePassing.message()` : define an explicit NN for $\phi$, use it to calculate "messages" between a node $x_i^{(k-1)}$ and its neighbors $x_j^{(k-1)}$, $j\in\mathcal{N}(i)$, leveraging edge features $e_{j,i}$ if applicable
- `MessagePassing.propagate()` : in this step, messages are calculated via the `message` function and aggregated across each receiving node; the keyword `aggr` (which can be `'add'`, `'max'`, or `'mean'`) is used to specify the specific permutation invariant function $\square_{j\in\mathcal{N}(i)}$ used for message aggregation. 
- `MessagePassing.update()` : the results of message passing are used to update the node features $x_i^{(k)}$ through the $\gamma$ MLP 

The specific implementations of `message()`, `propagate()`, and `update()` are up to the user. A specific example is available in the PyG [Creating Message Passing Networks tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)

#### Message-Passing with ZINC Data
Returning to the ZINC molecular compound dataset, we can design a message-passing layer to aggregate messages across molecular graphs. Here, we'll define a multi-layer perceptron (MLP) class and use it to build a message passing layer (MPL) the following equation:
$$\mathbf{x}_i' = \gamma \left( \mathbf{x}_i, \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \, \phi\left([\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right]) \right)$$

Here, the MLP dimensions are constrained. Since $\textbf{x}_i,\textbf{e}_{i,j}\in\mathbb{R}$, the $\phi$ MLP must map $\mathbb{R}^3$ to $\mathbb{R}^{\text{message_size}}$. Similarly, $\gamma$ must map $\mathbb{R}^{1+\text{message_size}}$ to $\mathbb{R}^{\text{output_size}}$. 
```python
from torch_geometric.nn import MessagePassing
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        return self.layers(x)
    
class MPLayer(MessagePassing):
    def __init__(self, n_node_feats, n_edge_feats, message_size, output_size):
        super(MPLayer, self).__init__(aggr='mean', 
                                      flow='source_to_target')
        self.phi = MLP(2*n_node_feats + n_edge_feats, message_size)
        self.gamma = MLP(message_size + n_node_feats, output_size)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):       
        return self.phi(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.gamma(torch.cat([x, aggr_out], dim=1))
```

Let's apply this layer to one of the ZINC molecules:
```python
molecule = train_dataset[0]
torch.Size([29, 1]) # 29 atoms and 1 feature (atom label)
mpl = MPLayer(1, 1, 16, 8).to(device) # message_size = 16, output_size = 8
xprime = mpl(graph.x.float(), graph.edge_index, graph.edge_attr.unsqueeze(1))
xprime.shape
>>> torch.Size([29, 8]) # 29 atoms and 8 features
```
There we have it - the message passing layer has produced 8 new features for each atom. 

## GDL in Particle Physics
Here we list some examples of GDL in particle physics, citing git repositories where applicable. For a complete list of ML/HEP references, please see the [Living Review of ML for Particle Physics](https://iml-wg.github.io/HEPML-LivingReview/)
### Survey Papers / Articles 
- [Graph Neural Networks in Particle Physics](https://arxiv.org/abs/2007.13681) (paper, 2021)
- [The Next Big Thing: The Use of Graph Neural Networks to Discover New Particles](https://news.fnal.gov/2020/09/the-next-big-thing-the-use-of-graph-neural-networks-to-discover-particles/) (article, 2021)
- [Graph Neural Networks for Particle Reconstruction in High Energy Physics](https://inspirehep.net/literature/1788428) (paper, 2020)

### GNN Tracking
#### Exa.TrkX ([git repo](https://exatrkx.github.io/))
- [Physics and Computing Performance of the Exa.TrkX TrackML Pipeline](https://arxiv.org/abs/2103.06995) (paper, 2021)
- [Graph Neural Network for Object Reconstruction in Liquid Argon Time Projection Chambers](https://arxiv.org/abs/2103.06233) (paper, 2021)
#### IRIS-HEP Accelerated GNN Tracking ([project homepage](https://iris-hep.org/projects/accel-gnn-tracking.html))
- [Charged Particle Tracking via Edge-Classifying Interaction Networks](http://inspirehep.net/record/1854743) (paper, 2021)
- [Instance Segmentation GNNs for One-Shot Conformal Tracking at the LHC](http://inspirehep.net/record/1851132) (paper, 2020)
- [Acelerated Charged Particle Tracking with Graph Neural Networks on FPGAs](http://inspirehep.net/record/1834621)
#### HEP.TrkX ([git repo](https://heptrkx.github.io/))
- [Charged Particle Tracking Using Graph Neural Networks](https://indico.cern.ch/event/742793/contributions/3274328/) (talk, 2019)
- [Novel Deep Learning Methods for Track Reconstruction](https://arxiv.org/abs/1810.06111) (paper, 2018)

### GNN Calorimetry
- [Distance-Weighted Graph Neural Networks on FPGAs for Real-Time Particle Reconstruction in High Energy Physics](https://arxiv.org/abs/2008.03601) (paper, 2020)
    - PyG Implementation: [GravNetConv]( https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GravNetConv)
- [Learning representations of irregular particle-detector geometry with distance-weighted graph networks](https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9) (paper, 2019)

### GNN Jet Classification
- [Supervised Jet Clustering with Graph Neural Networks for Lorentz Boosted Bosons](https://arxiv.org/abs/2008.06064) (paper, 2020)
- [Casting a Graph Net to Detect Dark Showers](https://arxiv.org/abs/2006.08639) (paper, 2020)
- [Particle Net: Jet Tagging via Point Clouds](https://arxiv.org/abs/1902.08570) (paper, 2919)
- [Energy Flow Networks: Deep Sets for Particle Physics](https://arxiv.org/abs/1810.05165) (paper, 2018)

### GNN Event Classification
- [Probing Stop Pair Production at the LHC using Graph Neural Networks](https://link.springer.com/article/10.1007%2FJHEP08%282019%29055) (paper, 2019)
- [Graph Neural Networks for IceCube Signal Classification](https://arxiv.org/abs/1809.06166) (paper, 2018)

### GNN Pileup Mitigation 
- [ABC-Net: An Attention-Based Method for Particle Tagging](https://link.springer.com/article/10.1140%2Fepjp%2Fs13360-020-00497-3) (paper, 2020)
- [Pileup Mitigation at the Large Hadron Collider with Graph Neural Networks](https://arxiv.org/abs/1810.07988) (paper, 2018)

### GNN Particle Flow Reconstruction
- [MLPF: efficient machine-learned particle-flow reconstruction using graph neural networks](https://link.springer.com/article/10.1140/epjc/s10052-021-09158-w) (paper, 2021)
- [Object condensation: one-stage grid-free multi-object reconstruction in physics detectors, graph and image data](https://arxiv.org/abs/2002.03605) (paper, 2020)














