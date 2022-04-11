# PyTorch Geometric 

Geometric deep learning (GDL) is an emerging field focused on applying machine learning (ML) techniques to non-Euclidean domains such as graphs, point clouds, and manifolds. The PyTorch Geometric (PyG) library extends PyTorch to include GDL functionality, for example classes necessary to handle data with irregular structure. PyG is introduced at a high level in [Fast Graph Representation Learning with PyTorch Geometric](https://arxiv.org/abs/1903.02428) and in detail in the [PyG docs](https://pytorch-geometric.readthedocs.io/en/latest/). 

## GDL with PyG 
A complete reveiw of GDL is available in the following recently-published (and freely-available) textbook: [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478). The authors specify several key GDL architectures including convolutional neural networks (CNNs) operating on grids, Deep Sets architectures operating on sets, and graph neural networks (GNNs) operating on graphs, collections of nodes connected by edges. PyG is focused in particular on graph-structured data, which naturally encompases set-structured data. In fact, many state-of-the-art GNN architectures are implemented in PyG (see [the docs](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html))! A review of the landscape of GNN architectures is available in [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434). 


### The *Data* Class: PyG Graphs 
Graphs are data structures designed to encode data structured as a set of objects and relations. Objects are embedded as graph nodes $u\in\mathcal{V}$, where $\mathcal{V}$ is the node set. Relations are represented by edges $(i,j)\in\mathcal{E}$ between nodes, where $\mathcal{E}$ is the edge set. Denote the sizes of the node and edge sets as $|\mathcal{V}|=n_\mathrm{nodes}$ and $|\mathcal{E}|=n_\mathrm{edges}$ respectively. The choice of edge connectivity determines the local structure of a graph, which has important downstream effects on graph-based learning algorithms. Graph construction is the process of embedding input data onto a graph structure. Graph-based learning algorithms are correspondingly imbued with a relational inductive bias based on the choice of graph representation; a graph's edge connectivity defines its local structure. The simplest graph construction routine is to construct no edges, yielding a permutation invariant set of objects. On the other hand, fully-connected graphs connect every node-node pair with an edge, yielding $n_\mathrm{edges}=n_\mathrm{nodes}(n_\mathrm{nodes}-1)/2$ edges. This representation may be feasible for small inputs like particle clouds corresponding to a jet, but is intractible for large-scale applications such as high-pileup tracking datasets. Notably, dynamic graph construction techniques operate on input point clouds, constructing edges on them dynamically during inference. For example, EdgeConv and GravNet GNN layers dynamically construct edges between nodes projected into a latent space; multiple such layers may be applied in sequence, yielding many intermediate graph representations on an input point cloud.

In general, nodes can have positions $\{p_i\}_{i=1}^{n_\mathrm{nodes}}$, $p_i\in\mathbb{R}^{n_\mathrm{space\_dim}}$, and features (attributes) $\{x_i\}_{i=1}^{n_\mathrm{nodes}}$, $x_i\in\mathbb{R}^{n_\mathrm{node\_dim}}$. In some applications like GNN-based particle tracking, node positions are taken to be the features. In others, e.g. jet identification, positional information may be used to seed dynamic graph consturction while kinematic features are propagated as edge features. Edges, too, can have features $\{e_{ij}\}_{(i,j)\in\mathcal{E}}$, $e_{ij}\in\mathbb{R}^{n_\mathrm{edge\_dim}}$, but do not have positions; instead, edges are defined by the nodes they connect, and may therefore be represented by, for example, the distance between the respective node-node pair. In PyG, graphs are stored as instances of the `data` class, whose fields fully specify the graph:

- `data.x`: node feature matrix, $X\in\mathbb{R}^{n_\mathrm{nodes}\times n_\mathrm{node\_dim}}$
- `data.edge_index`: node indices at each end of each edge, $I\in\mathbb{R}^{2\times n_\mathrm{edges}}$ 
- `data.edge_attr`: edge feature matrix, $E\in\mathbb{R}^{n_\mathrm{edges}\times n_\mathrm{edge\_dim}}$ 
- `data.y`: training target with arbitary shape ($y\in\mathbb{R}^{n_\mathrm{nodes}\times n_\mathrm{out}}$ for node-level targets, $y\in\mathbb{R}^{n_\mathrm{edges}\times n_\mathrm{out}}$ for edge-level targets or $y\in\mathbb{R}^{1\times n_\mathrm{out}}$ for node-level targets). 
- `data.pos`: Node position matrix, $P\in\mathbb{R}^{n_\mathrm{nodes}\times n_\mathrm{space\_dim}}$
 
 
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

$$x_i^{(k)} = \gamma^{(k)} \left(x_i^{(k-1)}, \square_{j \in \mathcal{N}(i)} \, \phi^{(k)}\left(x_i^{(k-1)}, x_j^{(k-1)},e_{ij}\right) \right)$$

Here, $\gamma$ and $\phi$ are learnable functions (which we can approximate as multilayer perceptrons), $\square$ is a permutation-invariant function (e.g. mean, max, add), and $\mathcal{N}(i)$ is the neighborhood of node $i$. In PyG, you'd write your own MPNN by using the `MessagePassing` base class, implementing each of the above mathematical objects as an explicit function. 

- `MessagePassing.message()` : define an explicit NN for $\phi$, use it to calculate "messages" between a node $x_i^{(k-1)}$ and its neighbors $x_j^{(k-1)}$, $j\in\mathcal{N}(i)$, leveraging edge features $e_{ij}$ if applicable
- `MessagePassing.propagate()` : in this step, messages are calculated via the `message` function and aggregated across each receiving node; the keyword `aggr` (which can be `'add'`, `'max'`, or `'mean'`) is used to specify the specific permutation invariant function $\square_{j\in\mathcal{N}(i)}$ used for message aggregation. 
- `MessagePassing.update()` : the results of message passing are used to update the node features $x_i^{(k)}$ through the $\gamma$ MLP 

The specific implementations of `message()`, `propagate()`, and `update()` are up to the user. A specific example is available in the PyG [Creating Message Passing Networks tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)

#### Message-Passing with ZINC Data
Returning to the ZINC molecular compound dataset, we can design a message-passing layer to aggregate messages across molecular graphs. Here, we'll define a multi-layer perceptron (MLP) class and use it to build a message passing layer (MPL) the following equation:

$$x_i' = \gamma \left(x_i, \frac{1}{|\mathcal{N}(i)|}\sum_{j \in \mathcal{N}(i)} \, \phi\left([x_i, x_j, e_{j,i}\right]) \right)$$

Here, the MLP dimensions are constrained. Since $x_i, e_{i,j}\in\mathbb{R}$, the $\phi$ MLP must map $\mathbb{R}^3$ to $\mathbb{R}^\mathrm{message\_size}$. Similarly, $\gamma$ must map $\mathbb{R}^{1+\mathrm{\mathrm{message\_size}}}$ to $\mathbb{R}^\mathrm{out}$. 
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