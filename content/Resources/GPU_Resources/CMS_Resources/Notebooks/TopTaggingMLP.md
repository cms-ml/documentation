```python
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import uproot3
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import awkward0
```
```python
class MultiLayerPerceptron(nn.Module):
    r"""Parameters
    ----------
    input_dims : int
        Input feature dimensions.
    num_classes : int
        Number of output classes.
    layer_params : list
        List of the feature size for each layer.
    """

    def __init__(self, input_dims, num_classes,
                 layer_params=(256,64,16),
                 **kwargs):
                
        super(MultiLayerPerceptron, self).__init__(**kwargs)
        channels = [input_dims] + list(layer_params) + [num_classes]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Sequential(nn.Linear(channels[i], channels[i + 1]),
                                        nn.ReLU()))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: the feature vector initally read from the data structure, in dimension (N, C, P)
        x = x.flatten(start_dim=1) # (N, L), where L = C * P
        return self.mlp(x)
    
    def predict(self,x):
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(1)
            else:
                ans.append(0)
        return torch.tensor(ans)
```


```python
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args["dry_run"]:
                break
```


```python
input_branches = [
                  'Part_Etarel',
                  'Part_Phirel',
                  'Part_E_log',
                  'Part_P_log'
                 ]

output_branches = ['is_signal_new']
```


```python
train_dataset = uproot3.open("TopTaggingMLP/train.root")["Events"].arrays(input_branches+output_branches,namedecode='utf-8')
train_dataset = {name:train_dataset[name].astype("float32") for name in input_branches+output_branches}
test_dataset = uproot3.open("/eos/user/c/coli/public/weaver-benchmark/top_tagging/samples/prep/top_test_0.root")["Events"].arrays(input_branches+output_branches,namedecode='utf-8')
test_dataset = {name:test_dataset[name].astype("float32") for name in input_branches+output_branches}
```


```python
for ds in [train_dataset,test_dataset]:
    for name in ds.keys():
        if isinstance(ds[name],awkward0.JaggedArray):
            ds[name] = ds[name].pad(30,clip=True).fillna(0).regular().astype("float32")
```


```python
class PF_Features(Dataset):
    def __init__(self,mode = "train"):
        if mode == "train":
            self.x = {key:train_dataset[key] for key in input_branches}
            self.y = {'is_signal_new':train_dataset['is_signal_new']}
        elif mode == "test":
            self.x = {key:test_dataset[key] for key in input_branches}
            self.y = {'is_signal_new':test_dataset['is_signal_new']}
        elif model == "val":
            self.x = {key:test_dataset[key] for key in input_branches}
            self.y = {'is_signal_new':test_dataset['is_signal_new']}
    
    def __len__(self):
        return len(self.y['is_signal_new'])
    
    def __getitem__(self,idx):
        X = [self.x[key][idx].copy() for key in input_branches]
        X = np.vstack(X)
        y = self.y['is_signal_new'][idx].copy()
        return X,y
```


```python
torch.cuda.is_available() # Check if cuda is available
```




    True




```python
device = torch.device("cuda")
```


```python
train_kwargs = {"batch_size":1000}
test_kwargs = {"batch_size":10}
cuda_kwargs = {'num_workers': 1,
               'pin_memory': True,
               'shuffle': True}
train_kwargs.update(cuda_kwargs)
test_kwargs.update(cuda_kwargs)
```


```python
model = MultiLayerPerceptron(input_dims = 4 * 30, num_classes=2).to(device)
```


```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```


```python
train_loader = torch.utils.data.DataLoader(PF_Features(mode="train"),**train_kwargs)
test_loader = torch.utils.data.DataLoader(PF_Features(mode="test"),**test_kwargs)
```


```python
loss_func = torch.nn.CrossEntropyLoss()
```


```python
args = {"dry_run":False, "log_interval":500}
for epoch in range(1,10+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs = data.to(device)#.flatten(start_dim=1)
        target = target.long().to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
```