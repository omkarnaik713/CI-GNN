import torch
import torch.nn as nn 
from torch_geometric.nn import GINConv
import torch.nn.functional as F 
from torch_geometric.nn import global_mean_pool

class classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(classifier, self).__init__()
        self.gnn_layers = nn.ModuleList()
        
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim, bias = False),
            nn.BatchNorm1d(2*hidden_dim)
        ),train_eps=True))
        
        for i in range(5):
            self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim, bias = False),
            nn.BatchNorm1d(2*hidden_dim)
            ),train_eps=True))
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2, bias = False)
        ),train_eps=True))
        
        self.gnn_non_linear = nn.ReLU()
        
        
    def forward(self, x , edge_index, edge_attribute,batch ):
        for conv in self.gnn_layers:
            x = conv(x,edge_index,edge_attribute)
            x = F.dropout(x)
            x = self.gnn_non_linear(x)
        x = global_mean_pool(x,batch)
        return x
        
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0 
    correct = 0 
    total = 0
    for epoch in range(250):
        for batch in loader :
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr,batch.batch)
            pred = torch.argmax(out, dim = 1)
            print('Pred', pred)
            loss = loss_fn(out,batch.y)
            print(batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (pred==batch.y).sum().item()
            total += batch.y.size(0)
    print('Total Loss ', total_loss)
    print('Total', total)
    print('correct', correct)
    return (total_loss)/len(loader),(correct/total)



