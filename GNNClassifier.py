import torch
import torch.nn as nn 
from torch_geometric.nn import GINConv
import torch.nn.functional as F 
from torch_geometric.nn import global_mean_pool
from utils import convert_df
from torch_geometric.loader import DataLoader
from causal_discovery import discover_graph, edge_list

class classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(classifier, self).__init__()
        self.gnn_layers = nn.ModuleList()
        
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2, bias = False),
            nn.BatchNorm1d(hidden_dim//2)
        ),train_eps=True))
        
        for i in range(5):
            self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2, bias = False),
            nn.BatchNorm1d(hidden_dim//2)
            ),train_eps=True))
        
        self.gnn_layers.append(GINConv(nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim, bias = False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2, bias = False)
        ),train_eps=True))
        
        self.gnn_non_linear = nn.ReLU()
        
        
    def forward(self, x , edge_index,batch=None ):
        for conv in self.gnn_layers:
            x = conv(x,edge_index)
            x = F.dropout(x, p = 0.2, training = True)
            x = self.gnn_non_linear(x)
        x = global_mean_pool(x,batch)
        return x
        
def train(classifier,model, loader,og_data ,method, alpha):
    opt = torch.optim.Adam(model.parameters(),lr = 0.0005, weight_decay= 5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.0005, weight_decay=5e-4)
    correct = 0
    total_loss = 0
    total = 0
    for i in range(200):
        if i < 50 :
            model.train()
            total_loss = 0
            for batch in loader :
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                dataset = DataLoader(convert_df(out), batch_size = 32,shuffle = False)
                cl_out = classifier(batch.x, batch.edge_index, batch= batch.batch )
                pred = torch.argmax(cl_out, dim = 1)
                loss = loss_fn(cl_out, batch.y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            print(total_loss)
        else : 
            model.eval()
            classifier.train()
            node_rep = []
            for graph in og_data:
                rep = model(graph.x, graph.edge_index)
                node_rep.append(rep)
            dataset = convert_df(node_rep)
            dataset['label'] = og_data.y
            print(dataset)
            if i == 50 :
                print('Classifier Training starts from here')
                total_loss = 0
                
                cg = discover_graph(dataset, method, alpha)
                edge_index = edge_list(cg, method)
                edge_index = torch.tensor(edge_index, dtype = torch.int64)
                edge_index = edge_index.T
                print(edge_index)
            for batch in loader : 
                optimizer.zero_grad()
                
                out_cl = classifier(batch.x, edge_index, batch.batch)
                pred = torch.argmax(out_cl, dim = 1)
                loss = loss_fn(out_cl, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (pred==batch.y).sum().item()
                total += batch.y.size(0)
            print(total_loss)
            print(correct)
    return total_loss/150 , correct/total