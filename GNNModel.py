import torch.nn as nn
import torch
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool

class model(nn.Module) :
    '''This model takes in an input of the node features and returns a tensor of N*1 
    which is the node representation or we can say node embedding. The reason for converting into an N*1 
    node embedding was to make it simpler to pass it to the causal discovery model, 
    and since we are more concerned about the graph classification rather than node classification.
    
    Parameters : 
        input_dim : the number of features a node has
        hidden_dim : the number of hidden neurons 
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(model, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2*hidden_dim)
        self.conv3 = GCNConv(2*hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, 2)
        self.conv5 = GCNConv(2,1)
        
    def forward(self, x, edge_index,batch=None) : 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,p = 0.5)
        
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5)
        
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        if batch != None :
            x = global_mean_pool(x,batch)
        else : 
            x = self.conv5(x, edge_index)
            x = F.relu(x)
        return x
    
