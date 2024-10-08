import torch
import torch.nn.functional as F
from torch_geometric.datasets import BA2MotifDataset
from GNNModel import model
from utils import convert_df,add_padding
from causal_discovery import discover_graph, interpreter, edge_list
from GNNClassifier import classifier, train
from torch_geometric.loader import DataLoader

# Hyper-parameters 
method = 'lingam' # other options are 'pc' , 'fci' 
alpha = 0.1 # confidence interval
hidden_dim = 256
lr = 0.0005
output_dim = 1
#node_rep_layers = 3
#classifier_layers = 5

# loading the dataset which has 1000 graphs
print('Loading the dataset')
dataset = BA2MotifDataset(root='/Users/omkarnaik/CI-GNN/Data')
input_dim = dataset[0].x.size(1)
output_dim = 1
motif_dataset = DataLoader(dataset, batch_size=32,shuffle = True )
model = model(input_dim,hidden_dim ,output_dim)
max_nodes = 25 # depends on the graph dataset you are working with 
padded_graphs = add_padding(dataset,max_nodes, dataset.edge_index)
padded_dataset = DataLoader(padded_graphs,batch_size=32, shuffle = True)

classifier_model = classifier(input_dim, hidden_dim)

loss, accuracy = train(classifier_model,model,padded_dataset,dataset, method , alpha)

print('Accuracy : ', accuracy)
print('Loss : ', loss)
# display the causal graph 
