import torch
import torch.nn.functional as F
from torch_geometric.datasets import BA2MotifDataset
from GNNModel import model, train_rep
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
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, weight_decay=5e-3)
loss_fn = torch.nn.CrossEntropyLoss()
rep_model = train_rep(model, motif_dataset, optimizer, loss_fn )
max_nodes = 25 # depends on the graph dataset you are working with 

# calculating node representation 
node_representation = []
print('Calculating Node Representation')
for graph in dataset :
    with torch.no_grad() :
        rep = rep_model(dataset.x, dataset.edge_index)
        padded_rep = F.pad(rep, (0,0,0,max_nodes-rep.size(0)))
        node_representation.append(padded_rep)

graph_df = convert_df(node_representation)
graph_df['Label'] = dataset.y
print('Discovering Causal Graph')
# causal graph discovery 
causal_graph = discover_graph(graph_df,method, alpha)
edge_index = edge_list(causal_graph,method)
edge_index = torch.tensor(edge_index)
edge_index = edge_index.T
padded_graph = add_padding(dataset, max_nodes, edge_index)
padded_dataset = DataLoader(padded_graph, batch_size = 16, shuffle = True)
print('Performing Classification')
classifier_model = classifier(input_dim, hidden_dim)
optimizer = torch.optim.Adam(classifier_model.parameters(), lr = lr, weight_decay= 5e-4)
loss_fn = torch.nn.CrossEntropyLoss()
loss, accuracy = train(classifier_model,padded_dataset,optimizer, loss_fn)

print('Accuracy : ', accuracy)
print('Loss : ', loss)
# display the causal graph 
interpreter(causal_graph,method)
