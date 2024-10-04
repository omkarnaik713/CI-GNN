import numpy as np 
import networkx as nx 
import torch
import pandas as pd 

def convert_df(data) :
    graph_data = [rep.reshape(-1) for rep in data]
    df = pd.DataFrame(graph_data)
    
    for col in df.columns: 
        df[col] = df[col].astype(float)
    
    return df

def padding(graph, diff, edge_index):
    padding = np.zeros((diff,10))
    graph_x = graph.x.numpy()
    padded_graph = torch.tensor(np.concatenate((graph_x,padding), axis = 0))
    graph.edge_index = edge_index
    graph.x = padded_graph
    return graph

def add_padding(dataset, max_nodes, edge_index) :
    updated_data = []
    for graph in dataset :
        if graph.num_nodes != max_nodes :
            diff = max_nodes - graph.num_nodes
            padded_graph = padding(graph, diff, edge_index)
            updated_data.append(padded_graph)
        else :
            updated_data.append(graph)
    return updated_data