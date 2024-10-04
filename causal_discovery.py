from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci 
from causallearn.search.FCMBased import lingam
import networkx as nx
import matplotlib.pyplot as plt 

def discover_graph(dataset, method, alpha):
    if method == 'pc' :
        cg = pc(dataset.to_numpy(), aplha = alpha ,indep_test='fisherz') # change depending upon the computational resources and requirements 
    elif method == 'fci' :
        cg, edge = fci(dataset.to_numpy(),independence_test_method='fisherz', alpha = alpha,max_path_length=2)
    elif method == 'lingam' :
        cg = lingam.DirectLiNGAM(random_state=32)
        cg.fit(dataset.to_numpy())
    return cg

def edge_list(graph, method) :
    if method  == 'pc':
        adj = graph.find_adj()
        edge_index = [ (int(x), int(y)) for x,y in adj]
    elif method == 'fci' :
        adj = graph.graph
        nx_graph = nx.from_numpy_array(adj)
        edge_index = list(nx_graph.edges())
    elif method == 'lingam' :
        adj = graph.adjacency_matrix_
        nx_graph = nx.from_numpy_array(adj)
        edge_index = list(nx_graph.edges())
    return edge_index

def interpreter(graph,method) :
    if method == 'pc' :
        graph.draw_pydot_graph()
    elif method == 'fci' :
        adj = graph.graph 
        nx_graph = nx.from_numpy_array(adj)
        plt.figure(figsize=(10,10))
        nx.draw_networkx(nx_graph, arrows = True , with_labels= True)
        plt.show()
    elif method == 'lingam' :
        adj = graph.adjacency_matrix_
        nx_graph = nx.from_numpy_array(adj)
        plt.figure(figsize=(10,10))
        nx.draw_networkx(nx_graph, arrows = True , with_labels = True)
        plt.show()