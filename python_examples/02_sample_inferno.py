#!/usr/bin/env python
import torch
import torch.nn as nn
import torchex.nn as exnn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from frontier_graph import NetworkxInterface

from inferno.extensions.layers.reshape import Concatenate
from inferno.extensions.containers import Graph



def linear():
    return exnn.Linear(16)

def conv():
    return exnn.Conv2d(16, 1)

def flatten():
    return exnn.Flatten()

def construct_module(graph, edge_indices, starts, ends):
    module = Graph()
    for i in starts:
        module.add_input_node(f'{i}')

    node_dict = {}
    # register parent nodes
    for (src, dst) in [list(graph.edges())[i-1] for i in edge_indices]:
        if not dst in node_dict.keys():
            node_dict[dst] = [src]
        else:
            node_dict[dst].append(src)

    for key, previous in sorted(node_dict.items(), key=lambda x: x[0]):
        flag = np.random.randint(3)
        if len(previous) != 1:
            mod = Concatenate()
            module.add_node(f'{key}', mod, previous=[str(p) for p in previous])            
            continue
        if flag == 0:
            mod = linear()
        elif flag == 1:
            mod = flatten()
        else:
            mod = conv()            
        module.add_node(f'{key}', mod, previous=[str(p) for p in previous])

    x1 = torch.randn(1, 1, 14, 14)
    x2 = torch.randn(1, 1, 14, 14)
    module(x1, x2)
    return module

if __name__ == '__main__':
    graph = nx.DiGraph()
    starts = [1, 2]
    ends = [9,]
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 5)
    graph.add_edge(3, 6)
    graph.add_edge(4, 5)
    graph.add_edge(4, 7)
    graph.add_edge(5, 6)
    graph.add_edge(5, 7)
    graph.add_edge(5, 8)
    graph.add_edge(6, 8)
    graph.add_edge(7, 8)
    graph.add_edge(8, 9)
    ns = NetworkxInterface(graph)
    graphs = ns.sample(starts, ends, 100)
    # caution: edge_indices starts from 1
    for i in range(len(graphs)):
        g = construct_module(graph, graphs[i], starts, ends)        
