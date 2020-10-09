#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from frontier_graph import NetworkxInterface


pos_dir = {
    '1': np.array([-0.25, 0.25]),
    '2': np.array([0.25, 0.25]),
    '3': np.array([-0.25, 0]),
    '4': np.array([0.25, 0]),
    '5': np.array([0, -0.25]),
    '6': np.array([-0.25, -0.50]),
    '7': np.array([0.25, -0.50]),
    '8': np.array([0, -0.75]),
    '9': np.array([0, -1.0]),
    '10': np.array([0, -1.25]),
    '11': np.array([0, -1.5]),                        
}


def construct_graph(graph, edge_indcies):
    edges = [[int(e[0]), int(e[1])] for e in graph.edges()]
    edges = [edges[i - 1] for i in edge_indcies]
    _g = nx.DiGraph()
    for e in edges:
        _g.add_edge(*e)
    return _g

def draw_graph(graph, filename='test.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-2, 1])
    ax.set_aspect('equal')
    pos = nx.spring_layout(graph)
    for k in pos_dir:
        pos[k] = pos_dir[k]
    pos = {n: pos_dir[str(n)] for n in graph.nodes()}
    labels = {n: str(n) for n in graph.nodes()}
    nx.draw(graph, pos, ax)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.savefig(filename)
    plt.clf()


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
    draw_graph(graph, 'original.png')
    graphs = ns.sample(starts, ends, 100)
    # caution: edge_indices starts from 1
    for i in range(len(graphs)):
        g = construct_graph(graph, graphs[i])        
        draw_graph(g, f'subgraph-{i:03}.png')
