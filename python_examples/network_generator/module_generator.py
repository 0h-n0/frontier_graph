import networkx as nx

from typing import List, Dict
import matplotlib.pyplot as plt

from inferno.extensions.containers import Graph
from inferno.extensions.layers.reshape import Concatenate

import torchex.nn as exnn
import torch.nn as nn

from layer import find_conv_layer, conv2d


class ModuleGenerator():
    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        input_size: int,
        output_sizes: Dict[int, int]
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.input_size = input_size
        self.output_sizes = output_sizes

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2

    def get_node_type(self, v: int) -> str:
        if self.is_concat_node(v):
            return "concat"

    def add_type_to_module(self, module: nx.DiGraph):
        for v in self.g.nodes:
            if v in self.starts:
                module.nodes[v]['type'] = 'input'
            elif self.is_concat_node(v):
                module.nodes[v]['type'] = 'concat'
            elif module.nodes[v]['output_size'] == module.nodes[v]['input_size']:
                module.nodes[v]['type'] = 'linear'
            else:
                module.nodes[v]['type'] = 'conv2d'

    def add_sizes_to_module(self, module: nx.DiGraph):
        for v in self.starts:
            module.nodes[v]['input_size'] = self.input_size
        for v in self.g.nodes:
            module.nodes[v]['output_size'] = self.output_sizes[v]
        for s, t in self.g.edges:
            module.nodes[t]['input_size'] = self.output_sizes[s]

    def run(self):
        module = nx.DiGraph()
        module.add_edges_from(self.g.edges)
        self.add_sizes_to_module(module)
        self.add_type_to_module(module)
        return module


class NNModuleGenerator():
    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        input_size: int,
        output_sizes: Dict[int, int]
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.input_sizes = self.get_input_sizes()

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2

    def get_input_sizes(self):
        input_sizes = {}
        for v in self.starts:
            input_sizes[v] = self.input_size
        for s, t in self.g.edges:
            input_sizes[t] = self.output_sizes[s]
        return input_sizes

    def add_layer(self, v: int, module):
        previous_nodes = [f"{u}" for (_, u) in self.g_inv.edges([v])]
        if v in self.starts:
            module.add_input_node(f"{v}")
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Concatenate())
        elif self.output_sizes[v] == self.input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=nn.ReLU())
        else:
            k, s = find_conv_layer(self.input_sizes[v], self.output_sizes[v], [1, 2, 3], [1, 2, 3])
            module.add_node(f"{v}", previous=previous_nodes, module=conv2d(out_channels=3, kernel_size=k, stride=s))

    def run(self):
        module = Graph()
        for v in sorted(list(self.g.nodes)):
            self.add_layer(v, module)
        module.add_output_node('output', previous=[f"{t}" for t in self.ends])
        return module


def plot_graph(module: nx.Graph, file_name: str):
    input_sizes = nx.get_node_attributes(module, 'input_size')
    output_sizes = nx.get_node_attributes(module, 'output_size')
    layer_types = nx.get_node_attributes(module, 'type')
    labels = {
        v: f"index:{v}\n" +
        f"input:({input_sizes[v]}, {input_sizes[v]})\n" +
        f"output:({output_sizes[v]}, {output_sizes[v]})\n" +
        f"type:{layer_types[v]}" for v in module.nodes
    }
    nx.draw(module, pos=nx.spectral_layout(module), labels=labels, font_size=6, node_shape="s")
    # nx.draw(g, labels=labels, pos=nx.spectral_layout(g))
    plt.savefig(f"tests/my_test/images/{file_name}.png")
    plt.clf()


if __name__ == "__main__":
    g = nx.DiGraph()
    starts = [1, 2]
    ends = [9, ]
    g.add_edges_from([(1, 3), (3, 5), (3, 6), (5, 7), (6, 8), (2, 4), (4, 7), (7, 8), (8, 9)])
    input_size = 28
    output_sizes = {1: 28, 3: 27, 5: 13, 6: 13, 2: 28, 4: 13, 7: 13, 8: 13, 9: 11}
    mg = ModuleGenerator(g, starts, ends, input_size, output_sizes)
    module = mg.run()
    plot_graph(module, "name")
