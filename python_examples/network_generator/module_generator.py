import networkx as nx

from typing import List, Dict
import matplotlib.pyplot as plt

from inferno.extensions.containers import Graph
from inferno.extensions.layers.reshape import Concatenate

import torchex.nn as exnn
import torch.nn as nn

from layer import find_conv_layer, conv2d, ConcatConv, FlattenLinear


class NNModuleGenerator():
    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        input_size: int,
        output_sizes: Dict[int, int],
        kernel_sizes: List[int],
        strides: List[int]
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.input_sizes = self.get_input_sizes()
        self.kernel_sizes = kernel_sizes
        self.strides = strides

    def is_concat_conv_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.output_sizes[v] != self.input_sizes[v]

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.output_sizes[v] == self.input_sizes[v]

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
        elif v in self.ends:
            module.add_node(f"{v}", previous=previous_nodes, module=FlattenLinear(10))
        elif self.is_concat_conv_node(v):
            k, s = find_conv_layer(self.input_sizes[v], self.output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(f"{v}", previous=previous_nodes, module=ConcatConv(out_channels=3, kernel_size=k, stride=s))
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Concatenate())
        elif self.output_sizes[v] == self.input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=nn.ReLU())
        else:
            k, s = find_conv_layer(self.input_sizes[v], self.output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(f"{v}", previous=previous_nodes, module=conv2d(out_channels=3, kernel_size=k, stride=s))

    def run(self):
        module = Graph()
        for v in sorted(list(self.g.nodes)):
            self.add_layer(v, module)
        module.add_output_node('output', previous=[f"{t}" for t in self.ends])
        return module
