import networkx as nx
from inferno.extensions.containers import Graph
from inferno.extensions.layers.reshape import Concatenate
import torchex.nn as exnn
import torch.nn as nn

from typing import List, Dict

from layer import find_conv_layer, conv2d, ConcatConv, FlattenLinear, ConcatFlatten


# constructorがまあまあなロジック持ってるけどどうしよう
class NNModuleGenerator():
    """
    与えられたグラフなどからネットワークを作るためのクラス
    ------------
    Attributes(分かりにくそうなもののみ記載):
       network_input_sizes: (各入力のnodeについて)nodeの番号がkey, 入力サイズがvalueのdict 
       node_output_sizes: (各nodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
       network_output_sizes: (各出力のnodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
       network_output_dimensions: (各出力のnodeについて)nodeの番号がkey, 出力の次元がvalueのdict 
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        network_input_sizes: Dict[int, int],
        node_output_sizes: Dict[int, int],
        node_output_dimensions: Dict[int, int],
        network_output_sizes: Dict[int, int],
        kernel_sizes: List[int],
        strides: List[int],
        output_channel_candidates: List[int]
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.network_input_sizes = network_input_sizes
        self.node_output_sizes = node_output_sizes
        self.node_output_dimensions = node_output_dimensions
        self.network_output_sizes = network_output_sizes
        self.node_input_sizes = self.get_input_sizes()
        self.node_input_dimensions = self.get_input_dimensions()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.output_channels = self.__calc_output_channels(output_channel_candidates)

    def is_concat_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) <= 1 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_concat_conv_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] != self.node_input_sizes[v] and self.node_output_dimensions[v] == 4

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] == self.node_input_sizes[v]

    def is_linear_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) == 1 and self.node_output_dimensions[v] == 1 and self.node_input_dimensions[v] == 1

    def get_input_sizes(self):
        input_sizes = {}
        for v, s in self.network_input_sizes.items():
            input_sizes[v] = s
        for s, t in self.g.edges:
            input_sizes[t] = self.node_output_sizes[s]
        return input_sizes

    def get_input_dimensions(self):
        input_dimensions = {}
        for v in self.starts:
            input_dimensions[v] = 4
        for s, t in self.g.edges:
            input_dimensions[t] = self.node_output_dimensions[s]
        return input_dimensions

    def add_layer(self, v: int, module):
        previous_nodes = [f"{u}" for (_, u) in self.g_inv.edges([v])]
        out_channels = self.output_channels[v]
        if v in self.starts:
            module.add_input_node(f"{v}")
        elif v in self.ends:
            module.add_node(f"{v}", previous=previous_nodes, module=FlattenLinear(self.network_output_sizes[v]))
        elif self.is_concat_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=ConcatFlatten(out_channels))
        elif self.is_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=FlattenLinear(out_channels))
        elif self.is_concat_conv_node(v):
            k, s = find_conv_layer(self.node_input_sizes[v], self.node_output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(
                f"{v}", previous=previous_nodes, module=ConcatConv(out_channels=out_channels, kernel_size=k, stride=s))
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Concatenate())
        elif self.is_linear_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=exnn.Linear(out_channels))
        elif self.node_output_sizes[v] == self.node_input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=nn.ReLU())
        else:
            k, s = find_conv_layer(self.node_input_sizes[v], self.node_output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(
                f"{v}", previous=previous_nodes, module=conv2d(out_channels=out_channels, kernel_size=k, stride=s))

    def __calc_output_channels(self, output_channel_candidates: List[int]):
        """
        親のoutput_channelsの和以下のものがcandidatesにあったらその内最大のものを採用。
        そうでないときはmin(candidates)を採用
        """
        output_channels = {v: 3 for v in self.starts}
        self.g_inv = self.g.reverse()
        for v in sorted(list(self.g.nodes)):
            if v in self.starts: continue
            sum_inputs = sum([output_channels[u] for (_, u) in self.g_inv.edges(v)])
            if sum_inputs < min(output_channel_candidates):
                output_channels[v] = min(output_channel_candidates)
            else:
                output_channels[v] = max(filter(lambda x: x <= sum_inputs, output_channel_candidates))
        return output_channels

    def run(self):
        module = Graph()
        for v in sorted(list(self.g.nodes)):
            self.add_layer(v, module)
        module.add_node('concat', previous=[f"{t}" for t in self.ends], module=Concatenate())
        module.add_output_node('output', previous='concat')
        return module
