#!/usr/bin/env python
import torch
import torch.nn as nn
import networkx as nx

from inferno.extensions.layers.reshape import Concatenate
from inferno.extensions.containers import Graph

module = Graph()
module.add_input_node('input', hello=3)
module.add_node('linear1', nn.Linear(3, 18), previous='input', hello=3)

g = module.graph
n1 = g.nodes(data=True)
print(g.nodes['input'])
print(g.nodes.data())
print(n1)
