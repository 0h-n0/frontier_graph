import networkx as nx

from typing import List

from test_data_generator import make_graph, generate_graph
from frame_generator import FrameGenerator
from graph_generator import GraphGenerator
from module_generator import ModuleGenerator, NNModuleGenerator, plot_graph

import torch

if __name__ == "__main__":
    n_nodes = 15
    starts = [1, 2]
    ends = [n_nodes]
    g = generate_graph(n_nodes, starts, ends, 0.1)
    print(g.edges)
    input_size = 28
    fg = FrameGenerator(g, starts, ends)
    l = fg.list_valid_graph()
    x = torch.rand(1, 3, 28, 28)
    dryrun_args = (x, x)
    total_found = 0
    for idx, graph in enumerate(l):
        print("===============")
        print(graph.edges)
        gg = GraphGenerator(graph, starts, ends, input_size)
        s = gg.list_valid_output_sizes(input_size)
        opt = max(s, key=lambda x: len(set(x.values())) * (max(x.values()) - min(x.values())))
        print(f"found {len(s)} networks")
        mg = NNModuleGenerator(graph, starts, ends, input_size, opt)
        module = mg.run()
        y = module(*dryrun_args)
        print(f"---example---\noutout sizes:{opt}\nnetwork:{module}")

        total_found += len(s)

    print(f"found total {total_found} graphs")
