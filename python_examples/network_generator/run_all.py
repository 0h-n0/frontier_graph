import networkx as nx

from typing import List

from test_data_generator import make_graph, generate_random_graph, generate_graph
from frame_generator import FrameGenerator
from graph_generator import GraphGenerator
from module_generator import NNModuleGenerator, plot_graph
import torchvision.models as models

from torchviz import make_dot
import torch

if __name__ == "__main__":
    g, starts, ends = generate_graph(4, 3)
    print(g.edges)
    input_size = 28
    fg = FrameGenerator(g, starts, ends)
    x = torch.rand(1, 3, input_size, input_size)
    dryrun_args = (x,) * len(starts)
    total_found = 0
    for idx in range(100):
        graph = fg.random_sample()
        gg = GraphGenerator(graph, starts, ends, input_size, True)
        if len(gg.g_compressed.nodes) <= 3:
            continue

        print(gg.g_compressed.edges)
        # s = gg.list_valid_output_sizes(input_size)
        s = [gg.sample_valid_output_size(input_size) for _ in range(100)]
        opt = max(s, key=lambda x: len(set(x.values())) * (max(x.values()) - min(x.values())))
        print(f"found {len(s)} networks")
        mg = NNModuleGenerator(graph, starts, ends, input_size, opt)
        print(opt)
        module = mg.run()
        print(f"---example---\noutout sizes:{opt}\nnetwork:{module}")
        out = module(*dryrun_args)
        dot = make_dot(out)
        dot.format = 'png'
        dot.render(f'test_outputs/graph_image_{idx}')
        total_found += len(s)

    print(f"found total {total_found} graphs")
