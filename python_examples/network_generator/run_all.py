import networkx as nx

from typing import List

from test_data_generator import generate_graph
from frame_generator import FrameGenerator
from graph_generator import GraphGenerator
from module_generator import NNModuleGenerator

from torchviz import make_dot
import torch

if __name__ == "__main__":
    g, starts, ends = generate_graph(4, 2)
    kernel_sizes = [1, 2, 3]
    strides = [1, 2, 3]
    output_channel_candidates = [32, 64, 128, 192]
    input_sizes = [28, 28, 56, 112]

    fg = FrameGenerator(g, starts, ends)
    dryrun_args = tuple([torch.rand(1, 3, s, s) for s in input_sizes])

    for idx in range(100):
        frame = fg.random_sample()
        gg = GraphGenerator(frame, starts, ends, max(input_sizes), True, kernel_sizes, strides)
        # あまりに単純な場合は省く
        if len(gg.g_compressed.nodes) <= 3: continue

        output_sizes = [gg.sample_valid_output_size(input_sizes) for _ in range(100)]
        opt = max(output_sizes, key=lambda x: len(set(x.values())) * (max(x.values()) - min(x.values())))
        mg = NNModuleGenerator(frame, starts, ends, input_sizes, opt, kernel_sizes, strides, output_channel_candidates)
        module = mg.run()

        print(gg.g_compressed.edges)
        print(f"found {len(output_sizes)} networks")
        print(opt)
        print(f"---example---\noutout sizes:{opt}\nnetwork:{module}")
        out = module(*dryrun_args)
        dot = make_dot(out)
        dot.format = 'png'
        dot.render(f'test_outputs/graph_image_{idx}')
