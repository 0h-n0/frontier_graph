import networkx as nx

from typing import List, Dict, Callable

from test_data_generator import generate_graph
from frame_generator import FrameGenerator
from output_size_searcher import OutputSizeSearcher
from module_generator import NNModuleGenerator

from torchviz import make_dot
import torch


def calc_network_quality(output_sizes: Dict[int, int], output_dims: Dict[int, int]) -> int:
    """ サイズの変化とバリエーションで評価 """
    return len(set(output_sizes.values())) * (max(output_sizes.values()) - min(output_sizes.values()))


def list_networks(
    g: nx.DiGraph,
    starts: List[int],
    ends: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    output_channel_candidates: List[int],
    network_input_sizes: Dict[int, int],
    network_output_sizes: Dict[int, int],
    allow_param_in_concat: bool,
    n_networks: int,
    n_network_candidates: int,
    calc_network_quality: Callable[[Dict[int, int], Dict[int, int]], int]
):
    """
    有効なnetworkをn_networks件列挙します。
    1件に対しn_network_candidates件候補をあげ、calc_network_qualityの値が最大のものを選びます
    """
    networks = []
    for _ in range(n_networks):
        frame = fg.sample_graph()
        oss = OutputSizeSearcher(frame, starts, ends, max(network_input_sizes.values()),
                                 allow_param_in_concat, kernel_sizes, strides)

        output_sizes = []
        for _ in range(n_network_candidates):
            output_dimensions = oss.sample_output_dimensions()
            result = oss.sample_valid_output_size(network_input_sizes, output_dimensions)
            if result == False: break
            else: output_sizes.append((result, output_dimensions))

        if len(output_sizes) == 0: continue

        opt_sizes, opt_dims = max(output_sizes, key=lambda x: calc_network_quality(x[0], x[1]))
        # print(frame.edges)
        # print(opt_sizes)
        # print(opt_dims)
        mg = NNModuleGenerator(frame, starts, ends, network_input_sizes, opt_sizes, opt_dims, network_output_sizes,
                               kernel_sizes, strides, output_channel_candidates)

        module = mg.run()
        networks.append(module)
    return networks


if __name__ == "__main__":
    g, starts, ends = generate_graph(1, 12, 13, 1)
    kernel_sizes = [1, 2, 3]
    strides = [1, 2, 3]
    output_channel_candidates = [32, 64, 128, 192]
    network_input_sizes = {v: 224 for v in starts}
    network_output_sizes = {v: 1 for v in ends}
    allow_param_in_concat = True

    fg = FrameGenerator(g, starts, ends)
    dryrun_args = tuple([torch.rand(1, 3, s, s) for s in network_input_sizes.values()])

    networks = list_networks(
        g, starts, ends, kernel_sizes, strides,
        output_channel_candidates, network_input_sizes,
        network_output_sizes, allow_param_in_concat,
        100, 100, calc_network_quality)

    for idx, network in enumerate(networks):
        print(network)
        out = network(*dryrun_args)
        dot = make_dot(out)
        dot.format = 'png'
        dot.render(f'test_outputs/graph_image_{idx}')
