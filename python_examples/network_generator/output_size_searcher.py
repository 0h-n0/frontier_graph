import networkx as nx
from networkx.algorithms.components import strongly_connected_components

import itertools
from typing import List, Dict

from test_data_generator import make_graph
from layer import conv_output_size

import random
from functools import reduce
from operator import and_, or_
from frame_generator import FrameGenerator


def make_size_transition_graph(max_size: int, kernel_sizes: List[int], strides: List[int]):
    g = nx.DiGraph()
    for x_in in range(1, max_size + 1):
        for s, k in itertools.product(kernel_sizes, strides):
            x_out = conv_output_size(x_in, k, s)
            if x_out > 0: g.add_edge(x_in, x_out)
    return g


# TODO 先に次元を決めておいた方がサボれるのでそうしたい。
class OutputSizeSearcher():
    """
    与えられるグラフについて、各nodeに有効な出力サイズを割り振ります。
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        max_input_size: int,
        allow_param_in_concat: bool,  # concat + conv などを許すか
        kernel_sizes: List[int],
        strides: List[int]
    ):
        self.g = g
        self.n_nodes = max(g.nodes)
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.allow_param_in_concat = allow_param_in_concat
        self.size_transision_graph = make_size_transition_graph(max_input_size, kernel_sizes, strides)
        self.scc_idx, self.g_compressed = self.compress_graph()
        self.g_compressed_inv = self.g_compressed.reverse()
        self.t_sorted = list(nx.topological_sort(self.g_compressed))

    def compress_graph(self):
        """ 出力サイズが同じ頂点を縮約したグラフを作る """
        g_for_scc = self.g.copy()
        for v in self.g.nodes:
            # vの入力があるnodeとvのoutput_sizeは同じ
            if self.is_concat_node(v):
                if self.allow_param_in_concat:
                    v_edges = list(self.g_inv.edges([v]))
                    for (_, f), (__, t) in zip(v_edges[:-1], v_edges[1:]):
                        g_for_scc.add_edge(f, t)
                        g_for_scc.add_edge(t, f)
                else:
                    for _, u in self.g_inv.edges([v]):
                        g_for_scc.add_edge(v, u)

        scc_idx = [0] * (self.n_nodes + 1)
        scc = strongly_connected_components(g_for_scc)
        for idx, nodes in enumerate(scc):
            for v in nodes:
                scc_idx[v] = idx

        g_compressed = nx.DiGraph()
        for v in self.g.nodes:
            g_compressed.add_node(scc_idx[v])
        for s, t in self.g.edges:
            rs = scc_idx[s]
            rt = scc_idx[t]
            if rs != rt: g_compressed.add_edge(rs, rt)

        return scc_idx, g_compressed

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2

    def __as_size_dict(self, size_list):
        """ self.t_sortedに対応するsizeを{vertex: size}のdictに変換する """
        size_dict = {}
        for v in self.g.nodes:
            rv = self.scc_idx[v]
            idx = self.t_sorted.index(rv)
            size_dict[v] = size_list[idx]
        return size_dict

    # topological順序で見るのでこれていい
    def __list_reachable_nodes_in_compressed_graph(self, v: int):
        """ 縮約されたグラフ上でvから到達可能な点を全て列挙します """
        reachabilities = [False] * len(self.t_sorted)
        reachable_nodes = [v]
        reachabilities[v] = True
        for u in self.t_sorted:
            if u == v or len(self.g_compressed_inv.edges([u])) == 0:
                continue
            is_reachable = reduce(or_, [reachabilities[f] for (_, f) in self.g_compressed_inv.edges([u])])
            if is_reachable:
                reachabilities[u] = True
                reachable_nodes.append(u)
        return reachable_nodes

    def sample_output_dimensions(self):
        middle_nodes = set(self.g.nodes) - (set(self.starts) - set(self.ends))
        middle_node_scc_indices = list({self.scc_idx[v] for v in middle_nodes})
        seed_node = random.choice(middle_node_scc_indices)
        one_dimensional_nodes = set(self.__list_reachable_nodes_in_compressed_graph(seed_node))
        return {v: 1 if self.scc_idx[v] in one_dimensional_nodes else 4 for v in self.g.nodes}

    # 次元はこの関数で決めるのではなくて外で決めて渡すインターフェースがいいと思ったのでこうなったが恐ろしいくらい分かりにくいw
    def sample_valid_output_size(self, input_sizes: Dict[int, int], output_dimensions: Dict[int, int], max_failures=100):
        """
        有効な出力サイズを探して１つ返します。
        Parameters
        ----------
        input_sizes: input nodeの番号がkey, 入力サイズがvalueのdict  
        output_dimensions: nodeの番号がkey, 出力の次元(1 or 4)がvalueのdict   
        max_failures: max_failures回失敗したら諦めてFalseを返します
        Returns
        ----------
        output_sizes: nodeの番号がkey, 出力のサイズがvalueのdict(次元が1のnodeについては本関数では決めず-1を返す) 
        """
        scc_idx_output_dimensions = {self.scc_idx[v]: output_dimensions[v] for v in self.g.nodes}
        assert len(input_sizes) == len(self.starts)
        find = False
        starts = [self.scc_idx[s] for s in self.starts]
        fail_count = 0
        while not find:
            g_labeled = nx.DiGraph()
            g_labeled.add_nodes_from(self.t_sorted)
            for v, s in input_sizes.items(): g_labeled.nodes[self.t_sorted[v]]['size'] = s

            for v in self.t_sorted:
                if v in starts: continue
                is_end = v == self.t_sorted[-1]
                # 出力が1次元のものはここでは決めない
                if scc_idx_output_dimensions[v] == 1:
                    g_labeled.nodes[v]['size'] = -1
                else:
                    s = list(self.g_compressed_inv.edges([v]))[0][1]  # どこでもいいのでvに入る頂点をpick up
                    valid_sizes = []  # 割り当て可能なsize
                    for _, sz in self.size_transision_graph.edges([g_labeled.nodes[s]['size']]):
                        validities = [
                            self.size_transision_graph.has_edge(g_labeled.nodes[u]['size'], sz) for _, u in self.g_compressed_inv.edges([v])
                        ]
                        is_valid_size = reduce(and_, validities)
                        if is_valid_size:
                            valid_sizes.append(sz)
                    if len(valid_sizes) == 0:
                        fail_count += 1
                        if fail_count >= max_failures: return False
                        break

                    g_labeled.nodes[v]['size'] = random.sample(valid_sizes, 1)[0]
                if is_end:
                    return self.__as_size_dict([g_labeled.nodes[v]['size'] for v in self.t_sorted])
