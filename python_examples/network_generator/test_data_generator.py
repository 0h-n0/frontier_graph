import networkx as nx
import random

from typing import List


def generate_random_graph(size: int, starts: List[int], ends: List[int], p: float) -> nx.DiGraph:
    """
    確率pくらいで辺を張ることでグラフを作成します。
    """
    g = nx.DiGraph()
    g_inv = nx.DiGraph()
    for v in range(1, size):
        if not v in starts and len(g_inv.edges([v])) == 0:
            s = random.randint(1, v - 1)
            g.add_edge(s, v)
            g_inv.add_edge(v, s)
        added = []
        for to in range(v + 1, size + 1):
            if random.random() < p:
                g.add_edge(v, to)
                g_inv.add_edge(to, v)
                added.append(to)
        if len(added) == 0:
            to = random.randint(v + 1, size)
            g.add_edge(v, to)
            g_inv.add_edge(to, v)
    return g


def generate_graph(n_inputs: int, n_outputs: int, max_width: int, max_width_count: int):
    """ 実際に入力されるグラフを作成します。
    Parameters
    ----------
    n_inputs : int
        inputのnode数
    n_outputs : int
        outputのnode数
    max_width_count : int
        幅が最大(n_inputs+1)になる層の数
    """
    assert max_width > n_inputs
    assert max_width > n_outputs
    l = [n_inputs] + list(range(n_inputs, max_width - 1)) + [max_width - 1, max_width] * max_width_count +\
        list(reversed(range(n_outputs, max_width))) + [n_outputs]
    g = nx.DiGraph()
    cumsum = 0
    for n_cur_layer, n_next_layer in zip(l[0:-1], l[1:]):
        if n_cur_layer < n_next_layer:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_cur_layer)])
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer + 1) for j in range(n_cur_layer)])
        elif n_cur_layer == n_next_layer:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_cur_layer)])
        else:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_next_layer)])
            g.add_edges_from([(cumsum + j + 1, cumsum + j + n_cur_layer) for j in range(n_next_layer)])
        cumsum += n_cur_layer

    starts = list(range(0, n_inputs))
    ends = list(range(cumsum, cumsum + n_outputs))
    return g, starts, ends
