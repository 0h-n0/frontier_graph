from unittest import TestCase

import networkx as nx

import frontier_graph.frontier as f
from frontier_graph import NetworkxInterface


def test_instance():
    out = f.calc_frontier_combination(
        4,
        [(1, 2), (1, 3), (2, 4), (3, 4)],
        [1,],
        [4,],
        2)
    assert([[2, 4], [1, 3]], out)


class TestNetworkxInterface(TestCase):
    def test_sample(self):
        g = nx.DiGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(2, 4)
        g.add_edge(3, 4)
        ns = NetworkxInterface(g)
        graphs = ns.sample([1,], [4,], 3)
        assert([[2, 4], [1, 3]], graphs)
