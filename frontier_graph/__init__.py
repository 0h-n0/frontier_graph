import typing

import networkx as nx

from . import frontier


class NetworkxInterface:
    def __init__(self, digraph: nx.DiGraph):
        self.digraph = digraph
        self.n_nodes = len(digraph)

    def sample(self, starts: typing.List[int], ends: typing.List[int], n_sample) -> typing.List[typing.List[int]]:
        assert isinstance(self.n_nodes, int), "TypeError: {type(self.n_nodes}"
        assert isinstance(starts, list), "TypeError: {type(starts}"
        assert isinstance(ends, list), "TypeError: {type(ends}"
        assert isinstance(n_sample, int), "TypeError: {type(n_sample}"
        out = frontier.calc_frontier_combination(
            self.n_nodes,
            list(self.digraph.edges()),
            starts,
            ends,
            n_sample)
        return out
