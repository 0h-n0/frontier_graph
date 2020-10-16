import networkx as nx

import copy
from typing import List


class FrameGenerator():
    def __init__(self, g: nx.DiGraph, starts: List[int], ends: List[int]):
        self.g = g
        self.starts = starts
        self.ends = ends
        self.__valid_graphs: List[nx.DiGraph] = []

    def __dfs(self, v: int, cur_graph: nx.DiGraph, cur_graph_inv: nx.DiGraph):
        if len(self.__valid_graphs) > 100: return  # TODO specify from outside
        # endsに含まれていて入次数が0。endsが一つならここに来ることはない。
        if v in self.ends and len(cur_graph_inv.edges([v])) == 0:
            return
        # 最後の頂点
        if v == max(self.ends):
            # 使われていない頂点は除いたgraphを作成する
            g_generated = nx.DiGraph()
            g_generated.add_edges_from(cur_graph.edges)
            self.__valid_graphs.append(g_generated)
            return

        # 自分への入次数が0かつstartsに含まれない
        if len(cur_graph_inv.edges([v])) == 0 and (not v in self.starts):
            self.__dfs(v + 1, cur_graph, cur_graph_inv)
            return

        # 自分への入次数が1以上かstart
        edges = self.g.edges([v])
        for edge_selection in reversed(list(range(1, 1 << len(edges)))):
            for i, (_, to) in enumerate(edges):
                if (1 << i) & edge_selection:
                    cur_graph.add_edge(v, to)
                    cur_graph_inv.add_edge(to, v)
            self.__dfs(v + 1, cur_graph, cur_graph_inv)
            for i, (_, to) in enumerate(edges):
                if (1 << i) & edge_selection:
                    cur_graph.remove_edge(v, to)
                    cur_graph_inv.remove_edge(to, v)

    # 頂点番号の昇順がトポロジカル順序になっていること(自分より番号の小さい頂点への辺がないこと)
    def list_valid_graph(self):
        start = min(self.starts)
        self.__dfs(start, nx.DiGraph(), nx.DiGraph())
        return self.__valid_graphs
