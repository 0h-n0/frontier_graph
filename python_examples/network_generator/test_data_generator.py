import networkx as nx
import random

from typing import List
import matplotlib.pyplot as plt


def make_graph():
    g = nx.DiGraph()
    starts = [1, 2]
    ends = [9, ]
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(4, 5)
    g.add_edge(4, 7)
    g.add_edge(5, 6)
    g.add_edge(5, 7)
    g.add_edge(5, 8)
    g.add_edge(6, 8)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    return g, starts, ends


def generate_random_graph(size: int, p: float) -> nx.DiGraph:
    g = nx.DiGraph()
    for v in range(1, size):
        added = []
        for to in range(v + 1, size + 1):
            if random.random() < p:
                g.add_edge(v, to)
                added.append(to)
        if len(added) == 0:
            to = random.randint(v + 1, size)
            g.add_edge(v, to)
    return g


def generate_graph(size: int, starts: List[int], ends: List[int], p: float) -> nx.DiGraph:
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


if __name__ == "__main__":
    g = generate_graph(15, [1, 2], [15], 0.2)
    labels = {v: v for v in range(1, 16)}
    nx.draw(g, labels=labels)

    # nx.draw(g, labels=labels, pos=nx.spectral_layout(g))
    plt.show()
