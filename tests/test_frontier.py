import frontier_graph.frontier as f


def test_instance():
    frontier = f.FrontierInterface(1)
    print(frontier.method())
    out = f.calc_frontier_combination(
        4,
        [(1, 2), (1, 3), (2, 4), (3, 4)],
        [1,],
        [4,],
        2)
    assert([[2, 4], [1, 3]], out)
