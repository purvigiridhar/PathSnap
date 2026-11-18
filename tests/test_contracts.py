def test_imports():
    import pathsnap  # noqa: F401


def test_reference_alignment_preserves_route_order():
    import numpy as np

    from pathsnap.data.alignment import align_reference

    nodes = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])
    endpoints = np.array([[0, 1], [2, 3]])
    observations = np.array([[0, 0], [10, 0], [1, 0], [11, 0]])
    aligned = align_reference(observations, nodes, endpoints, np.array([0, 1]))
    assert aligned.tolist() == [0, 0, 0, 1]
