import numpy as np
import pytest

from face_database import euclidean_distance


def test_euclidean_distance_for_same_close_and_far_vectors():
    same = np.array([1.0, 0.0])
    close = np.array([0.95, 0.05])
    far = np.array([0.0, 1.0])

    assert euclidean_distance(same, same) == pytest.approx(0.0)
    assert euclidean_distance(same, close) < euclidean_distance(same, far)
    assert euclidean_distance(same, far) == pytest.approx(np.sqrt(2.0))


def test_distance_rejects_different_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        euclidean_distance(np.zeros(2), np.zeros(3))
