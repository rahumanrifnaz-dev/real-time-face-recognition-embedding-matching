import numpy as np
import pytest

from face_database import load_database, match_embedding, save_database


def sample_database():
    return {
        "Alice": [np.array([1.0, 0.0]), np.array([0.9, 0.1])],
        "Bob": [np.array([0.0, 1.0])],
    }


def test_close_embedding_is_recognized():
    label, distance, closest = match_embedding(np.array([0.95, 0.05]), sample_database(), 0.2)
    assert label == "Alice"
    assert closest == "Alice"
    assert distance < 0.2


def test_far_embedding_is_unknown_but_reports_closest_identity():
    label, distance, closest = match_embedding(np.array([-1.0, 0.0]), sample_database(), 0.3)
    assert label == "Unknown"
    assert closest in {"Alice", "Bob"}
    assert distance > 0.3


def test_empty_database_returns_unknown():
    label, distance, closest = match_embedding(np.array([1.0, 0.0]), {}, 0.6)
    assert label == "Unknown"
    assert distance == float("inf")
    assert closest is None


def test_database_round_trip(tmp_path):
    path = tmp_path / "faces.npz"
    save_database(sample_database(), path)
    loaded = load_database(path)
    assert set(loaded) == {"Alice", "Bob"}
    assert len(loaded["Alice"]) == 2
    np.testing.assert_allclose(loaded["Bob"][0], np.array([0.0, 1.0]))


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        match_embedding(np.array([1.0, 0.0]), sample_database(), 0)


def test_distance_equal_to_threshold_is_accepted():
    database = {"Person_A": [np.array([0.0, 0.0])]}
    label, distance, closest = match_embedding(np.array([0.3, 0.4]), database, 0.5)
    assert distance == pytest.approx(0.5)
    assert label == "Person_A"
    assert closest == "Person_A"
