import numpy as np
import pytest

from person_retrieval import (
    IdentityNotEnrolledError,
    parse_person_queries,
    parse_person_query,
    retrieve_people,
    retrieve_person,
)


BOXES = [(0, 10, 10, 0), (20, 30, 30, 20), (40, 50, 50, 40)]


def database():
    return {
        "Person_A": [np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.0, 0.0])],
        "Person_B": [np.array([3.0, 3.0, 3.0])],
    }


def test_closest_face_is_returned_with_original_index_and_box():
    faces = [np.array([2.0, 2.0, 2.0]), np.array([0.1, 0.1, 0.1]), np.ones(3)]
    result = retrieve_person(faces, BOXES, "Person_A", database(), threshold=0.6)
    assert result["found"] is True
    assert result["face_index"] == 1
    assert result["bbox"] == BOXES[1]
    assert result["distance"] == pytest.approx(np.sqrt(0.03))


def test_distance_equal_to_threshold_is_accepted():
    result = retrieve_person(
        [np.array([0.3, 0.4, 0.0])], [BOXES[0]], "Person_A",
        {"Person_A": [np.zeros(3)]}, threshold=0.5,
    )
    assert result["found"] is True


def test_threshold_rejection_has_no_localization():
    result = retrieve_person([np.ones(3)], [BOXES[0]], "Person_A", database(), threshold=0.5)
    assert result["found"] is False
    assert result["distance"] > result["threshold"]
    assert result["face_index"] is None
    assert result["bbox"] is None


def test_failed_encoding_keeps_detection_indices_aligned():
    result = retrieve_person(
        [None, np.array([0.05, 0.0, 0.0]), None], BOXES,
        "Person_A", database(), threshold=0.6,
    )
    assert result["face_index"] == 1
    assert result["bbox"] == BOXES[1]


def test_missing_identity_is_reported():
    with pytest.raises(IdentityNotEnrolledError, match="not enrolled"):
        retrieve_person([], [], "Person_X", database())


def test_empty_detection_list_returns_not_found():
    result = retrieve_person([], [], "Person_A", database())
    assert result["found"] is False
    assert result["distance"] == float("inf")
    assert result["face_index"] is None
    assert result["bbox"] is None


def test_embedding_and_box_counts_must_match():
    with pytest.raises(ValueError, match="equal lengths"):
        retrieve_person([np.zeros(3)], [], "Person_A", database())


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Person_A", "Person_A"),
        ("Find Person_A", "Person_A"),
        ("Where is person_a?", "Person_A"),
        ("Please highlight Person A", "Person A"),
    ],
)
def test_text_query_maps_only_to_enrolled_labels(query, expected):
    assert parse_person_query(query, ["Person_A", "Person A", "Person_B"]) == expected


def test_text_query_rejects_non_enrolled_name():
    with pytest.raises(IdentityNotEnrolledError, match="does not contain"):
        parse_person_query("Find Celebrity_X", ["Person_A", "Person_B"])


def test_text_query_returns_multiple_enrolled_labels_in_request_order():
    people = parse_person_queries(
        "Highlight Person_B, Person_A and Person_B",
        ["Person_A", "Person_B", "Person_C"],
    )
    assert people == ["Person_B", "Person_A"]


def test_text_query_prefers_longest_overlapping_label():
    people = parse_person_queries("Find Ann Marie and Bob", ["Ann", "Ann Marie", "Bob"])
    assert people == ["Ann Marie", "Bob"]


def test_multiple_people_are_assigned_to_different_faces():
    faces = [np.array([0.04, 0.0]), np.array([0.25, 0.0])]
    boxes = [BOXES[0], BOXES[1]]
    references = {
        "Person_A": [np.array([0.0, 0.0])],
        "Person_B": [np.array([0.1, 0.0])],
    }

    results = retrieve_people(
        faces, boxes, ["Person_A", "Person_B"], references, threshold=0.3
    )

    assert [result["found"] for result in results] == [True, True]
    assert [result["face_index"] for result in results] == [0, 1]
    assert [result["bbox"] for result in results] == boxes


def test_one_face_cannot_be_assigned_to_two_people():
    results = retrieve_people(
        [np.array([0.01, 0.0])],
        [BOXES[0]],
        ["Person_A", "Person_B"],
        {
            "Person_A": [np.array([0.0, 0.0])],
            "Person_B": [np.array([0.05, 0.0])],
        },
        threshold=0.1,
    )

    assert results[0]["found"] is True
    assert results[0]["face_index"] == 0
    assert results[1]["found"] is False
    assert results[1]["face_index"] is None


def test_matching_reassigns_a_face_to_find_more_requested_people():
    results = retrieve_people(
        [np.array([0.004]), np.array([-0.04])],
        [BOXES[0], BOXES[1]],
        ["Person_A", "Person_B"],
        {
            "Person_A": [np.array([0.0])],
            "Person_B": [np.array([0.01])],
        },
        threshold=0.045,
    )

    assert [result["found"] for result in results] == [True, True]
    assert results[0]["face_index"] == 1
    assert results[1]["face_index"] == 0
