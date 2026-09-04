"""Find one enrolled local identity among embeddings from a group image."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import re
from typing import Any

import numpy as np

from face_database import DEFAULT_THRESHOLD, match_embedding
from face_detector import FaceLocation


class IdentityNotEnrolledError(ValueError):
    """Raised when a requested label is absent from the local database."""


def parse_person_queries(query: str, enrolled_identities: Sequence[str]) -> list[str]:
    """Return every enrolled local label mentioned in a text request.

    This parser processes text only. It does not inspect the image to determine a
    name and does not contact an LLM or any external identity service.
    """
    cleaned_query = query.strip().casefold()
    if not cleaned_query:
        raise IdentityNotEnrolledError("enter an enrolled identity to search for")

    exact_matches = [name for name in enrolled_identities if name.casefold() == cleaned_query]
    if len(exact_matches) == 1:
        return exact_matches

    mentions: list[tuple[int, int, str]] = []
    for name in enrolled_identities:
        pattern = rf"(?<!\w){re.escape(name.casefold())}(?!\w)"
        for match in re.finditer(pattern, cleaned_query):
            mentions.append((match.start(), match.end(), name))

    # Prefer the longest label when enrolled names overlap, such as "Ann" and
    # "Ann Marie", while retaining the request's left-to-right order.
    selected: list[tuple[int, int, str]] = []
    for start, end, name in sorted(mentions, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(start < other_end and end > other_start for other_start, other_end, _ in selected):
            continue
        selected.append((start, end, name))

    people: list[str] = []
    for _, _, name in sorted(selected):
        if name not in people:
            people.append(name)
    if people:
        return people
    raise IdentityNotEnrolledError("the request does not contain an enrolled identity")


def parse_person_query(query: str, enrolled_identities: Sequence[str]) -> str:
    """Backward-compatible helper returning the first requested local label."""
    return parse_person_queries(query, enrolled_identities)[0]


def resolve_identity(
    person: str, database: dict[str, list[np.ndarray]]
) -> str:
    """Return the stored label, allowing an unambiguous case-insensitive lookup."""
    if person in database and database[person]:
        return person
    matches = [
        name
        for name, references in database.items()
        if references and name.casefold() == person.casefold()
    ]
    if len(matches) == 1:
        return matches[0]
    raise IdentityNotEnrolledError(f"identity '{person}' is not enrolled")


def retrieve_people(
    detected_embeddings: Sequence[np.ndarray | None],
    bounding_boxes: Sequence[FaceLocation],
    people: Sequence[str],
    database: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Retrieve multiple identities while assigning each detected face once.

    A small assignment search first maximizes how many requested identities are
    found, then minimizes their combined Euclidean distance. This prevents two
    labels from highlighting the same detected face.
    """
    if len(detected_embeddings) != len(bounding_boxes):
        raise ValueError("detected embeddings and bounding boxes must have equal lengths")
    if threshold <= 0:
        raise ValueError("threshold must be greater than zero")
    if not people:
        raise ValueError("at least one person must be requested")

    stored_people: list[str] = []
    for person in people:
        stored_person = resolve_identity(person, database)
        if stored_person not in stored_people:
            stored_people.append(stored_person)

    results = []
    candidates: list[list[tuple[float, int]]] = []
    for person_index, stored_person in enumerate(stored_people):
        target_database = {stored_person: database[stored_person]}
        best_distance = float("inf")
        person_candidates: list[tuple[float, int]] = []
        for face_index, embedding in enumerate(detected_embeddings):
            if embedding is None:
                continue
            _, distance, _ = match_embedding(embedding, target_database, threshold)
            best_distance = min(best_distance, distance)
            if distance <= threshold:
                person_candidates.append((distance, face_index))
        candidates.append(sorted(person_candidates))
        results.append({
            "found": False,
            "person": stored_person,
            "distance": best_distance,
            "threshold": threshold,
            "face_index": None,
            "bbox": None,
        })

    @lru_cache(maxsize=None)
    def best_assignment(
        person_index: int, used_faces: int
    ) -> tuple[int, float, tuple[tuple[float, int] | None, ...]]:
        if person_index == len(stored_people):
            return 0, 0.0, ()

        found_count, total_distance, later_assignments = best_assignment(
            person_index + 1, used_faces
        )
        best = (found_count, total_distance, (None,) + later_assignments)

        for distance, face_index in candidates[person_index]:
            face_bit = 1 << face_index
            if used_faces & face_bit:
                continue
            later_count, later_distance, later_assignments = best_assignment(
                person_index + 1, used_faces | face_bit
            )
            option = (
                later_count + 1,
                later_distance + distance,
                ((distance, face_index),) + later_assignments,
            )
            if option[0] > best[0] or (option[0] == best[0] and option[1] < best[1]):
                best = option
        return best

    _, _, assignments = best_assignment(0, 0)
    for person_index, assignment in enumerate(assignments):
        if assignment is None:
            continue
        distance, face_index = assignment
        results[person_index].update({
            "found": True,
            "distance": distance,
            "face_index": face_index,
            "bbox": bounding_boxes[face_index],
        })

    return results


def retrieve_person(
    detected_embeddings: Sequence[np.ndarray | None],
    bounding_boxes: Sequence[FaceLocation],
    person: str,
    database: dict[str, list[np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Return the closest valid occurrence of ``person`` in a group.

    Embeddings and boxes must stay aligned. ``None`` may be used when detection
    succeeded but encoding failed, so the returned index still refers to the
    original detected face. Matching delegates to the project's existing
    nearest-reference matcher and therefore keeps its distance semantics.
    """
    return retrieve_people(
        detected_embeddings,
        bounding_boxes,
        [person],
        database,
        threshold,
    )[0]
