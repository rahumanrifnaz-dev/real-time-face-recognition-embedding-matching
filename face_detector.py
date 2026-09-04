"""Face detection helpers built on face_recognition's local HOG detector."""

from __future__ import annotations

from typing import Sequence

import numpy as np

FaceLocation = tuple[int, int, int, int]  # top, right, bottom, left


def _backend():
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "face_recognition is not installed. Install requirements.txt first."
        ) from exc
    return face_recognition


def detect_faces(rgb_image: np.ndarray, model: str = "hog") -> list[FaceLocation]:
    """Return (top, right, bottom, left) boxes for every detected face."""
    if rgb_image is None or rgb_image.size == 0:
        return []
    locations: Sequence[FaceLocation] = _backend().face_locations(rgb_image, model=model)
    return [tuple(map(int, location)) for location in locations]


def largest_face(locations: Sequence[FaceLocation]) -> FaceLocation | None:
    """Choose the largest box, useful for single-person enrollment/evaluation."""
    if not locations:
        return None
    return max(locations, key=lambda box: max(0, box[2] - box[0]) * max(0, box[1] - box[3]))


def face_size(location: FaceLocation) -> tuple[int, int]:
    top, right, bottom, left = location
    return max(0, right - left), max(0, bottom - top)
