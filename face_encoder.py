"""Extract 128-dimensional face embeddings from local images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from face_detector import FaceLocation, detect_faces, face_size, largest_face

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_ENROLLMENT_FACE_SIZE = 80
MIN_ENROLLMENT_BLUR_SCORE = 50.0


def _backend():
    try:
        import face_recognition
    except ImportError as exc:
        raise RuntimeError(
            "face_recognition is not installed. Install requirements.txt first."
        ) from exc
    return face_recognition


def get_face_embedding(
    rgb_image: np.ndarray,
    location: FaceLocation | None = None,
    *,
    num_jitters: int = 1,
) -> np.ndarray | None:
    """Encode one face. If no box is supplied, the largest detected face is used."""
    if location is None:
        location = largest_face(detect_faces(rgb_image))
    if location is None:
        return None
    encodings = _backend().face_encodings(
        rgb_image, known_face_locations=[location], num_jitters=num_jitters
    )
    return np.asarray(encodings[0], dtype=np.float32) if encodings else None


def image_files(folder: Path) -> Iterable[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def get_embeddings_from_folder(folder: str | Path) -> tuple[list[np.ndarray], list[Path]]:
    """Validate and encode enrollment images, returning embeddings and used paths."""
    paths = list(image_files(Path(folder)))
    if not paths:
        return [], []
    backend = _backend()
    embeddings: list[np.ndarray] = []
    used_paths: list[Path] = []
    for path in paths:
        try:
            rgb_image = backend.load_image_file(str(path))
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: image could not be opened ({exc})")
            continue
        locations = detect_faces(rgb_image)
        if not locations:
            print(f"Skipping {path}: no face found")
            continue
        if len(locations) > 1:
            print(f"Skipping {path}: multiple faces found ({len(locations)}); use one face per image")
            continue
        width, height = face_size(locations[0])
        if min(width, height) < MIN_ENROLLMENT_FACE_SIZE:
            print(
                f"Warning for {path}: face is very small ({width}x{height}); "
                "a closer image is recommended"
            )
        top, right, bottom, left = locations[0]
        face_crop = rgb_image[top:bottom, left:right]
        try:
            import cv2

            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur < MIN_ENROLLMENT_BLUR_SCORE:
                print(
                    f"Skipping {path}: face is too blurry "
                    f"({blur:.1f} < {MIN_ENROLLMENT_BLUR_SCORE:.1f})"
                )
                continue
        except ImportError:
            pass  # Detection still provides the essential validation without OpenCV.
        embedding = get_face_embedding(rgb_image, locations[0])
        if embedding is not None:
            embeddings.append(embedding)
            used_paths.append(path)
        else:
            print(f"Skipping {path}: face encoding failed")
    return embeddings, used_paths
