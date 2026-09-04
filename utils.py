"""Small shared image and filesystem helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def ensure_project_directories() -> None:
    for folder in (
        "data/enrolled",
        "data/test",
        "embeddings",
        "outputs/figures",
        "outputs/metrics",
        "outputs/examples",
        "outputs/group_search",
        "examples/group",
    ):
        Path(folder).mkdir(parents=True, exist_ok=True)


def safe_identity_name(name: str) -> str:
    """Allow readable folder names without path traversal or shell-sensitive punctuation."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip().replace(" ", "_")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("name must contain letters or numbers")
    if cleaned.casefold() == "unknown":
        raise ValueError("Unknown is reserved for people who are not enrolled")
    return cleaned


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def blur_score(image: np.ndarray) -> float:
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clamp_box(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    top, right, bottom, left = box
    height, width = shape[:2]
    return max(0, top), min(width, right), min(height, bottom), max(0, left)
