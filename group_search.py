"""Retrieve and localize an enrolled identity in a permitted group image."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from face_database import DEFAULT_DATABASE_PATH, DEFAULT_THRESHOLD, load_database
from face_detector import FaceLocation, detect_faces
from face_encoder import get_face_embedding
from person_retrieval import IdentityNotEnrolledError, retrieve_person
from utils import bgr_to_rgb, clamp_box

DEFAULT_OUTPUT_DIR = Path("outputs/group_search")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find one enrolled local identity in a permitted group image."
    )
    parser.add_argument("--image", type=Path, required=True, help="path to the local group image")
    parser.add_argument("--person", required=True, help="label already present in the local database")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="maximum accepted Euclidean distance (default: 0.60)",
    )
    parser.add_argument("--output", type=Path, help="annotated output path")
    parser.add_argument("--show", action="store_true", help="display the annotated image")
    parser.add_argument(
        "--show-all-faces", action="store_true",
        help="draw subtle boxes around non-matching detected faces",
    )
    return parser.parse_args()


def available_path(path: Path) -> Path:
    """Avoid overwriting an earlier result by adding a numeric suffix."""
    if not path.exists():
        return path
    for number in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose an unused output name near: {path}")


def default_output_path(image_path: Path, person: str) -> Path:
    safe_person = re.sub(r"[^A-Za-z0-9_-]+", "_", person).strip("_").lower() or "person"
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}_{safe_person}.png"


def annotate_image(frame, locations: list[FaceLocation], result: dict, show_all_faces: bool):
    """Return a copy with the valid target emphasized and optional context boxes."""
    return annotate_people(frame, locations, [result], show_all_faces)


def annotate_people(
    frame, locations: list[FaceLocation], results: list[dict], show_all_faces: bool
):
    """Highlight every valid target with a distinct color."""
    import cv2

    annotated = frame.copy()
    matched_indices = {
        result["face_index"] for result in results if result["found"]
    }
    if show_all_faces:
        for index, box in enumerate(locations):
            if index in matched_indices:
                continue
            top, right, bottom, left = clamp_box(box, annotated.shape)
            cv2.rectangle(annotated, (left, top), (right, bottom), (150, 150, 150), 1)

    colors = [
        (0, 200, 0),
        (0, 165, 255),
        (255, 0, 180),
        (255, 200, 0),
        (0, 80, 255),
        (180, 80, 255),
    ]
    for result_index, result in enumerate(results):
        matched_index = result["face_index"] if result["found"] else None
        if matched_index is None:
            continue
        top, right, bottom, left = clamp_box(locations[matched_index], annotated.shape)
        color = colors[result_index % len(colors)]
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 3)
        label = f"{result['person']} | d={result['distance']:.3f}"
        text_y = top - 10 if top >= 30 else min(annotated.shape[0] - 8, bottom + 24)
        cv2.putText(
            annotated, label, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )
    return annotated


def main() -> int:
    args = parse_args()
    if args.threshold <= 0:
        print("Error: --threshold must be greater than zero.")
        return 2
    if not args.image.is_file():
        print(f"Error: image path does not exist: {args.image}")
        return 2

    try:
        import cv2
    except ImportError:
        print("Error: OpenCV is not installed. Install requirements.txt first.")
        return 2

    try:
        database = load_database(args.database)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}")
        return 2

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Error: image could not be opened: {args.image}")
        return 2

    started = time.perf_counter()
    rgb = bgr_to_rgb(frame)
    try:
        locations = detect_faces(rgb)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2
    if not locations:
        print("No faces were detected in the supplied image.")
        return 1

    try:
        embeddings = [get_face_embedding(rgb, box) for box in locations]
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2
    encodable_count = sum(embedding is not None for embedding in embeddings)
    try:
        result = retrieve_person(embeddings, locations, args.person, database, args.threshold)
    except IdentityNotEnrolledError:
        print(f"Error: identity '{args.person}' is not enrolled.")
        return 2
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    annotated = annotate_image(frame, locations, result, args.show_all_faces)
    requested_output = args.output or default_output_path(args.image, result["person"])
    output_path = available_path(requested_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        saved = cv2.imwrite(str(output_path), annotated)
    except cv2.error as exc:
        print(f"Error: could not save annotated image: {output_path} ({exc})")
        return 2
    if not saved:
        print(f"Error: could not save annotated image: {output_path}")
        return 2

    total_ms = (time.perf_counter() - started) * 1000
    best_distance = result["distance"]
    print(f"Target identity: {result['person']}")
    print(f"Faces detected: {len(locations)}")
    print(f"Faces encoded: {encodable_count}")
    print(f"Best distance: {best_distance:.3f}" if best_distance != float("inf") else "Best distance: unavailable")
    print(f"Threshold: {args.threshold:.3f}")
    print(f"Match: {'FOUND' if result['found'] else 'NOT FOUND'}")
    if result["found"]:
        print(f"Face index: {result['face_index']}")
    elif encodable_count == 0:
        print("No detected face could be encoded.")
    else:
        print(f"{result['person']} was not found within the current threshold.")
    print(f"Total processing time: {total_ms:.1f} ms")
    print(f"Saved: {output_path}")

    if args.show:
        cv2.imshow(f"Group search - {result['person']}", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if result["found"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
