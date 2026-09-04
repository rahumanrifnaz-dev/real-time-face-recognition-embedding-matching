"""Measure actual face detection/encoding/matching speed on stored images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
_IMPORT_ERROR = None
try:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:  # Keep --help available before optional packages are installed.
    _IMPORT_ERROR = exc

from face_database import DEFAULT_DATABASE_PATH, DEFAULT_THRESHOLD, load_database, match_embedding
from face_detector import detect_faces
from face_encoder import IMAGE_SUFFIXES, get_face_embedding
from person_retrieval import IdentityNotEnrolledError, retrieve_person
from utils import bgr_to_rgb, ensure_project_directories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference runtime on stored, consented images.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/performance_metrics.csv"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/figures/latency_results.png"))
    parser.add_argument(
        "--group-manifest", type=Path,
        help="optional CSV with image,target rows for group-search timing",
    )
    parser.add_argument(
        "--group-output", type=Path,
        default=Path("outputs/metrics/group_search_benchmark.csv"),
    )
    return parser.parse_args()


def benchmark_group_search(args: argparse.Namespace, database) -> None:
    """Measure detection, encoding, and requested-identity matching separately."""
    if not args.group_manifest.is_file():
        raise SystemExit(f"Group manifest not found: {args.group_manifest}")
    manifest = pd.read_csv(args.group_manifest)
    required_columns = {"image", "target"}
    if not required_columns.issubset(manifest.columns):
        raise SystemExit("Group manifest must contain image and target columns.")

    rows = []
    for item in manifest.to_dict("records"):
        image_path = Path(str(item["image"]))
        if not image_path.is_absolute():
            image_path = args.group_manifest.parent / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable group image: {image_path}")
            continue

        total_started = time.perf_counter()
        rgb = bgr_to_rgb(image)
        detection_started = time.perf_counter()
        locations = detect_faces(rgb)
        detection_ms = (time.perf_counter() - detection_started) * 1000

        encoding_started = time.perf_counter()
        embeddings = [get_face_embedding(rgb, box) for box in locations]
        encoding_ms = (time.perf_counter() - encoding_started) * 1000

        matching_started = time.perf_counter()
        try:
            result = retrieve_person(
                embeddings, locations, str(item["target"]), database, args.threshold
            )
        except IdentityNotEnrolledError as exc:
            raise SystemExit(f"Group manifest error: {exc}") from exc
        matching_ms = (time.perf_counter() - matching_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000
        rows.append({
            "image": str(item["image"]),
            "target": result["person"],
            "num_faces": len(locations),
            "detection_ms": detection_ms,
            "encoding_ms": encoding_ms,
            "matching_ms": matching_ms,
            "total_ms": total_ms,
            "found": result["found"],
            "best_distance": result["distance"] if np.isfinite(result["distance"]) else np.nan,
        })

    if not rows:
        raise SystemExit("No readable group images were processed.")
    args.group_output.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.group_output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {args.group_output}")


def main() -> None:
    args = parse_args()
    if _IMPORT_ERROR is not None:
        raise SystemExit(f"Missing dependency ({_IMPORT_ERROR}). Install requirements.txt first.")
    ensure_project_directories()
    database = load_database(args.database)
    if args.group_manifest is not None:
        benchmark_group_search(args, database)
        return
    paths = sorted(path for path in args.test_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit("DATA COLLECTION REQUIRED: no stored images found in the selected directory.")
    timings = []
    detected_faces = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        started = time.perf_counter()
        rgb = bgr_to_rgb(image)
        locations = detect_faces(rgb)
        for box in locations:
            embedding = get_face_embedding(rgb, box)
            if embedding is not None:
                match_embedding(embedding, database, args.threshold)
        timings.append((time.perf_counter() - started) * 1000)
        detected_faces += len(locations)
    if not timings:
        raise SystemExit("No readable images were processed.")
    average = float(np.mean(timings))
    summary = pd.DataFrame([{
        "images_processed": len(timings),
        "faces_detected": detected_faces,
        "average_latency_ms": average,
        "median_latency_ms": float(np.median(timings)),
        "minimum_latency_ms": float(np.min(timings)),
        "maximum_latency_ms": float(np.max(timings)),
        "approximate_fps": 1000.0 / average if average > 0 else np.nan,
    }])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plt.hist(timings, bins=min(15, max(3, len(timings))), color="#4daf4a", edgecolor="white")
    plt.axvline(average, color="#d62728", linestyle="--", label=f"mean {average:.1f} ms")
    plt.xlabel("Detection + encoding + matching latency (ms)")
    plt.ylabel("Image count")
    plt.title(f"Runtime benchmark on permitted stored images (n={len(timings)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.figure, dpi=160)
    plt.close()
    print(summary.to_string(index=False))
    print(f"Saved {args.output}")
    print(f"Saved {args.figure}")


if __name__ == "__main__":
    main()
