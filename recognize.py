"""Run local, open-set face recognition from a saved image or webcam."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from face_database import DEFAULT_DATABASE_PATH, DEFAULT_THRESHOLD, load_database, match_embedding
from face_detector import detect_faces
from face_encoder import get_face_embedding
from utils import bgr_to_rgb, clamp_box

MATCH_THRESHOLD = DEFAULT_THRESHOLD  # Maximum accepted Euclidean distance.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize explicitly enrolled local identities from an image or webcam."
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--threshold", type=float, default=MATCH_THRESHOLD,
        help="maximum Euclidean distance for a known match (default: 0.60)",
    )
    parser.add_argument("--image", type=Path, help="recognize faces in one permitted local image")
    parser.add_argument("--debug", action="store_true", help="print matching and timing details")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--scale", type=float,
        help="processing scale; defaults to 1.0 for --image and 0.5 for webcam",
    )
    parser.add_argument("--process-every", type=int, default=2, help="recognize every Nth camera frame")
    return parser.parse_args()


def recognize_frame(frame, database, threshold: float, scale: float):
    """Recognize all faces in one BGR frame; return results and detection count."""
    import cv2

    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb = bgr_to_rgb(small)
    locations = detect_faces(rgb)
    results = []
    for box in locations:
        embedding = get_face_embedding(rgb, box)
        if embedding is None:
            continue
        label, distance, closest = match_embedding(embedding, database, threshold)
        scaled_box = tuple(int(round(value / scale)) for value in box)
        results.append((scaled_box, label, distance, closest))
    return results, len(locations)


def print_debug(results, face_count: int, threshold: float, latency_ms: float) -> None:
    """Print compact diagnostics for one processed image/frame."""
    print(f"faces detected: {face_count}")
    if not results:
        print("decision: no encodable face")
    for index, (_, label, distance, closest) in enumerate(results, start=1):
        prefix = f"face {index}: " if len(results) > 1 else ""
        print(f"{prefix}best match: {closest or 'none'}")
        print(f"{prefix}best distance: {distance:.3f}")
        print(f"{prefix}threshold: {threshold:.3f}")
        print(f"{prefix}decision: {label}")
    print(f"processing time: {latency_ms:.1f} ms")


def recognize_image(path: Path, database, threshold: float, scale: float, debug: bool) -> None:
    """Recognize every face in one saved local image and print real decisions."""
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not open image: {path}")
    started = time.perf_counter()
    results, face_count = recognize_frame(frame, database, threshold, scale)
    latency_ms = (time.perf_counter() - started) * 1000
    if not results:
        print(f"No encodable face found in {path} (detected: {face_count}).")
    for index, (_, label, distance, _) in enumerate(results, start=1):
        if len(results) > 1:
            print(f"\nFace {index}")
        if label == "Unknown":
            print("Prediction: Unknown")
            print(f"Best distance: {distance:.3f}")
            print("Decision: Unknown")
        else:
            print(f"Prediction: {label}")
            print(f"Distance: {distance:.3f}")
            print("Decision: Known")
    if debug:
        print("\nDebug")
        print_debug(results, face_count, threshold, latency_ms)


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed. Install requirements.txt first.") from exc
    processing_scale = args.scale if args.scale is not None else (1.0 if args.image else 0.5)
    if not 0 < processing_scale <= 1:
        raise SystemExit("--scale must be greater than 0 and no more than 1")
    if args.process_every < 1:
        raise SystemExit("--process-every must be at least 1")
    if args.threshold <= 0:
        raise SystemExit("--threshold must be greater than 0")
    database = load_database(args.database)
    if not database:
        raise SystemExit("The embedding database is empty. Add permitted images and rebuild it.")

    if args.image is not None:
        recognize_image(args.image, database, args.threshold, processing_scale, args.debug)
        return

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    frame_number = 0
    latest_results = []
    latest_face_count = 0
    latencies: deque[float] = deque(maxlen=60)
    frame_times: deque[float] = deque(maxlen=60)
    previous_time = time.perf_counter()
    last_debug_time = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            now = time.perf_counter()
            frame_times.append(now - previous_time)
            previous_time = now
            if frame_number % args.process_every == 0:
                started = time.perf_counter()
                latest_results, latest_face_count = recognize_frame(
                    frame, database, args.threshold, processing_scale
                )
                processing_ms = (time.perf_counter() - started) * 1000
                latencies.append(processing_ms)
                if args.debug and now - last_debug_time >= 1.0:
                    print_debug(latest_results, latest_face_count, args.threshold, processing_ms)
                    print()
                    last_debug_time = now
            frame_number += 1

            for box, label, distance, closest in latest_results:
                top, right, bottom, left = clamp_box(box, frame.shape)
                color = (0, 180, 0) if label != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(
                    frame, label, (left, max(20, top - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )
                detail = f"distance: {distance:.3f}"
                if label == "Unknown" and closest:
                    detail += f" (closest: {closest})"
                cv2.putText(
                    frame, detail, (left, max(40, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
                )

            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            latency = sum(latencies) / len(latencies) if latencies else 0.0
            cv2.putText(
                frame, f"FPS: {fps:.1f} | processing: {latency:.1f} ms",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            cv2.imshow("Local face recognition - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
