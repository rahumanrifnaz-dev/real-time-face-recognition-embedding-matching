"""Capture a small, consented set of enrollment photographs from a webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

from face_detector import detect_faces, face_size
from utils import bgr_to_rgb, blur_score, clamp_box, ensure_project_directories, safe_identity_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll one consenting person using the local webcam.")
    parser.add_argument("--name", required=True, help="identity label, for example Alice")
    parser.add_argument("--samples", type=int, default=10, help="number of images to capture (default: 10)")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--min-face-size", type=int, default=100, help="minimum face width and height")
    parser.add_argument("--blur-threshold", type=float, default=80.0, help="minimum Laplacian variance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed. Install requirements.txt first.") from exc
    if not 1 <= args.samples <= 100:
        raise SystemExit("--samples must be between 1 and 100")
    try:
        identity = safe_identity_name(args.name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ensure_project_directories()
    output_dir = Path("data/enrolled") / identity
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    captured = 0
    message = "SPACE capture | Q quit"
    print("Capture frontal and slight left/right views. Only the selected still frames are saved.")
    try:
        while captured < args.samples:
            ok, frame = camera.read()
            if not ok:
                message = "Camera frame unavailable"
                continue
            clean_frame = frame.copy()
            locations = detect_faces(bgr_to_rgb(clean_frame))
            color = (0, 200, 0) if len(locations) == 1 else (0, 0, 255)
            for box in locations:
                top, right, bottom, left = clamp_box(box, frame.shape)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.putText(frame, f"Captured {captured}/{args.samples}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, message, (15, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.imshow("Consent-based face enrollment", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != ord(" "):
                continue
            if len(locations) != 1:
                message = f"Rejected: expected 1 face, found {len(locations)}"
                continue
            width, height = face_size(locations[0])
            if min(width, height) < args.min_face_size:
                message = f"Rejected: face too small ({width}x{height})"
                continue
            score = blur_score(clean_frame)
            if score < args.blur_threshold:
                message = f"Rejected: image blurry ({score:.0f} < {args.blur_threshold:.0f})"
                continue
            captured += 1
            path = output_dir / f"frame_{captured:02d}.jpg"
            if not cv2.imwrite(str(path), clean_frame):
                captured -= 1
                message = "Could not save image"
                continue
            message = f"Saved {path.name}; vary pose slightly"
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Captured {captured}/{args.samples} images in {output_dir}")
    if captured:
        print("Next: python face_database.py --build")


if __name__ == "__main__":
    main()
