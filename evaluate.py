"""Evaluate recognition on consented, labeled local test images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_IMPORT_ERROR = None
try:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
except ImportError as exc:  # Keep --help available before optional packages are installed.
    _IMPORT_ERROR = exc

from face_database import DEFAULT_DATABASE_PATH, DEFAULT_THRESHOLD, UNKNOWN_LABEL, load_database, match_embedding
from face_detector import detect_faces, face_size, largest_face
from face_encoder import IMAGE_SUFFIXES, get_face_embedding
from utils import bgr_to_rgb, ensure_project_directories

LIGHTING = ["normal", "dim", "bright"]
DISTANCE = ["near", "medium", "far"]
POSE = ["frontal", "left", "right", "up", "down"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate labeled, consented face images.")
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--manifest", type=Path,
                        help="optional CSV with image, expected_identity, condition columns")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--results", type=Path, default=Path("outputs/metrics/evaluation_results.csv"))
    return parser.parse_args()


def collect_samples(test_dir: Path, manifest: Path | None) -> list[dict[str, str]]:
    if manifest:
        table = pd.read_csv(manifest).fillna("")
        required = {"image", "expected_identity", "condition"}
        if not required.issubset(table.columns):
            raise ValueError(f"Manifest must contain: {', '.join(sorted(required))}")
        samples = []
        for row in table.to_dict("records"):
            path = Path(str(row["image"]))
            if not path.is_absolute():
                path = test_dir / path
            samples.append({"image": str(path), "expected_identity": str(row["expected_identity"]),
                            "condition": str(row["condition"]) or "unspecified"})
        return samples
    samples = []
    for identity_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for path in sorted(identity_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append({"image": str(path), "expected_identity": identity_dir.name,
                                "condition": "unspecified"})
    return samples


def evaluate_image(sample: dict[str, str], database, threshold: float) -> dict:
    path = Path(sample["image"])
    started = time.perf_counter()
    frame = cv2.imread(str(path))
    if frame is None:
        return {**sample, "predicted_identity": "Unreadable", "closest_identity": "",
                "distance": np.nan, "threshold": threshold, "latency_ms": np.nan,
                "face_width": 0, "face_height": 0, "faces_detected": 0, "correct": False}
    rgb = bgr_to_rgb(frame)
    locations = detect_faces(rgb)
    box = largest_face(locations)
    if box is None:
        prediction, closest, distance, width, height = "No face", "", np.nan, 0, 0
    else:
        width, height = face_size(box)
        embedding = get_face_embedding(rgb, box)
        if embedding is None:
            prediction, closest, distance = "Encoding failed", "", np.nan
        else:
            prediction, distance, closest_name = match_embedding(embedding, database, threshold)
            closest = closest_name or ""
    latency_ms = (time.perf_counter() - started) * 1000
    expected = sample["expected_identity"]
    correct = prediction.casefold() == expected.casefold()
    return {**sample, "predicted_identity": prediction, "closest_identity": closest,
            "distance": distance, "threshold": threshold, "latency_ms": latency_ms,
            "face_width": width, "face_height": height, "faces_detected": len(locations),
            "correct": bool(correct)}


def rate(series: pd.Series) -> float | None:
    return float(series.mean()) if len(series) else None


def calculate_metrics(results: pd.DataFrame) -> dict:
    expected_unknown = results["expected_identity"].str.casefold() == UNKNOWN_LABEL.casefold()
    known = results[~expected_unknown]
    unknown = results[expected_unknown]
    valid_distances = results["distance"].dropna()
    valid_latencies = results["latency_ms"].dropna()
    correct_count = int(results["correct"].sum())
    return {
        "number_of_test_images": int(len(results)),
        "correct_predictions": correct_count,
        "incorrect_predictions": int(len(results) - correct_count),
        "accuracy": rate(results["correct"]),
        "mean_match_distance": float(valid_distances.mean()) if len(valid_distances) else None,
        "median_match_distance": float(valid_distances.median()) if len(valid_distances) else None,
        "sample_count": int(len(results)),
        "overall_accuracy": rate(results["correct"]),
        "known_person_recognition_rate": rate(known["correct"]),
        "unknown_person_rejection_rate": rate(
            unknown["predicted_identity"].str.casefold() == UNKNOWN_LABEL.casefold()
        ),
        "average_distance": float(valid_distances.mean()) if len(valid_distances) else None,
        "average_latency_ms": float(valid_latencies.mean()) if len(valid_latencies) else None,
    }


def save_bar_for_conditions(results: pd.DataFrame, conditions: list[str], filename: str, title: str) -> bool:
    subset = results[results["condition"].str.casefold().isin(conditions)].copy()
    if subset.empty:
        return False
    subset["condition"] = subset["condition"].str.casefold()
    summary = subset.groupby("condition")["correct"].agg(["mean", "count"]).reindex(conditions).dropna()
    ax = summary["mean"].plot.bar(color="#377eb8", ylim=(0, 1), rot=0)
    ax.set_ylabel("Recognition accuracy")
    ax.set_title(title)
    for index, (_, row) in enumerate(summary.iterrows()):
        ax.text(index, row["mean"] + 0.025, f"n={int(row['count'])}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(Path("outputs/figures") / filename, dpi=160)
    plt.close()
    return True


def generate_figures(results: pd.DataFrame) -> list[str]:
    generated: list[str] = []
    figure_dir = Path("outputs/figures")
    metrics_dir = Path("outputs/metrics")
    robustness_conditions = set(LIGHTING + DISTANCE + POSE)
    robustness = results[results["condition"].str.casefold().isin(robustness_conditions)].copy()
    if not robustness.empty:
        robustness.to_csv(metrics_dir / "robustness_results.csv", index=False)
    for conditions, filename, title in (
        (LIGHTING, "lighting_results.png", "Recognition by qualitative lighting"),
        (DISTANCE, "distance_results.png", "Recognition by distance category"),
        (POSE, "pose_results.png", "Recognition by moderate pose"),
    ):
        if save_bar_for_conditions(results, conditions, filename, title):
            generated.append(filename)

    latency = results["latency_ms"].dropna()
    if len(latency):
        plt.hist(latency, bins=min(15, max(3, len(latency))), color="#4daf4a", edgecolor="white")
        plt.xlabel("End-to-end image processing latency (ms)")
        plt.ylabel("Image count")
        plt.title(f"Latency measurements (n={len(latency)})")
        plt.tight_layout(); plt.savefig(figure_dir / "latency_results.png", dpi=160); plt.close()
        generated.append("latency_results.png")

    usable = results.dropna(subset=["distance"]).copy()
    expected_unknown = usable["expected_identity"].str.casefold().eq(UNKNOWN_LABEL.casefold())
    if len(usable) and (~expected_unknown).any():
        thresholds = np.linspace(0.35, 0.8, 19)
        threshold_rows = []
        for threshold in thresholds:
            accepted = usable["distance"] <= threshold
            closest_is_expected = (
                usable["closest_identity"].str.casefold()
                == usable["expected_identity"].str.casefold()
            )
            known_ok = accepted & closest_is_expected & (~expected_unknown)
            false_accept = accepted & expected_unknown
            false_reject = (~accepted) & (~expected_unknown)
            incorrect_assignment = accepted & (~closest_is_expected) & (~expected_unknown)
            known_count = int((~expected_unknown).sum())
            unknown_count = int(expected_unknown.sum())
            threshold_rows.append({
                "threshold": float(threshold),
                "known_sample_count": known_count,
                "unknown_sample_count": unknown_count,
                "known_recognition_count": int(known_ok[~expected_unknown].sum()),
                "known_recognition_rate": float(known_ok[~expected_unknown].mean()),
                "unknown_rejection_count": int((~accepted)[expected_unknown].sum()),
                "unknown_rejection_rate": float((~accepted)[expected_unknown].mean()),
                "false_acceptance_count": int(false_accept.sum()),
                "false_acceptance_rate": (
                    float(false_accept.sum() / unknown_count) if unknown_count else np.nan
                ),
                "false_rejection_count": int(false_reject.sum()),
                "false_rejection_rate": float(false_reject.sum() / known_count),
                "incorrect_identity_assignment_count": int(incorrect_assignment.sum()),
                "incorrect_identity_assignment_rate": float(
                    incorrect_assignment.sum() / known_count
                ),
            })
        threshold_table = pd.DataFrame(threshold_rows)
        threshold_table.to_csv(metrics_dir / "threshold_results.csv", index=False)
        plt.plot(threshold_table["threshold"], threshold_table["known_recognition_rate"],
                 marker="o", label="Known recognition rate")
        if unknown_count:
            plt.plot(threshold_table["threshold"], threshold_table["unknown_rejection_rate"],
                     marker="o", label="Unknown rejection rate")
        else:
            plt.plot(threshold_table["threshold"], threshold_table["false_rejection_rate"],
                     marker="o", label="Known false-rejection rate")
        plt.xlabel("Maximum accepted Euclidean distance"); plt.ylabel("Rate")
        plt.ylim(0, 1.05); plt.legend(); plt.title("Threshold trade-off on collected data")
        plt.tight_layout(); plt.savefig(figure_dir / "threshold_analysis.png", dpi=160); plt.close()
        generated.append("threshold_analysis.png")

    labels = sorted(set(results["expected_identity"]))
    predicted_labels = set(results["predicted_identity"])
    if len(results) >= 4 and len(labels) >= 2 and predicted_labels.issubset(labels):
        matrix = confusion_matrix(results["expected_identity"], results["predicted_identity"], labels=labels)
        display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        display.plot(cmap="Blues", xticks_rotation=45, colorbar=False)
        plt.tight_layout(); plt.savefig(figure_dir / "confusion_matrix.png", dpi=160); plt.close()
        generated.append("confusion_matrix.png")

    identity_summary = results.groupby("expected_identity")["correct"].agg(["sum", "count"])
    if len(identity_summary):
        identity_summary["incorrect"] = identity_summary["count"] - identity_summary["sum"]
        ax = identity_summary[["sum", "incorrect"]].rename(
            columns={"sum": "correct"}
        ).plot.bar(stacked=True, color=["#4daf4a", "#e41a1c"], rot=0)
        ax.set_ylabel("Test image count")
        ax.set_title("Recognition outcomes by evaluated identity")
        plt.tight_layout(); plt.savefig(figure_dir / "recognition_summary.png", dpi=160); plt.close()
        generated.append("recognition_summary.png")

    failures = results[~results["correct"]].head(6)
    readable_failures = []
    for _, row in failures.iterrows():
        image = cv2.imread(str(row["image"]))
        if image is not None:
            readable_failures.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), row))
    if readable_failures:
        columns = min(3, len(readable_failures))
        rows = int(np.ceil(len(readable_failures) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, (image, row) in zip(axes.flat, readable_failures):
            distance = "n/a" if pd.isna(row["distance"]) else f"{row['distance']:.3f}"
            axis.imshow(image)
            axis.set_title(
                f"Expected: {row['expected_identity']}\nPredicted: {row['predicted_identity']} | d={distance}\n"
                f"Condition: {row['condition']}", fontsize=9
            )
            axis.axis("off")
        figure.suptitle("Observed failures from consented local test images")
        plt.tight_layout(); plt.savefig(figure_dir / "failure_cases.png", dpi=160); plt.close()
        generated.append("failure_cases.png")
    return generated


def main() -> None:
    args = parse_args()
    if _IMPORT_ERROR is not None:
        raise SystemExit(f"Missing dependency ({_IMPORT_ERROR}). Install requirements.txt first.")
    if args.threshold <= 0:
        raise SystemExit("--threshold must be greater than 0")
    ensure_project_directories()
    database = load_database(args.database)
    samples = collect_samples(args.test_dir, args.manifest)
    if not samples:
        raise SystemExit("DATA COLLECTION REQUIRED: no test images found. See EXPERIMENTS.md.")
    results = pd.DataFrame(evaluate_image(sample, database, args.threshold) for sample in samples)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results, index=False)
    metrics = calculate_metrics(results)
    metrics_path = args.results.parent / "evaluation_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    generated = generate_figures(results)
    print(json.dumps(metrics, indent=2))
    print(f"Saved per-image results to {args.results}")
    print(f"Generated figures: {', '.join(generated) if generated else 'none (more labeled data required)'}")


if __name__ == "__main__":
    main()
