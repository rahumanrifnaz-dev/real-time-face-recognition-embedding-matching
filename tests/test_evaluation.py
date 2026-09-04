from pathlib import Path

import pandas as pd

from evaluate import calculate_metrics, generate_figures


def known_only_results() -> pd.DataFrame:
    return pd.DataFrame([
        {"image": "a1.jpg", "expected_identity": "Obama", "predicted_identity": "Obama",
         "closest_identity": "Obama", "condition": "unspecified", "distance": 0.35,
         "threshold": 0.60, "latency_ms": 10.0, "correct": True},
        {"image": "a2.jpg", "expected_identity": "Obama", "predicted_identity": "Obama",
         "closest_identity": "Obama", "condition": "unspecified", "distance": 0.45,
         "threshold": 0.60, "latency_ms": 12.0, "correct": True},
        {"image": "b1.jpg", "expected_identity": "Elon", "predicted_identity": "Elon",
         "closest_identity": "Elon", "condition": "unspecified", "distance": 0.40,
         "threshold": 0.60, "latency_ms": 11.0, "correct": True},
        {"image": "b2.jpg", "expected_identity": "Elon", "predicted_identity": "Elon",
         "closest_identity": "Elon", "condition": "unspecified", "distance": 0.50,
         "threshold": 0.60, "latency_ms": 13.0, "correct": True},
    ])


def test_known_only_evaluation_writes_threshold_results_without_unknown_claims(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("outputs/figures").mkdir(parents=True)
    Path("outputs/metrics").mkdir(parents=True)
    results = known_only_results()

    metrics = calculate_metrics(results)
    figures = generate_figures(results)
    thresholds = pd.read_csv("outputs/metrics/threshold_results.csv")

    assert metrics["number_of_test_images"] == 4
    assert metrics["accuracy"] == 1.0
    assert thresholds["unknown_sample_count"].eq(0).all()
    assert thresholds["false_acceptance_rate"].isna().all()
    assert "threshold_analysis.png" in figures
    assert "recognition_summary.png" in figures
