#!/usr/bin/env python3
"""
Inference entry point for the Sensor Anomaly Detection model.

Loads the serialized scaler and Random Forest model from artifacts/,
accepts new sensor readings, and returns predictions with confidence scores.

Usage:
    python predict.py                         # interactive demo
    python predict.py --csv input.csv         # batch prediction from CSV
"""

import argparse
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SELECTED_FEATURES = [9, 2, 13]
FEATURE_NAMES = [f"Sensor {i}" for i in SELECTED_FEATURES]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "rf_model.joblib")


def load_model():
    """Load scaler and RF model from artifacts directory."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run sensor_clustering.py first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run sensor_clustering.py first."
        )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    log.info("Loaded scaler and RF model from %s", ARTIFACTS_DIR)
    return scaler, model


def predict(scaler, model, X_raw: np.ndarray) -> tuple:
    """Run prediction on raw sensor readings.

    Args:
        scaler: Fitted StandardScaler for the 3 selected features.
        model: Fitted RandomForestClassifier.
        X_raw: Array of shape (n_samples, 3) with columns [Sensor 9, Sensor 2, Sensor 13].

    Returns:
        (labels, confidences): predicted class labels and max class probabilities.
    """
    X_scaled = scaler.transform(X_raw)
    labels = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    confidences = proba.max(axis=1)
    return labels, confidences


def predict_from_csv(scaler, model, csv_path: str) -> pd.DataFrame:
    """Run prediction on a CSV file with 20 sensor columns."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 20:
        raise ValueError(f"Expected >=20 sensor columns, got {df.shape[1]}")

    X_selected = df.iloc[:, SELECTED_FEATURES].values
    labels, confidences = predict(scaler, model, X_selected)

    df["Predicted_Label"] = labels
    df["Confidence"] = confidences
    return df


def main():
    parser = argparse.ArgumentParser(description="Sensor Anomaly Detection — Inference")
    parser.add_argument("--csv", type=str, help="Path to input CSV with 20 sensor columns")
    parser.add_argument("--output", type=str, help="Path to save predictions CSV")
    args = parser.parse_args()

    scaler, model = load_model()

    if args.csv:
        log.info("Running batch prediction on %s", args.csv)
        result = predict_from_csv(scaler, model, args.csv)
        output_path = args.output or args.csv.replace(".csv", "_predictions.csv")
        result.to_csv(output_path, index=False)
        log.info("Predictions saved to %s (%d samples)", output_path, len(result))
    else:
        log.info("Interactive demo — enter sensor values for [S9, S2, S13]")
        log.info("Example: 0.5 0.3 -0.2")
        while True:
            try:
                line = input("\n> ").strip()
                if not line or line.lower() in ("quit", "exit", "q"):
                    break
                values = [float(v) for v in line.split()]
                if len(values) != 3:
                    print(f"Expected 3 values (S9, S2, S13), got {len(values)}")
                    continue
                X = np.array([values])
                labels, confs = predict(scaler, model, X)
                print(f"  Predicted: Class {labels[0]}, Confidence: {confs[0]:.1%}")
            except (EOFError, KeyboardInterrupt):
                break
            except ValueError as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()
