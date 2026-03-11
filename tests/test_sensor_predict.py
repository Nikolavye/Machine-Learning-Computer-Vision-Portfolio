import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sensor_anomaly_detection import predict as predict_module


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "sensor_anomaly_detection" / "data_sensors.csv"


class SensorPredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scaler, cls.model = predict_module.load_model()
        cls.df = pd.read_csv(DATA_PATH)

    def test_predict_returns_expected_shapes(self):
        X = self.df.loc[:4, predict_module.FEATURE_NAMES].to_numpy()
        labels, confidences = predict_module.predict(self.scaler, self.model, X)

        self.assertEqual(labels.shape, (5,))
        self.assertEqual(confidences.shape, (5,))
        self.assertTrue(np.all(confidences >= 0.0))
        self.assertTrue(np.all(confidences <= 1.0))

    def test_select_feature_matrix_prefers_named_columns(self):
        row = {f"Sensor {i}": float(i) for i in range(20)}
        shuffled = pd.DataFrame([row])[list(reversed(row.keys()))]

        X = predict_module.select_feature_matrix(shuffled)

        self.assertEqual(X.tolist(), [[9.0, 2.0, 13.0]])

    def test_predict_from_csv_appends_prediction_columns(self):
        result = predict_module.predict_from_csv(self.scaler, self.model, str(DATA_PATH))

        self.assertEqual(len(result), len(self.df))
        self.assertIn("Predicted_Label", result.columns)
        self.assertIn("Confidence", result.columns)

    def test_select_feature_matrix_rejects_missing_schema(self):
        with self.assertRaises(ValueError):
            predict_module.select_feature_matrix(pd.DataFrame({"Sensor 9": [1.0], "Sensor 2": [2.0]}))


if __name__ == "__main__":
    unittest.main()
