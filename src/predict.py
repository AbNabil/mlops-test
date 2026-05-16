"""
Prediction logic — shared between the API and any batch inference scripts.

Keeping this separate from the API layer means you can reuse it in:
- A batch job that scores a CSV file
- A test suite
- A CLI tool
"""

import json
import os
import numpy as np
import joblib

MODEL_DIR = os.getenv("MODEL_DIR", "models")


class IrisPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        model_path = os.path.join(model_dir, "model.joblib")
        metadata_path = os.path.join(model_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run src/train.py first."
            )

        self.model = joblib.load(model_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        self.class_names = metadata["class_names"]
        self.feature_names = metadata["features"]

    def predict(self, features: dict) -> dict:
        """
        Run inference on a single sample.

        Args:
            features: dict with keys matching self.feature_names

        Returns:
            dict with 'prediction' (class name) and 'confidence' (probability)
        """
        # Build the feature vector in the correct order
        X = np.array([[features[f] for f in self.feature_names]])

        class_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        return {
            "prediction": self.class_names[class_idx],
            "confidence": round(float(probabilities[class_idx]), 4),
            "all_probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(self.class_names, probabilities)
            },
        }


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = IrisPredictor()

    test_samples = [
        # (expected class, features)
        ("setosa",     {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}),
        ("versicolor", {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}),
        ("virginica",  {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.4}),
    ]

    print("Smoke test predictions:")
    for expected, sample in test_samples:
        result = predictor.predict(sample)
        status = "✓" if result["prediction"] == expected else "✗"
        print(f"  {status} Expected: {expected:12s} | Got: {result['prediction']:12s} | Confidence: {result['confidence']:.2%}")
