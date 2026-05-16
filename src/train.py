"""
Training script for the Iris classifier.

What this does (MLOps perspective):
- Loads data
- Trains a model
- Logs parameters, metrics, and the model artifact to MLflow
- Registers the best model in the MLflow Model Registry
"""

import os
import json
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "iris-classifier"
MODEL_NAME = "iris-rf-model"          # name in the Model Registry
MODEL_OUTPUT_DIR = "models"

# Hyperparameters — easy to swap in a real pipeline (e.g. from a config file)
HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
    "test_size": 0.2,
}


def load_data():
    """Load the Iris dataset and return train/test splits."""
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)  # ['setosa', 'versicolor', 'virginica']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=HYPERPARAMS["test_size"],
        random_state=HYPERPARAMS["random_state"],
        stratify=y,  # keep class balance in both splits
    )
    return X_train, X_test, y_train, y_test, class_names


def train(X_train, y_train):
    """Train a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=HYPERPARAMS["n_estimators"],
        max_depth=HYPERPARAMS["max_depth"],
        random_state=HYPERPARAMS["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, X_test, y_train, y_test, class_names):
    """Compute metrics for MLflow logging."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Cross-validation gives a more honest estimate than a single split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    report = classification_report(
        y_test, model.predict(X_test), target_names=class_names, output_dict=True
    )

    metrics = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std_accuracy": round(cv_scores.std(), 4),
    }
    return metrics, report


def save_model_locally(model, class_names):
    """Save model + metadata to disk (used by the API container)."""
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_OUTPUT_DIR}/model.joblib")

    metadata = {"class_names": class_names, "features": [
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ]}
    with open(f"{MODEL_OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {MODEL_OUTPUT_DIR}/")


def main():
    # Point MLflow at our tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")

    X_train, X_test, y_train, y_test, class_names = load_data()
    print(f"Data loaded — train: {len(X_train)} samples, test: {len(X_test)} samples")

    # Every mlflow.start_run() creates a new "run" — think of it as a git commit
    # for your experiment. You can compare runs in the MLflow UI.
    with mlflow.start_run() as run:
        print(f"MLflow run ID: {run.info.run_id}")

        # 1. Log hyperparameters
        mlflow.log_params(HYPERPARAMS)

        # 2. Train
        model = train(X_train, y_train)

        # 3. Evaluate
        metrics, report = evaluate(model, X_train, X_test, y_train, y_test, class_names)
        print(f"Metrics: {metrics}")

        # 4. Log metrics
        mlflow.log_metrics(metrics)

        # 5. Log the model artifact — MLflow wraps it with a standard interface
        #    so any downstream tool (serving, registry) can load it the same way
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,  # auto-registers in Model Registry
            input_example=X_train[:3],
        )

        # 6. Log a classification report as a JSON artifact (nice for auditing)
        mlflow.log_dict(report, "classification_report.json")

        print(f"\nTraining complete!")
        print(f"  Test accuracy : {metrics['test_accuracy']:.2%}")
        print(f"  CV accuracy   : {metrics['cv_mean_accuracy']:.2%} ± {metrics['cv_std_accuracy']:.4f}")
        print(f"\nView run in MLflow UI: {MLFLOW_TRACKING_URI}/#/experiments")

    # Also save locally so the API can load it without hitting MLflow
    save_model_locally(model, class_names)


if __name__ == "__main__":
    main()
