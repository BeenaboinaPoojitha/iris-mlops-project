"""Train and compare Iris classifiers while tracking experiments with MLflow."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    tracking_uri = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classification"))

    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE),
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    }

    results: list[dict] = []
    best_name = ""
    best_model = None
    best_accuracy = -1.0

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = float(accuracy_score(y_test, predictions))
            macro_f1 = float(f1_score(y_test, predictions, average="macro"))
            mlflow.log_params({"model_name": name, "test_size": TEST_SIZE, "random_state": RANDOM_STATE})
            mlflow.log_metrics({"accuracy": accuracy, "macro_f1": macro_f1})

            report = classification_report(y_test, predictions, target_names=iris.target_names, output_dict=True)
            report_path = ARTIFACT_DIR / f"{name}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions, display_labels=iris.target_names, ax=ax, colorbar=False
            )
            ax.set_title(f"{name.replace('_', ' ').title()} confusion matrix")
            fig.tight_layout()
            matrix_path = ARTIFACT_DIR / f"{name}_confusion_matrix.png"
            fig.savefig(matrix_path, dpi=130)
            plt.close(fig)
            mlflow.log_artifact(str(matrix_path), artifact_path="evaluation")
            mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=[
            "sklearn.metrics._dist_metrics.EuclideanDistance64",
            "sklearn.neighbors._kd_tree.KDTree",
            "sklearn.tree._tree.Tree"
                            ]
                        )

            results.append({
                "model": name,
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "run_id": run.info.run_id,
            })
            print(f"{name}: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_name = name
                best_model = model

    assert best_model is not None
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "best_accuracy": round(best_accuracy, 4),
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names),
        "results": sorted(results, key=lambda row: row["accuracy"], reverse=True),
        "dataset": "scikit-learn Iris dataset (150 samples; 4 features; 3 classes)",
        "note": "Results are from a single stratified 80/20 split; accuracy can vary with split and settings.",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nBest model: {best_name} (accuracy={best_accuracy:.4f})")
    print(f"Saved model: {MODEL_DIR / 'best_model.joblib'}")
    print(f"MLflow tracking store: {tracking_uri}")


if __name__ == "__main__":
    main()
