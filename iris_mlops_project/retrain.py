"""Train candidate Iris classifiers and promote only when validation accuracy improves.

Optional extra labeled data can be added at data/iris_labeled.csv with columns:
sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), species
where species is setosa, versicolor, or virginica.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
DATA_PATH = BASE_DIR / "data" / "iris_labeled.csv"
RANDOM_STATE = 42


def load_training_data():
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    if DATA_PATH.exists():
        extra = pd.read_csv(DATA_PATH)
        required = [*X.columns, "species"]
        missing = sorted(set(required) - set(extra.columns))
        if missing:
            raise ValueError(f"{DATA_PATH} is missing columns: {missing}")
        label_map = {name: i for i, name in enumerate(iris.target_names)}
        extra_y = extra["species"].astype(str).str.lower().map(label_map)
        if extra_y.isna().any():
            bad = sorted(extra.loc[extra_y.isna(), "species"].astype(str).unique())
            raise ValueError(f"Unknown species labels in {DATA_PATH}: {bad}")
        extra_X = extra[X.columns].apply(pd.to_numeric, errors="raise")
        X = pd.concat([X, extra_X], ignore_index=True)
        y = pd.concat([y.reset_index(drop=True), extra_y.astype(int).reset_index(drop=True)], ignore_index=True)
        print(f"Loaded {len(extra)} additional labeled rows from {DATA_PATH}")
    return X, y, list(iris.target_names)


def candidates():
    return {
        "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
        "decision_tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_STATE),
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y, target_names = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    current_accuracy = -1.0
    if MODEL_PATH.exists():
        current_model = joblib.load(MODEL_PATH)
        current_accuracy = float(accuracy_score(y_test, current_model.predict(X_test)))

    best_name, best_model, best_accuracy, best_f1 = "", None, -1.0, -1.0
    results = []
    for name, model in candidates().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, pred))
        macro_f1 = float(f1_score(y_test, pred, average="macro"))
        results.append({"model": name, "accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4)})
        print(f"{name}: accuracy={accuracy:.4f}, macro_f1={macro_f1:.4f}")
        if (accuracy, macro_f1) > (best_accuracy, best_f1):
            best_name, best_model, best_accuracy, best_f1 = name, model, accuracy, macro_f1

    if best_model is None:
        raise RuntimeError("No candidate model was trained.")

    promoted = current_accuracy < 0 or best_accuracy > current_accuracy
    if promoted:
        joblib.dump(best_model, MODEL_PATH)
        metadata = {
            "best_model": best_name,
            "best_accuracy": round(best_accuracy, 4),
            "feature_names": list(X.columns),
            "target_names": target_names,
            "results": sorted(results, key=lambda row: row["accuracy"], reverse=True),
            "dataset": f"scikit-learn Iris dataset plus {max(0, len(X) - 150)} optional labeled rows",
            "note": "Model promoted only when its held-out accuracy improves over the current model.",
            "validation_accuracy": round(best_accuracy, 4),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"PROMOTED {best_name}: accuracy={best_accuracy:.4f}; previous={current_accuracy:.4f}")
    else:
        print(f"No update: candidate accuracy={best_accuracy:.4f}; current model accuracy={current_accuracy:.4f}")
    print("Retraining job completed.")


if __name__ == "__main__":
    main()
