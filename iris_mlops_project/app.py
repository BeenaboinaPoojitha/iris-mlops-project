"""Streamlit interface for Iris classification with lightweight prediction monitoring."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
LOG_DIR = BASE_DIR / "monitoring"
LOG_PATH = LOG_DIR / "predictions.csv"

st.set_page_config(page_title="Iris MLOps Demo", page_icon="🌸", layout="wide")
st.title("🌸 Iris Flower Classification — MLOps Demo")
st.write("Predict an Iris species and inspect basic prediction monitoring.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The model is not trained yet. In the project terminal, run: `python train.py`")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
iris = load_iris(as_frame=True)
feature_names = list(iris.feature_names)
target_names = list(iris.target_names)

LOG_FIELDS = ["timestamp_utc", *feature_names, "predicted_species", "actual_species", "correct"]


def append_prediction(values, predicted_species, actual_species=""):
    """Append one prediction event to a CSV log (best-effort local monitoring)."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        needs_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **dict(zip(feature_names, values)),
            "predicted_species": predicted_species,
            "actual_species": actual_species,
            "correct": (str(predicted_species) == str(actual_species)) if actual_species else "",
        }
        with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        return True
    except OSError:
        return False


with st.sidebar:
    st.header("Flower measurements")
    defaults = [5.1, 3.5, 1.4, 0.2]
    ranges = [(4.3, 7.9, 0.1), (2.0, 4.4, 0.1), (1.0, 6.9, 0.1), (0.1, 2.5, 0.1)]
    values = []
    for feature, default, (minimum, maximum, step) in zip(feature_names, defaults, ranges):
        values.append(st.slider(feature.title(), float(minimum), float(maximum), float(default), float(step)))
    actual_option = st.selectbox(
        "Known correct species (optional)",
        ["Not known", *[name.title() for name in target_names]],
        help="Choose a true label only if you know it. This enables feedback-based accuracy monitoring.",
    )
    predict_button = st.button("Predict species", type="primary", use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Selected model")
    st.metric("Best model", metadata["best_model"].replace("_", " ").title())
    st.metric("Held-out test accuracy", f"{metadata['best_accuracy'] * 100:.1f}%")
    st.caption("This is the recorded test score from training, not a guarantee of future accuracy.")

with right:
    st.subheader("Prediction")
    if predict_button:
        input_df = pd.DataFrame([values], columns=feature_names)
        predicted_class = int(model.predict(input_df)[0])
        predicted_species = target_names[predicted_class].title()
        actual_species = "" if actual_option == "Not known" else actual_option
        logged = append_prediction(values, predicted_species, actual_species)
        st.success(f"Predicted species: **{predicted_species}**")
        if actual_species:
            if actual_species == predicted_species:
                st.success("Feedback recorded: prediction matches the supplied correct label.")
            else:
                st.warning(f"Feedback recorded: supplied label was {actual_species}.")
        if not logged:
            st.warning("Prediction worked, but this environment could not write the monitoring log.")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability_df = pd.DataFrame({"Species": [name.title() for name in target_names], "Probability": probabilities})
            st.bar_chart(probability_df.set_index("Species"))
            st.dataframe(probability_df.assign(Probability=probability_df["Probability"].map(lambda x: f"{x:.1%}")), hide_index=True, use_container_width=True)
        st.write("Input values")
        st.dataframe(input_df, hide_index=True, use_container_width=True)
    else:
        st.info("Adjust the measurements in the sidebar and click **Predict species**.")

st.divider()
st.subheader("Experiment comparison (MLflow-tracked)")
results = pd.DataFrame(metadata.get("results", []))
if not results.empty:
    display = results.copy()
    display["model"] = display["model"].str.replace("_", " ").str.title()
    display["accuracy"] = display["accuracy"].map(lambda x: f"{x:.1%}")
    display["macro_f1"] = display["macro_f1"].map(lambda x: f"{x:.3f}")
    st.dataframe(display[["model", "accuracy", "macro_f1"]], hide_index=True, use_container_width=True)
    st.bar_chart(results.set_index("model")[["accuracy", "macro_f1"]])

st.divider()
st.subheader("Prediction monitoring")
if LOG_PATH.exists():
    try:
        logs = pd.read_csv(LOG_PATH)
        if not logs.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Logged predictions", len(logs))
            m2.metric("Distinct predicted species", logs["predicted_species"].nunique())
            labelled = logs[logs["actual_species"].fillna("").astype(str).str.len() > 0]
            if not labelled.empty:
                correct = labelled["correct"].astype(str).str.lower().eq("true").mean()
                m3.metric("Feedback accuracy", f"{correct:.1%}", help="Accuracy only among predictions with a supplied correct label.")
            else:
                m3.metric("Labelled feedback", 0)
            st.caption("Prediction distribution")
            st.bar_chart(logs["predicted_species"].value_counts())
            st.dataframe(logs.tail(20), hide_index=True, use_container_width=True)
            st.download_button("Download prediction log CSV", logs.to_csv(index=False), "iris_prediction_logs.csv", "text/csv")
        else:
            st.info("No predictions logged yet. Make a prediction to start monitoring.")
    except (OSError, pd.errors.ParserError) as exc:
        st.warning(f"Could not read the monitoring log: {exc}")
else:
    st.info("No predictions logged yet. Make a prediction to start monitoring.")

st.warning(
    "Demo limitation: Streamlit Community Cloud's local files are not durable storage. "
    "The CSV log may be lost on restart/redeployment; use a persistent database or monitoring service for production."
)
st.caption("Dataset: built-in scikit-learn Iris dataset. Educational demonstration, not a production service.")
