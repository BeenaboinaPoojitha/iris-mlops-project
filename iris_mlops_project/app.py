"""Streamlit interface for the trained Iris classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

st.set_page_config(page_title="Iris MLOps Demo", page_icon="🌸", layout="wide")
st.title("🌸 Iris Flower Classification — MLOps Demo")
st.write("Choose four flower measurements to predict the Iris species using the best model selected during training.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.warning("The model is not trained yet. In the project terminal, run: `python train.py`")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
iris = load_iris(as_frame=True)
feature_names = list(iris.feature_names)
target_names = list(iris.target_names)

with st.sidebar:
    st.header("Flower measurements")
    defaults = [5.1, 3.5, 1.4, 0.2]
    ranges = [(4.3, 7.9, 0.1), (2.0, 4.4, 0.1), (1.0, 6.9, 0.1), (0.1, 2.5, 0.1)]
    values = []
    for feature, default, (minimum, maximum, step) in zip(feature_names, defaults, ranges):
        values.append(st.slider(feature.title(), float(minimum), float(maximum), float(default), float(step)))
    predict_button = st.button("Predict species", type="primary", use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Selected model")
    st.metric("Best model", metadata["best_model"].replace("_", " ").title())
    st.metric("Test accuracy", f"{metadata['best_accuracy'] * 100:.1f}%")
    st.caption("Accuracy is from a single held-out test split and is not a guarantee of future performance.")

with right:
    st.subheader("Prediction")
    if predict_button:
        input_df = pd.DataFrame([values], columns=feature_names)
        predicted_class = int(model.predict(input_df)[0])
        st.success(f"Predicted species: **{target_names[predicted_class].title()}**")
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
results = pd.DataFrame(metadata["results"])
if not results.empty:
    display = results.copy()
    display["model"] = display["model"].str.replace("_", " ").str.title()
    display["accuracy"] = display["accuracy"].map(lambda x: f"{x:.1%}")
    display["macro_f1"] = display["macro_f1"].map(lambda x: f"{x:.3f}")
    st.dataframe(display[["model", "accuracy", "macro_f1"]], hide_index=True, use_container_width=True)
    st.bar_chart(results.set_index("model")[["accuracy", "macro_f1"]])

st.caption("Dataset: the built-in scikit-learn Iris dataset. This is an educational demo, not a production deployment.")
