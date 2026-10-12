# Iris Flower Classification with MLflow (MLOps Mini Project)

A beginner-friendly Machine Learning + MLOps project. It trains four classifiers on the built-in Iris dataset, records experiment parameters and metrics with MLflow, saves the best model, and provides a Streamlit prediction dashboard.

## What is included?

- `train.py` — trains KNN, Decision Tree, Random Forest, and Logistic Regression; logs parameters, accuracy, macro-F1, classification reports, confusion matrices, and model artifacts to MLflow; saves the best model.
- `app.py` — interactive Streamlit app for predicting the Iris species and comparing model results.
- `models/` — generated at training time; contains `best_model.joblib` and `metadata.json`.
- `mlruns/` — generated MLflow experiment tracking data.
- `Dockerfile` and `docker-compose.yml` — container setup for the dashboard and MLflow UI.
- `.github/workflows/tests.yml` — basic automated dataset checks on GitHub Actions.

## Dataset

The project uses the Iris dataset bundled with scikit-learn. It contains 150 samples, four numeric features (sepal length, sepal width, petal length, petal width), and three classes (setosa, versicolor, virginica). No external CSV download is required.

## Run locally on Windows (VS Code / PowerShell)

1. Install Python 3.11 (recommended for easiest compatibility) and Git if needed.
2. Extract the ZIP and open the `iris_mlops_project` folder in VS Code.
3. Open **Terminal → New Terminal** and create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, try `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Train and track experiments:

   ```powershell
   python train.py
   ```

   This creates `models/` and `mlruns/` automatically. It prints the test accuracy for each model and saves the best one.
6. Start the MLflow tracking UI in a **second terminal** (keep the first terminal available):

   ```powershell
   mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
   ```

   Open `http://127.0.0.1:5000` to compare experiment runs.
7. Start the Streamlit app in another terminal:

   ```powershell
   streamlit run app.py
   ```

   Open the local URL shown, usually `http://localhost:8501`.

## Run using Docker

Install and start Docker Desktop first. From the project folder run:

```powershell
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

Stop the containers with `Ctrl+C`, then run `docker compose down`. Docker volumes keep the MLflow tracking data and generated model/artifacts between container recreations.

## Run tests

```powershell
pytest -q
```

## What does MLflow do here?

MLflow Tracking records each training run, parameters (such as model name and split settings), evaluation metrics (accuracy and macro-F1), model artifacts, classification reports, and confusion-matrix images. Visit the MLflow UI to compare runs. MLflow does not automatically make a model accurate; it helps organize and inspect experiments.

## MLOps workflow demonstrated

1. Load a known dataset.
2. Split into training and held-out test sets.
3. Train and evaluate several candidate models.
4. Track experiment details in MLflow.
5. Select and save the best model according to test accuracy.
6. Serve predictions through Streamlit.
7. Package the project with Docker and run automated basic tests in GitHub Actions.

## Important limitations

- This is an educational demo using the built-in Iris dataset, not a production service.
- The reported metric comes from one stratified 80/20 split. A single split is not a guarantee of generalization; use cross-validation and additional checks for more rigorous evaluation.
- The app is designed for the four numeric Iris features, not arbitrary user-uploaded data.
- No model registry, automatic production deployment, drift monitoring, or live data pipeline is configured in this starter project.
