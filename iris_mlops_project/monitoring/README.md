# Prediction monitoring

`app.py` records prediction timestamp, input measurements, predicted species, and optional user-supplied correct label in `monitoring/predictions.csv`.

- Prediction counts and predicted-species distribution are displayed in the Streamlit app.
- Feedback accuracy is calculated only for rows where a correct label was supplied.
- Download the CSV from the app to inspect it.

**Storage limitation:** Streamlit Community Cloud's local filesystem is not durable across restarts/redeployments. For persistent production monitoring, replace this local CSV with a managed database or telemetry service.

## Automated retraining

The workflow `.github/workflows/retrain.yml` runs on a weekly schedule, can be started manually from GitHub Actions, and runs when `data/iris_labeled.csv` changes. The retraining script evaluates candidate models on a held-out split and updates the saved model only if its accuracy exceeds the current model's accuracy. A commit to the repository then triggers Streamlit Cloud to redeploy the app.

To supply additional labeled samples, create `data/iris_labeled.csv` with columns:

`sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),species`

`species` must be `setosa`, `versicolor`, or `virginica`. Use genuine labeled samples; do not add predictions as if they were ground-truth labels.
