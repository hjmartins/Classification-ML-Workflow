# models/model_utils.py
import joblib
import pandas as pd
import json
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    y_test = pd.read_csv(ARTIFACTS_DIR / "y_test.csv").squeeze()
    y_pred = pd.read_csv(ARTIFACTS_DIR / "y_pred.csv").squeeze()
    y_proba = pd.read_csv(ARTIFACTS_DIR / "y_proba.csv").squeeze()
    X_train = pd.read_csv(ARTIFACTS_DIR / "X_train.csv").squeeze()
    X_test = pd.read_csv(ARTIFACTS_DIR / "X_test.csv").squeeze()
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    return model, y_test, y_pred, y_proba, metrics, X_train, X_test
