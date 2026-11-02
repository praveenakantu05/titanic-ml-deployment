"""Shared paths and column definitions used by every task."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
UNSEEN_PATH = BASE_DIR / "data" / "unseen_records.csv"   # held-out test split (never used for training)
MODEL_PATH = BASE_DIR / "model.pkl"
BATCH_OUT_PATH = BASE_DIR / "batch_predictions.csv"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"

LABELS = {0: "Did NOT Survive", 1: "Survived"}
