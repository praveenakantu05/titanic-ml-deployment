"""Task 1 - Model Training.

Loads Kaggle Titanic train.csv, preprocesses, trains a classifier, evaluates
accuracy, and saves the COMPLETE pipeline (preprocessing + model) to model.pkl.
The held-out test split is also saved so all later tasks use truly UNSEEN data.
"""
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CATEGORICAL_FEATURES, DATA_PATH, FEATURES, METRICS_PATH,
                    MODEL_PATH, NUMERIC_FEATURES, TARGET, UNSEEN_PATH)


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                   min_samples_leaf=2, random_state=42)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing values (before preprocessing):")
    print(df[FEATURES].isnull().sum().to_string())

    # 2. Split labelled data into train / test (test = unseen records)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain size: {len(X_train)} | Unseen/Test size: {len(X_test)}")

    # Save unseen records (keep PassengerId, Name and true label for reference)
    unseen = df.loc[X_test.index, ["PassengerId", "Name"] + FEATURES + [TARGET]]
    unseen.to_csv(UNSEEN_PATH, index=False)
    print(f"Unseen records saved to: {UNSEEN_PATH.name}")

    # 3. Train the pipeline ONCE
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # 4. Evaluate
    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"\nTraining accuracy : {train_acc:.4f}")
    print(f"Testing accuracy  : {test_acc:.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, test_pred))
    print("\nClassification report:\n", classification_report(y_test, test_pred))

    # 5. Save the complete pipeline
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model pipeline saved to: {MODEL_PATH.name}")

    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "model": "RandomForestClassifier (inside sklearn Pipeline)",
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_accuracy": round(train_acc, 4), "test_accuracy": round(test_acc, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
