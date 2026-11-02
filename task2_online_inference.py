"""Task 2 - Online Inference: load saved model, predict ONE unseen passenger."""
import joblib
import pandas as pd

from config import FEATURES, LABELS, MODEL_PATH, TARGET, UNSEEN_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    print("Model loaded from:", MODEL_PATH.name)

    unseen = pd.read_csv(UNSEEN_PATH)
    record = unseen.iloc[[0]]                      # one unseen passenger

    pred = int(model.predict(record[FEATURES])[0])
    proba = float(model.predict_proba(record[FEATURES])[0][1])

    print("\n===== ONLINE INFERENCE (single record) =====")
    print(f"Passenger : {record['Name'].iloc[0]} (ID {record['PassengerId'].iloc[0]})")
    print(record[FEATURES].to_string(index=False))
    print(f"\nPredicted class      : {pred} -> {LABELS[pred]}")
    print(f"Survival probability : {proba:.4f}")
    print(f"Actual label         : {int(record[TARGET].iloc[0])} -> {LABELS[int(record[TARGET].iloc[0])]}")


if __name__ == "__main__":
    main()
