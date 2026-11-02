"""Task 3 - Batch Inference: predict 20 unseen records together, save CSV."""
import joblib
import pandas as pd

from config import BATCH_OUT_PATH, FEATURES, LABELS, MODEL_PATH, TARGET, UNSEEN_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    unseen = pd.read_csv(UNSEEN_PATH)
    batch = unseen.iloc[1:21].copy()               # 20 unseen records

    # All 20 predicted in ONE call
    batch["Predicted"] = model.predict(batch[FEATURES])
    batch["Survival_Probability"] = model.predict_proba(batch[FEATURES])[:, 1].round(4)
    batch["Predicted_Label"] = batch["Predicted"].map(LABELS)

    out = batch[["PassengerId", "Name"] + FEATURES + [TARGET, "Predicted",
                                                      "Survival_Probability", "Predicted_Label"]]
    out.to_csv(BATCH_OUT_PATH, index=False)

    print("===== BATCH INFERENCE (20 records) =====")
    print(out[["PassengerId", "Sex", "Pclass", "Age", TARGET, "Predicted",
               "Survival_Probability"]].to_string(index=False))
    acc = (out[TARGET] == out["Predicted"]).mean()
    print(f"\nRecords predicted : {len(out)}")
    print(f"Batch accuracy    : {acc:.2%}")
    print(f"Results saved to  : {BATCH_OUT_PATH.name}")


if __name__ == "__main__":
    main()
