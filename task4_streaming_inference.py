"""Task 4 - Streaming Inference: 10 unseen records, one at a time, 1 sec delay."""
import time

import joblib
import pandas as pd

from config import FEATURES, LABELS, MODEL_PATH, UNSEEN_PATH


def main() -> None:
    model = joblib.load(MODEL_PATH)
    unseen = pd.read_csv(UNSEEN_PATH)
    stream = unseen.iloc[21:31]                    # 10 unseen records

    print("===== STREAMING INFERENCE (10 records, 1 sec delay) =====")
    for i, (_, row) in enumerate(stream.iterrows(), start=1):
        time.sleep(1)                              # simulate data arriving
        record = row[FEATURES].to_frame().T        # a single incoming record
        pred = int(model.predict(record)[0])
        proba = float(model.predict_proba(record)[0][1])
        print(f"[{time.strftime('%H:%M:%S')}] Record {i:02d} | PassengerId={int(row['PassengerId'])} "
              f"| {row['Sex']}, Pclass={row['Pclass']}, Age={row['Age']} "
              f"-> {LABELS[pred]} (prob={proba:.3f})")
    print("Stream finished.")


if __name__ == "__main__":
    main()
