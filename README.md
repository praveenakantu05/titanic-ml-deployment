# ML Model Deployment Strategies and Model Serving using FastAPI

## Project description
Trains a classifier on the Titanic dataset, saves the complete scikit-learn pipeline once
(`model.pkl`) and reuses it for **online**, **batch** and **streaming** inference and for a
**FastAPI** REST service.

## Dataset
Titanic - Machine Learning from Disaster (Kaggle): https://www.kaggle.com/competitions/titanic/data
(`data/train.csv` is used; 891 labelled rows.)

## Project structure
| File | Purpose |
|---|---|
| `task1_train.py` | Task 1 - preprocess, train, evaluate, save `model.pkl` |
| `task2_online_inference.py` | Task 2 - 1 unseen record |
| `task3_batch_inference.py` | Task 3 - 20 unseen records -> `batch_predictions.csv` |
| `task4_streaming_inference.py` | Task 4 - 10 records, 1 sec delay |
| `app.py` | Task 5 - FastAPI (`GET /health`, `POST /predict`) |
| `client.py` | Task 5 - Python test client |
| `make_report.py` / `report.pdf` | Task 6 - observations report |
| `config.py` | shared paths/columns |

## Execution steps (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python task1_train.py            # trains + saves model.pkl + unseen_records.csv
python task2_online_inference.py
python task3_batch_inference.py
python task4_streaming_inference.py

uvicorn app:app --reload         # keep running; open http://127.0.0.1:8000/docs
python client.py                 # run in a SECOND terminal
python make_report.py            # (optional) regenerate report.pdf
```

## Swagger UI test inputs
Valid:
```json
{"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0, "Parch": 0, "Fare": 80, "Embarked": "S"}
```
Invalid (returns 422):
```json
{"Pclass": 7, "Sex": "unknown", "Age": -5, "SibSp": 0, "Parch": 0, "Fare": 10, "Embarked": "S"}
```

## Results
- Model: Random Forest in a scikit-learn Pipeline
- Train/test split: 712 / 179 (stratified, `random_state=42`)
- Test accuracy: **81.01%** (train 86.24%)
- Online inference: 1 record; Batch: 20 records (`batch_predictions.csv`); Streaming: 10 records
- FastAPI: `/health` -> 200, valid `/predict` -> 200, invalid `/predict` -> 422
