"""Task 5 - Model serving with FastAPI.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from config import FEATURES, LABELS, MODEL_PATH

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the previously saved pipeline (model.pkl) once at startup
    if not MODEL_PATH.exists():
        raise RuntimeError("model.pkl not found. Run task1_train.py first.")
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Titanic Survival Prediction API",
              description="Serves the saved scikit-learn pipeline (model.pkl).",
              version="1.0.0", lifespan=lifespan)


class Passenger(BaseModel):
    """Input schema - validated by Pydantic."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"Pclass": 1, "Sex": "female", "Age": 29.0,
                    "SibSp": 0, "Parch": 0, "Fare": 80.0, "Embarked": "S"}})

    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2 or 3)")
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years (optional)")
    SibSp: int = Field(..., ge=0, le=10, description="Siblings/spouses aboard")
    Parch: int = Field(..., ge=0, le=10, description="Parents/children aboard")
    Fare: float = Field(..., ge=0, le=1000, description="Ticket fare")
    Embarked: Optional[Literal["S", "C", "Q"]] = Field(None, description="Port of embarkation")


class Prediction(BaseModel):
    prediction: int
    label: str
    survival_probability: float


@app.get("/health")
def health():
    """Check the API status."""
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict(passenger: Passenger):
    """Accept passenger details and return a survival prediction."""
    try:
        df = pd.DataFrame([passenger.model_dump()])[FEATURES]
        model = state["model"]
        pred = int(model.predict(df)[0])
        proba = float(model.predict_proba(df)[0][1])
    except Exception as exc:                       # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return Prediction(prediction=pred, label=LABELS[pred],
                      survival_probability=round(proba, 4))
