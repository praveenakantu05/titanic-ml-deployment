"""Python client to test the FastAPI service (start the server first!).

Terminal 1:  uvicorn app:app --reload
Terminal 2:  python client.py
"""
import json

import requests

BASE_URL = "http://127.0.0.1:8000"


def show(title: str, resp: requests.Response) -> None:
    print(f"\n--- {title} ---")
    print("HTTP status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2))


def main() -> None:
    # 1. Health check
    show("GET /health", requests.get(f"{BASE_URL}/health", timeout=10))

    # 2. Valid request
    valid = {"Pclass": 1, "Sex": "female", "Age": 29.0, "SibSp": 0,
             "Parch": 0, "Fare": 80.0, "Embarked": "S"}
    show("POST /predict (VALID)", requests.post(f"{BASE_URL}/predict", json=valid, timeout=10))

    # 3. Another valid request (male, 3rd class)
    valid2 = {"Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1,
              "Parch": 0, "Fare": 7.25, "Embarked": "S"}
    show("POST /predict (VALID #2)", requests.post(f"{BASE_URL}/predict", json=valid2, timeout=10))

    # 4. Invalid request (Pclass=7, Sex wrong, Age negative) -> expect 422
    invalid = {"Pclass": 7, "Sex": "unknown", "Age": -5, "SibSp": 0,
               "Parch": 0, "Fare": 10.0, "Embarked": "S"}
    show("POST /predict (INVALID)", requests.post(f"{BASE_URL}/predict", json=invalid, timeout=10))


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Could not connect. Start the API first:  uvicorn app:app --reload")
