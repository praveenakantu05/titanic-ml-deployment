"""Generates report.pdf (Task 6 - Observations). Run: python make_report.py"""
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from config import BASE_DIR, METRICS_PATH

m = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=11, leading=16)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], spaceBefore=12)

QA = [
    ("1. How does Online Inference differ from Batch Inference?",
     "Online inference predicts for a single record (or a very small number) immediately, as "
     "soon as the request arrives, so it is optimised for low latency - e.g. a user checking "
     "one passenger's survival chance through an API. Batch inference collects many records "
     "(here 20) and scores them together in one call, usually offline or on a schedule, and "
     "stores the results (batch_predictions.csv). It is optimised for throughput and "
     "efficiency rather than instant response."),
    ("2. What is the difference between Streaming Inference and Batch Inference?",
     "Batch inference works on a fixed, already-collected dataset that is processed all at "
     "once. Streaming inference handles data that arrives continuously: each record is "
     "scored individually as it arrives (here one record every 1 second) and the result is "
     "available immediately, without waiting for the full dataset. Streaming gives near "
     "real-time results on unbounded data, while batch gives results only after the whole "
     "group is processed."),
    ("3. Why is FastAPI used for model serving?",
     "FastAPI is fast (built on Starlette/ASGI), simple to write, and uses Python type hints. "
     "Pydantic gives automatic request validation, so invalid input is rejected with a clear "
     "422 error before it reaches the model. It auto-generates interactive Swagger UI (/docs) "
     "documentation for testing, supports async, and makes the model reachable by any "
     "client (web, mobile, other services) through a standard REST interface."),
    ("4. Why should the same trained model be reused for all inference strategies?",
     "Reusing one saved pipeline guarantees consistency: the same preprocessing and the same "
     "learned weights are applied whether a record is scored online, in a batch, in a stream "
     "or via the API, so the same input always gives the same prediction. It avoids "
     "training-serving skew, saves the cost of retraining, makes results reproducible and "
     "comparable, and means that only one artifact (model.pkl) has to be versioned, "
     "tested and maintained."),
]

doc = SimpleDocTemplate(str(BASE_DIR / "report.pdf"), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
s = [Paragraph("ML Model Deployment Strategies and Model Serving using FastAPI", ss["Title"]),
     Paragraph("<b>Dataset:</b> Titanic - Machine Learning from Disaster (Kaggle), train.csv", body),
     Paragraph("<b>Model:</b> Random Forest inside a scikit-learn Pipeline "
               "(imputation + scaling + one-hot encoding), saved as model.pkl with Joblib.", body)]
if m:
    s.append(Paragraph(f"<b>Split:</b> {m['train_rows']} training / {m['test_rows']} unseen test records. "
                       f"<b>Test accuracy:</b> {m['test_accuracy']:.2%} "
                       f"(train accuracy {m['train_accuracy']:.2%}).", body))
s.append(Paragraph("Task 6: Observations", h2))
for q, a in QA:
    s += [Paragraph(f"<b>{q}</b>", body), Spacer(1, 3), Paragraph(a, body), Spacer(1, 8)]
doc.build(s)
print("report.pdf created")
