# Disease Prediction System — Web App
### Group Project | Python + Flask + scikit-learn

A machine learning web application that predicts diseases from symptoms using a **Random Forest Classifier**.

---

## Project Structure

```
disease_prediction/
├── app.py                  ← Flask backend (ML model lives here)
├── requirements.txt        ← Python dependencies
├── README.md
└── templates/
    └── index.html          ← Frontend website
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask app
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

---

## How It Works

1. **Dataset** — A symptom-disease table is created in Python (Fever, Cough, Headache, Fatigue, Body Pain → Disease).
2. **Model** — A Random Forest Classifier (100 trees) is trained using scikit-learn.
3. **Web API** — Flask exposes a `/predict` POST endpoint that accepts symptom inputs as JSON.
4. **Frontend** — `index.html` sends symptom selections via `fetch()` to Flask and displays the result.

---

## Diseases Predicted
| Disease   | Symptoms Typical                        |
|-----------|-----------------------------------------|
| COVID-19  | Fever, Cough, Fatigue, Body Pain        |
| Flu       | Fever, Headache, Fatigue, Body Pain     |
| Migraine  | Headache, Fatigue                       |
| Cold      | Cough, Fatigue                          |
| Healthy   | No significant symptoms                 |

---

> **Disclaimer:** Educational project only. Not for real medical use.
