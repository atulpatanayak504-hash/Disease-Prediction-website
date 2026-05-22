# Disease Prediction System using Machine Learning
# Group Project

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    'Fever': [1,1,0,1,0,1,0,0],
    'Cough': [1,1,1,0,0,1,0,1],
    'Headache': [1,0,1,1,0,1,0,0],
    'Fatigue': [1,1,1,0,0,1,0,1],
    'BodyPain': [1,1,0,1,0,1,0,0],
    'Disease': [
        'Flu',
        'COVID-19',
        'Migraine',
        'Flu',
        'Healthy',
        'COVID-19',
        'Healthy',
        'Cold'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Encoding Disease Labels
# -----------------------------
label_encoder = LabelEncoder()
df['Disease'] = label_encoder.fit_transform(df['Disease'])

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop('Disease', axis=1)
y = df['Disease']

# -----------------------------
# Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -----------------------------
# Model Accuracy
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
model_accuracy = round(accuracy * 100, 2)
print("Model Accuracy:", model_accuracy, "%")


# -----------------------------
# Disease Prediction Function
# -----------------------------
def predict_disease(fever, cough, headache, fatigue, bodypain):
    user_input = np.array([[fever, cough, headache, fatigue, bodypain]])
    prediction = model.predict(user_input)
    disease_name = label_encoder.inverse_transform(prediction)[0]

    # Health Suggestions
    suggestions = {
        "COVID-19": "Please consult a doctor immediately and isolate yourself from others.",
        "Flu":      "Take rest and drink plenty of fluids. Consult a doctor if symptoms worsen.",
        "Migraine": "Avoid stress, rest in a dark quiet room, and take proper medication.",
        "Cold":     "Drink warm water, take proper rest, and stay warm.",
        "Healthy":  "You seem healthy! Maintain a good diet and exercise routine."
    }

    suggestion = suggestions.get(disease_name, "Please consult a doctor for further evaluation.")
    return disease_name, suggestion


# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html', accuracy=model_accuracy)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fever    = int(data.get('fever', 0))
    cough    = int(data.get('cough', 0))
    headache = int(data.get('headache', 0))
    fatigue  = int(data.get('fatigue', 0))
    bodypain = int(data.get('bodypain', 0))

    disease, suggestion = predict_disease(fever, cough, headache, fatigue, bodypain)

    return jsonify({
        'disease': disease,
        'suggestion': suggestion
    })


if __name__ == '__main__':
    app.run(debug=True)
