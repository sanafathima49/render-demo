from flask import Flask, request, jsonify, render_template, url_for
import joblib
import json
import matplotlib.pyplot as plt
import os
from random import random

print("🔥 Starting Flask application...")
print("👀 app.py is running...")


# Load components with error handling
def safe_load(path, name):
    try:
        print(f"📦 Loading {name}...")
        obj = joblib.load(path)
        print(f"✅ {name} loaded.")
        return obj
    except Exception as e:
        print(f"❌ Error loading {name}: {e}")
        return None

# Initialize Flask app
app = Flask(__name__)

# Load model components
model = safe_load('symptom_classifier_model.pkl', 'model')
vectorizer = safe_load('vectorizer.pkl', 'vectorizer')
label_encoder = safe_load('label_encoder.pkl', 'label_encoder')

# Load metrics.json
try:
    with open('metrics.json') as f:
        metrics = json.load(f)
except Exception as e:
    print("❌ Error loading metrics.json:", e)
    metrics = {}

# Generate a static chart for the dashboard
def generate_chart():
    if not metrics or 'label_distribution' not in metrics:
        print("⚠️ No label_distribution found in metrics. Skipping chart generation.")
        return None

    label_counts = metrics["label_distribution"]
    labels = list(label_counts.keys())
    counts = [int(v) for v in label_counts.values()]  # Ensure int

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title("Diagnosis Label Distribution")
    plt.tight_layout()

    # Ensure 'static' folder exists
    if not os.path.exists('static'):
        os.makedirs('static')

    chart_path = os.path.join("static", "chart.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"✅ Chart saved at: {chart_path}")
    return chart_path

# Create chart at app startup
chart_path = generate_chart()

# ========== ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    if not all([model, vectorizer, label_encoder]):
        return jsonify({'error': 'Model components not loaded'}), 500

    data = request.get_json()
    symptoms = data.get('symptoms', '')

    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    try:
        X = vectorizer.transform([symptoms])
        y_pred = model.predict(X)
        diagnosis = label_encoder.inverse_transform(y_pred)[0]

        return jsonify({'symptoms': symptoms, 'diagnosis': diagnosis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    accuracy = round(metrics.get('accuracy', 0) * 100, 2)
    label_counts = metrics.get('label_distribution', {})
    return render_template(
        "dashboard.html",
        accuracy=accuracy,
        label_counts=label_counts,
        chart_url=url_for('static', filename='chart.png'),
        random=random
    )

# ========== MAIN ==========

if __name__ == '__main__':
    print("🚀 Running on http://127.0.0.1:5000/")
    app.run(debug=True)
