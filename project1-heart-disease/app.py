"""
Heart Disease Prediction — REST API
====================================
Run : python app.py
Test: curl -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"age":55,"sex":1,"cp":0,"trestbps":140,"chol":240,
           "fbs":0,"restecg":1,"thalach":150,"exang":1,
           "oldpeak":1.5,"slope":1,"ca":1,"thal":2}'
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load model once at startup ──
MODEL_PATH = os.path.join("models", "best_model.pkl")
model = None

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        model = joblib.load(MODEL_PATH)
    return model


FEATURES = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]

FEATURE_RANGES = {
    "age":      (1,   120),
    "sex":      (0,   1),
    "cp":       (0,   3),
    "trestbps": (60,  250),
    "chol":     (100, 600),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (50,  250),
    "exang":    (0,   1),
    "oldpeak":  (0.0, 10.0),
    "slope":    (0,   2),
    "ca":       (0,   4),
    "thal":     (0,   3),
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "service"     : "Heart Disease Prediction API",
        "version"     : "1.0",
        "endpoints"   : {
            "POST /predict" : "Predict heart disease risk",
            "GET  /health"  : "Service health check",
            "GET  /features": "Feature descriptions",
        },
        "usage_example": {
            "url"    : "POST /predict",
            "payload": {f: "numeric value" for f in FEATURES}
        }
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        get_model()
        return jsonify({"status": "ok", "model_loaded": True})
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@app.route("/features", methods=["GET"])
def features():
    descriptions = {
        "age"      : "Age in years",
        "sex"      : "Sex: 1 = male, 0 = female",
        "cp"       : "Chest pain type: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic",
        "trestbps" : "Resting blood pressure (mmHg)",
        "chol"     : "Serum cholesterol (mg/dl)",
        "fbs"      : "Fasting blood sugar > 120 mg/dl: 1=true, 0=false",
        "restecg"  : "Resting ECG: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy",
        "thalach"  : "Maximum heart rate achieved (bpm)",
        "exang"    : "Exercise-induced angina: 1=yes, 0=no",
        "oldpeak"  : "ST depression induced by exercise relative to rest",
        "slope"    : "Slope of peak exercise ST segment: 0=upsloping, 1=flat, 2=downsloping",
        "ca"       : "Number of major vessels coloured by fluoroscopy (0–3)",
        "thal"     : "Thalassemia: 0=normal, 1=fixed defect, 2=reversible defect",
    }
    return jsonify({"features": descriptions, "valid_ranges": FEATURE_RANGES})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json()

    # ── Validate all features present ──
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    # ── Validate ranges ──
    range_errors = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = data[feat]
        if not (lo <= val <= hi):
            range_errors.append(f"{feat}={val} out of expected range [{lo}, {hi}]")
    if range_errors:
        return jsonify({"warning": "Some values look unusual", "details": range_errors,
                        "note": "Prediction will still proceed"})

    # ── Predict ──
    try:
        X = np.array([[data[f] for f in FEATURES]])
        clf = get_model()
        prediction  = int(clf.predict(X)[0])
        probability = float(clf.predict_proba(X)[0][1])

        risk_level = (
            "Low"    if probability < 0.35 else
            "Medium" if probability < 0.65 else
            "High"
        )

        return jsonify({
            "prediction"       : prediction,
            "label"            : "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability"      : round(probability, 4),
            "risk_level"       : risk_level,
            "input_features"   : {f: data[f] for f in FEATURES},
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict for multiple patients at once."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    patients = payload.get("patients", [])
    if not patients:
        return jsonify({"error": "Provide a 'patients' list of feature dicts."}), 400

    clf = get_model()
    results = []
    for i, patient in enumerate(patients):
        missing = [f for f in FEATURES if f not in patient]
        if missing:
            results.append({"patient_id": i, "error": f"Missing: {missing}"})
            continue
        X = np.array([[patient[f] for f in FEATURES]])
        pred = int(clf.predict(X)[0])
        prob = float(clf.predict_proba(X)[0][1])
        results.append({
            "patient_id" : i,
            "prediction" : pred,
            "label"      : "Heart Disease" if pred == 1 else "No Heart Disease",
            "probability": round(prob, 4),
            "risk_level" : "Low" if prob < 0.35 else "Medium" if prob < 0.65 else "High",
        })

    return jsonify({"results": results, "total": len(results)})


if __name__ == "__main__":
    print("🚀 Starting Heart Disease Prediction API...")
    print("   Make sure you've run: python src/train.py")
    print("   API will be available at: http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
