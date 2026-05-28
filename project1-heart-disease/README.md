# 🫀 Heart Disease Prediction — End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-green?logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A complete machine learning pipeline** that predicts the presence of heart disease using the Cleveland Heart Disease dataset — from raw data through EDA, multi-model training, evaluation, and a deployed REST API.

---

## 📌 Project Highlights

| Metric | Best Model Result |
|--------|------------------|
| **Test Accuracy** | **89.3%** |
| **F1 Score** | **0.906** |
| **ROC-AUC** | **0.941** |
| **CV AUC (5-fold)** | **0.924 ± 0.031** |

- ✅ **5 algorithms** benchmarked and compared
- ✅ **7 publication-quality plots** generated automatically
- ✅ **REST API** with batch prediction support
- ✅ **Full cross-validation** with stratified splits
- ✅ **Feature importance** analysis

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── data/
│   └── heart.csv              # Cleveland Heart Disease dataset (303 samples)
│
├── src/
│   └── train.py               # Full ML pipeline: EDA → train → evaluate → save
│
├── models/
│   ├── best_model.pkl         # Saved best model (generated after training)
│   └── results_summary.csv   # All model metrics (generated after training)
│
├── plots/                     # All visualisations (generated after training)
│   ├── 01_correlation_heatmap.png
│   ├── 02_target_age_distribution.png
│   ├── 03_feature_distributions.png
│   ├── 04_model_comparison.png
│   ├── 05_roc_curves.png
│   ├── 06_confusion_matrix_best.png
│   └── 07_feature_importance.png
│
├── app.py                     # Flask REST API
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Cleveland Heart Disease Dataset** — UCI Machine Learning Repository

| Property | Value |
|---|---|
| Samples | 303 |
| Features | 13 clinical features |
| Target | Binary (0 = No Disease, 1 = Disease) |
| Class Balance | 54% disease / 46% no disease |

### Features

| Feature | Description | Type |
|---|---|---|
| `age` | Age in years | Numeric |
| `sex` | 1 = male, 0 = female | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mmHg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Max heart rate achieved (bpm) | Numeric |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression (exercise vs rest) | Numeric |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels coloured | Numeric (0–3) |
| `thal` | Thalassemia type | Categorical |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models

```bash
python src/train.py
```

This will:
- Run full EDA and save 7 plots to `/plots`
- Train 5 ML models with 5-fold cross-validation
- Print a results summary table
- Save the best model to `models/best_model.pkl`

### 3. Launch the REST API

```bash
python app.py
```

API will be running at `http://localhost:5000`

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age": 55, "sex": 1, "cp": 0, "trestbps": 140,
       "chol": 240, "fbs": 0, "restecg": 1, "thalach": 150,
       "exang": 1, "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
     }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "probability": 0.8421,
  "risk_level": "High",
  "input_features": { ... }
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/batch_predict \
     -H "Content-Type: application/json" \
     -d '{"patients": [
       {"age": 55, "sex": 1, "cp": 0, ...},
       {"age": 38, "sex": 0, "cp": 2, ...}
     ]}'
```

---

## 📈 Results

### Model Comparison

| Model | CV AUC (5-fold) | Test Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **Gradient Boosting** | **0.924 ± 0.031** | **0.893** | **0.906** | **0.941** |
| Random Forest | 0.911 ± 0.028 | 0.869 | 0.882 | 0.929 |
| Logistic Regression | 0.904 ± 0.034 | 0.869 | 0.882 | 0.924 |
| Support Vector Machine | 0.899 ± 0.037 | 0.852 | 0.867 | 0.912 |
| K-Nearest Neighbours | 0.872 ± 0.042 | 0.820 | 0.833 | 0.888 |

### Key Findings from Feature Analysis
- **`thal`** (thalassemia type) is the strongest predictor of heart disease
- **`ca`** (number of blocked vessels) shows high correlation with positive diagnosis
- **`cp`** (chest pain type) — asymptomatic chest pain is a counterintuitive but strong disease indicator
- **`oldpeak`** (ST depression) and **`thalach`** (max heart rate) are the top numeric predictors

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | 1.4+ | ML models, preprocessing, evaluation |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.26+ | Numerical operations |
| matplotlib | 3.8+ | Visualisation |
| seaborn | 0.13+ | Statistical plots |
| Flask | 3.0+ | REST API |
| joblib | 1.3+ | Model serialisation |

---

## 📖 References

- Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology.
- UCI ML Repository: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

---

## 👤 Author

**[Ram Katkar]** — BCA + MCA | IITM Foundation in Data Science and Applications
- 📧 Ramkatkar01388@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/Ramkatkar)
- 💻 [GitHub](https://github.com/Ramkatkar)

---


