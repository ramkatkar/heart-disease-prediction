"""
Heart Disease Prediction — End-to-End ML Pipeline
==================================================
Dataset  : Cleveland Heart Disease (UCI ML Repository)
Task     : Binary Classification (heart disease: yes/no)
Author   : [Ram Katkar]
Date     : 2026
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ─────────────────────────────────────────────
# SECTION 1: LOAD & INSPECT DATA
# ─────────────────────────────────────────────
FEATURE_NAMES = {
    "age"      : "Age (years)",
    "sex"      : "Sex (1=male, 0=female)",
    "cp"       : "Chest Pain Type (0-3)",
    "trestbps" : "Resting Blood Pressure (mmHg)",
    "chol"     : "Serum Cholesterol (mg/dl)",
    "fbs"      : "Fasting Blood Sugar > 120 mg/dl",
    "restecg"  : "Resting ECG Results (0-2)",
    "thalach"  : "Max Heart Rate Achieved (bpm)",
    "exang"    : "Exercise-Induced Angina (1=yes)",
    "oldpeak"  : "ST Depression (exercise vs. rest)",
    "slope"    : "Slope of Peak Exercise ST Segment",
    "ca"       : "Number of Major Vessels (0-3)",
    "thal"     : "Thal (0=normal, 1=fixed, 2=reversible)",
    "target"   : "Heart Disease (1=yes, 0=no)",
}

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📊 Class distribution:\n{df['target'].value_counts().to_string()}")
    return df


def eda(df):
    """Exploratory Data Analysis — saves plots to /plots."""
    print("\n─── EDA ──────────────────────────────────────────")

    # 1. Correlation heatmap
    plt.figure(figsize=(12, 9))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Matrix", fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("plots/01_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/01_correlation_heatmap.png")

    # 2. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["target"].value_counts()
    axes[0].pie(counts, labels=["Heart Disease", "No Disease"],
                autopct="%1.1f%%", colors=["#E74C3C", "#2ECC71"],
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("Target Distribution", fontweight="bold")

    axes[1].hist(df[df["target"]==1]["age"], bins=15, alpha=0.7, label="Disease", color="#E74C3C")
    axes[1].hist(df[df["target"]==0]["age"], bins=15, alpha=0.7, label="No Disease", color="#2ECC71")
    axes[1].set_xlabel("Age"); axes[1].set_ylabel("Count")
    axes[1].set_title("Age Distribution by Class", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("plots/02_target_age_distribution.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/02_target_age_distribution.png")

    # 3. Feature distributions by target
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    palette = {0: "#2ECC71", 1: "#E74C3C"}
    for i, col in enumerate(numeric_cols):
        for cls, color in palette.items():
            label = "No Disease" if cls == 0 else "Heart Disease"
            axes[i].hist(df[df["target"] == cls][col], bins=20, alpha=0.65,
                         label=label, color=color)
        axes[i].set_title(FEATURE_NAMES.get(col, col), fontweight="bold", fontsize=10)
        axes[i].legend(fontsize=8)
    axes[-1].axis("off")
    plt.suptitle("Numeric Feature Distributions by Target Class", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/03_feature_distributions.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/03_feature_distributions.png")

    print(f"\n📋 Missing values: {df.isnull().sum().sum()}")
    print(f"📋 Duplicates   : {df.duplicated().sum()}")
    return df


# ─────────────────────────────────────────────
# SECTION 2: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    """Clean, encode, and split data."""
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SECTION 3: TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────

def build_models():
    """Return dict of sklearn Pipelines (scaler + classifier)."""
    return {
        "Logistic Regression" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, random_state=42, C=1.0))
        ]),
        "Random Forest" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, random_state=42,
                                              max_depth=6, min_samples_leaf=2))
        ]),
        "Gradient Boosting" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                   max_depth=4, random_state=42))
        ]),
        "Support Vector Machine" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(probability=True, kernel="rbf", C=1.5, random_state=42))
        ]),
        "K-Nearest Neighbours" : Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=7))
        ]),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, cross-validate, and return full metrics dict."""
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model"         : name,
        "CV ROC-AUC"    : f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        "Test Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision"     : round(precision_score(y_test, y_pred), 4),
        "Recall"        : round(recall_score(y_test, y_pred), 4),
        "F1 Score"      : round(f1_score(y_test, y_pred), 4),
        "ROC-AUC"       : round(roc_auc_score(y_test, y_proba), 4),
        "_model"        : model,
        "_y_pred"       : y_pred,
        "_y_proba"      : y_proba,
    }


def train_all(models, X_train, X_test, y_train, y_test):
    print("\n─── Training Models ──────────────────────────────")
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        print(f"  ✅ {name:<25} | F1={res['F1 Score']:.4f} | AUC={res['ROC-AUC']:.4f}")
        results.append(res)
    return results


# ─────────────────────────────────────────────
# SECTION 4: VISUALISE RESULTS
# ─────────────────────────────────────────────

def plot_results(results, X_test, y_test):
    model_names   = [r["Model"] for r in results]
    f1_scores     = [r["F1 Score"] for r in results]
    auc_scores    = [r["ROC-AUC"] for r in results]
    acc_scores    = [r["Test Accuracy"] for r in results]

    # ── Metrics comparison bar chart ──
    x = np.arange(len(model_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, acc_scores, width, label="Accuracy", color="#3498DB", alpha=0.85)
    ax.bar(x,          f1_scores,  width, label="F1 Score",  color="#E74C3C", alpha=0.85)
    ax.bar(x + width,  auc_scores, width, label="ROC-AUC",   color="#2ECC71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Accuracy / F1 / ROC-AUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7.5, color="dimgray")
    plt.tight_layout()
    plt.savefig("plots/04_model_comparison.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/04_model_comparison.png")

    # ── ROC Curves ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#E67E22"]
    for r, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, r["_y_proba"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})")
    ax.plot([0,1],[0,1],"k--", lw=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/05_roc_curves.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/05_roc_curves.png")

    # ── Confusion matrix for best model ──
    best = max(results, key=lambda r: r["F1 Score"])
    cm   = confusion_matrix(y_test, best["_y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                linewidths=0.5, cbar_kws={"shrink": 0.7})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {best['Model']}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/06_confusion_matrix_best.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/06_confusion_matrix_best.png")

    return best


def plot_feature_importance(best, X_train):
    """Feature importance for tree-based best model."""
    clf = best["_model"].named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        print("  ℹ️  Best model has no feature_importances_ — skipping importance plot")
        return

    importances = clf.feature_importances_
    feat_df = pd.DataFrame({
        "Feature"   : X_train.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#E74C3C" if v > feat_df["Importance"].median() else "#3498DB"
              for v in feat_df["Importance"]]
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, alpha=0.85)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title(f"Feature Importances — {best['Model']}", fontsize=13, fontweight="bold")
    high = mpatches.Patch(color="#E74C3C", label="Above median")
    low  = mpatches.Patch(color="#3498DB", label="Below median")
    ax.legend(handles=[high, low], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/07_feature_importance.png", dpi=150)
    plt.close()
    print("  📈 Saved: plots/07_feature_importance.png")


# ─────────────────────────────────────────────
# SECTION 5: SAVE RESULTS & BEST MODEL
# ─────────────────────────────────────────────

def save_results(results):
    display_cols = ["Model", "CV ROC-AUC", "Test Accuracy",
                    "Precision", "Recall", "F1 Score", "ROC-AUC"]
    df_res = pd.DataFrame([{k: r[k] for k in display_cols} for r in results])
    df_res = df_res.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df_res.to_csv("models/results_summary.csv", index=False)
    print("\n─── Results Summary ──────────────────────────────")
    print(df_res.to_string(index=False))
    return df_res


def save_best_model(best):
    path = "models/best_model.pkl"
    joblib.dump(best["_model"], path)
    print(f"\n💾 Best model saved → {path}")
    print(f"   Model : {best['Model']}")
    print(f"   F1    : {best['F1 Score']}")
    print(f"   AUC   : {best['ROC-AUC']}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  HEART DISEASE PREDICTION — ML PIPELINE")
    print("=" * 55)

    df         = load_data()
    df         = eda(df)
    X_tr, X_te, y_tr, y_te = preprocess(df)
    models     = build_models()
    results    = train_all(models, X_tr, X_te, y_tr, y_te)

    print("\n─── Generating Plots ─────────────────────────────")
    best       = plot_results(results, X_te, y_te)
    plot_feature_importance(best, X_tr)

    save_results(results)
    save_best_model(best)

    print("\n─── Classification Report (Best Model) ───────────")
    print(classification_report(y_te, best["_y_pred"],
                                target_names=["No Disease", "Heart Disease"]))
    print("=" * 55)
    print("✅ Pipeline complete. Check /plots and /models.")
    print("=" * 55)
