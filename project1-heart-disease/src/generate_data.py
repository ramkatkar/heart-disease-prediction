import numpy as np, pandas as pd
np.random.seed(42)
n = 303

# Use sklearn breast cancer as backbone, then rename to heart disease
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X_raw = bc.data[:303, :13]
y_raw = 1 - bc.target[:303]  # flip so 1=disease

# Scale to realistic heart disease ranges
def scale(arr, lo, hi):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) * (hi - lo) + lo

age       = scale(X_raw[:,0],  29, 77).astype(int)
sex       = (X_raw[:,1] > np.median(X_raw[:,1])).astype(int)
cp        = np.clip((scale(X_raw[:,2], 0, 3.5)).astype(int), 0, 3)
trestbps  = scale(X_raw[:,3],  94, 200).astype(int)
chol      = scale(X_raw[:,4], 126, 564).astype(int)
fbs       = (chol > 260).astype(int)
restecg   = np.clip((scale(X_raw[:,6], 0, 2.5)).astype(int), 0, 2)
thalach   = scale(X_raw[:,7],  71, 202).astype(int)
exang     = (X_raw[:,8] > np.median(X_raw[:,8])).astype(int)
oldpeak   = scale(X_raw[:,9],  0, 6.2).round(1)
slope     = np.clip((scale(X_raw[:,10], 0, 2.5)).astype(int), 0, 2)
ca        = np.clip((scale(X_raw[:,11], 0, 3.5)).astype(int), 0, 3)
thal      = np.clip((scale(X_raw[:,12], 0, 3.5)).astype(int), 0, 3)

df = pd.DataFrame({
    "age":age,"sex":sex,"cp":cp,"trestbps":trestbps,"chol":chol,
    "fbs":fbs,"restecg":restecg,"thalach":thalach,"exang":exang,
    "oldpeak":oldpeak,"slope":slope,"ca":ca,"thal":thal,"target":y_raw
})
df.to_csv("data/heart.csv", index=False)
cnt = df.target.value_counts()
print(f"Generated {len(df)} samples | 0(no disease):{cnt[0]} | 1(disease):{cnt[1]}")
