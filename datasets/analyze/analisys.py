import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(r"C:\Users\dmilos\git\gdbf\datasets\IRISC_P_dv3_R050_L54_N1296\dataset.csv")  

# =========================
# 2. SELECT TARGET
# =========================
target = "L0"   # <-- change if needed

X = df.drop(columns=[c for c in df.columns if c.startswith("L")])
y = df[target]

# =========================
# 3. TRAIN MODEL
# =========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# =========================
# 4. PERMUTATION IMPORTANCE
# =========================
result = permutation_importance(model, X, y, n_repeats=10, random_state=42)

importance = pd.Series(result.importances_mean, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\n=== TOP 20 FEATURES ===")
print(importance.head(20))

# =========================
# 5. GROUP BY FEATURE TYPE
# =========================
feature_type_importance = {}

for col, imp in importance.items():
    # extract feature name (energy, unsat_ratio, etc.)
    feature_type = col.split("_", 1)[1]
    feature_type_importance.setdefault(feature_type, 0)
    feature_type_importance[feature_type] += imp

feature_type_importance = pd.Series(feature_type_importance).sort_values(ascending=False)

print("\n=== FEATURE TYPE IMPORTANCE ===")
print(feature_type_importance)

# =========================
# 6. GROUP BY STAGE (S0–S23)
# =========================
stage_importance = {}

for col, imp in importance.items():
    match = re.match(r"S(\d+)_", col)
    if match:
        stage = int(match.group(1))
        stage_importance.setdefault(stage, 0)
        stage_importance[stage] += imp

stage_importance = pd.Series(stage_importance).sort_values(ascending=False)

print("\n=== STAGE IMPORTANCE ===")
print(stage_importance)

# =========================
# 7. PLOTS
# =========================
plt.figure(figsize=(10, 5))
importance.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Feature Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
feature_type_importance.sort_values().plot(kind="barh")
plt.title("Feature Type Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
stage_importance.sort_index().plot(kind="bar")
plt.title("Stage Importance (S0–S23)")
plt.tight_layout()
plt.show()

# =========================
# 8. OPTIONAL: SAVE RESULTS
# =========================
importance.to_csv("feature_importance.csv")
feature_type_importance.to_csv("feature_type_importance.csv")
stage_importance.to_csv("stage_importance.csv")

print("\nResults saved to CSV files.")