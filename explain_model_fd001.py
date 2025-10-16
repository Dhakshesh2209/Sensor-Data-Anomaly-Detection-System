import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

# === LOAD DATA ===
df = pd.read_csv("anomaly_predictions_FD001.csv")
feature_cols = [col for col in df.columns if 'sensor_' in col and ('mean' in col or 'std' in col or col.endswith(tuple(str(i) for i in range(1, 22))) )]
X = df[feature_cols]

# === TRAIN MODEL AGAIN FOR SHAP ===
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
model.fit(X)

# === SHAP EXPLAINABILITY ===
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# === VISUALIZE FOR A SINGLE SAMPLE ===
shap.plots.waterfall(shap_values[0])
