import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Load cleaned data
df = pd.read_csv("cleaned_drugcomb.csv")
# Create target
df["synergy"] = (df["synergy_zip"] > 10).astype(int)
# Features
features = [
    "css",
    "ic50_row",
    "ic50_col",
    "ri_row",
    "ri_col",
    "css_row",
    "css_col"
]
X = df[features]
y = df["synergy"]
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)
# SHAP analysis
explainer = shap.TreeExplainer(model)
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)
# Summary plot
shap.summary_plot(shap_values[:, :, 1], X_sample, show=False)
plt.savefig("shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("SHAP analysis completed successfully.")