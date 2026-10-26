import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("cleaned_drugcomb.csv")
df["synergy"] = (df["synergy_zip"] > 10).astype(int)
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
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})
importance = importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance)
plt.figure(figsize=(8,5))
plt.bar(
    importance["Feature"],
    importance["Importance"]
)
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()