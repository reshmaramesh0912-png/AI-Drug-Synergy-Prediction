import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
df = pd.read_csv("cleaned_drugcomb.csv")
# 1=synergistic,0=non-synergistic
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
#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#model training
model = RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)
model.fit(X_train, y_train)
print("Model training completed")
#predict
prob=model.predict_proba(X_test)[:, 1]
y_pred = (prob > 0.35).astype(int)
print("\nAccuracy")
print(accuracy_score(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
