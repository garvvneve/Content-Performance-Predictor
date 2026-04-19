import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer


# Load ML-ready dataset

df = pd.read_csv("ml_ready_posts.csv")
print("Dataset loaded")
print("Shape:", df.shape)

TARGET_COL = "performance_binary"

print("\nTarget distribution:")
print(df[TARGET_COL].value_counts())

# Remove junk Excel columns
df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]

# Split X and y

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Only numeric features
X = X.select_dtypes(include=[np.number])

# Save feature names (VERY IMPORTANT)
feature_names = X.columns.tolist()

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Handle missing values

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train and using Random Forest for now

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluation

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("------------------")
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Feature Importance


feature_importance = pd.Series(
    model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save Graphs (visuals section)

os.makedirs("graphs", exist_ok=True)

# Target distribution
plt.figure(figsize=(6,4))
df[TARGET_COL].value_counts().plot(kind="bar")
plt.title("Post Performance Distribution")
plt.xlabel("Performance")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("graphs/target_distribution.png", dpi=300)
plt.close()

# Confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png", dpi=300)
plt.close()

# Feature importance
plt.figure(figsize=(8,5))
feature_importance.head(10).plot(kind="bar")
plt.title("Top Features Influencing Post Performance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("graphs/feature_importance.png", dpi=300)
plt.close()

print("\n Graphs saved in /graphs folder")

# SAVEING MODEL ARTIFACTS

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/content_model.pkl")
joblib.dump(imputer, "model/imputer.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print(" Model artifacts saved for Flask inference")
