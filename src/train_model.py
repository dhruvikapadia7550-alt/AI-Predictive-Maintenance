import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/ai4i2020.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)


# ==========================================
# ==========================================
# STEP 2: SELECT FEATURES AND TARGET
# ==========================================

# Remove unnecessary identification columns
data = data.drop(["UDI", "Product ID"], axis=1)

# Convert categorical Type column into numerical values
data["Type"] = data["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Select input features
X = data.drop(
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"],
    axis=1
)

# Select target variable
y = data["Machine failure"]

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTarget Variable:")
print("Machine failure")

print("\nCategorical Conversion Completed!")

# ==========================================
# STEP 3: SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ==========================================
# STEP 4: CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    max_depth=12
)


# ==========================================
# STEP 5: TRAIN MODEL
# ==========================================

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")


# ==========================================
# STEP 6: MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# STEP 7: EVALUATE MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# STEP 8: FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(feature_importance)


# ==========================================
# STEP 9: SAVE TRAINED MODEL
# ==========================================

joblib.dump(model, "src/maintenance_model.pkl")

print("\nModel Saved Successfully!")