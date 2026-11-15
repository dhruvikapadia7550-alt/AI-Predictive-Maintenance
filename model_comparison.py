import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load Dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Remove unnecessary columns
data = data.drop(columns=["UDI", "Product ID"])

# Encode Machine Type
data["Type"] = data["Type"].map({
    "L": 0,
    "M": 1,
    "H": 2
})

# Prepare Features and Target
X = data.drop(columns=[
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

y = data["Machine failure"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define Models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=12,
        random_state=42
    )
}

results = []

# Train and Evaluate Models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    # Generate Confusion Matrix for Random Forest

    if name == "Random Forest":

        cm = confusion_matrix(y_test, predictions)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Failure"],
            columns=["Predicted Normal", "Predicted Failure"]
        )

        cm_df.to_csv("confusion_matrix.csv")

        print("\nRandom Forest Confusion Matrix:")
        print(cm_df)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

# Display Results
results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON RESULTS =====\n")

print(
    results_df.round(4).to_string(index=False)
)

# Save Results
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to model_comparison_results.csv")