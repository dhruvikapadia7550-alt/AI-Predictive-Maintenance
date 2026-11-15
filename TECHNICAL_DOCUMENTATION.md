# Technical Documentation
## AI-Based Predictive Maintenance and Equipment Failure Prediction System

> This document describes the implementation details established during project development. Where exact source-code details may vary, verify them against the current project files before presenting them as code-level facts.

## 1. Technology Stack

| Technology | Role in the project |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning, transforming, and organizing tabular data |
| NumPy | Numerical operations |
| Scikit-learn | Classification models, train/test split, and evaluation metrics |
| Matplotlib | Static data visualizations |
| Plotly | Interactive charts in the dashboard |
| Streamlit | Web application interface |
| Joblib | Saving and loading the trained model |

## 2. Dataset and Feature Preparation

The project uses the AI4I 2020 Predictive Maintenance Dataset with 10,000 records.

The six model input features are:

| Feature | Meaning |
|---|---|
| Type | Machine product/type category (L, M, or H) |
| Air temperature [K] | Air temperature around the machine |
| Process temperature [K] | Machine process temperature |
| Rotational speed [rpm] | Rotational speed |
| Torque [Nm] | Operating torque |
| Tool wear [min] | Accumulated tool wear |

The target column is `Machine failure`.

The failure-type indicator columns (TWF, HDF, PWF, OSF, and RNF) are excluded from model inputs. These columns describe failure outcomes and should not be supplied as predictors when the goal is to predict whether a failure occurs.

The project preprocessing also removes identifier columns such as `UDI` and `Product ID` from the model feature set and encodes machine type as numeric values:
- L = 0
- M = 1
- H = 2

The same feature order and encoding must be used at both training time and prediction time.

## 3. Model Training

The project evaluates three classification algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest

The documented Random Forest configuration is:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    max_depth=12
)
```

The documented train/test split uses:
- Test size: 20%
- Random state: 42
- Stratification by the target label

The trained Random Forest model is saved as:

```text
src/maintenance_model.pkl
```

The Streamlit application loads this saved model to perform predictions. If the training pipeline or feature encoding is changed, retrain the model and confirm that the application uses the matching feature order and preprocessing.

## 4. Class Imbalance

The dataset contains:
- Normal records: 9,661
- Failure records: 339
- Failure rate: 3.39%

This is an imbalanced classification problem. A model can achieve high accuracy by predicting the majority class while missing some failures. Therefore, accuracy should not be considered alone.

The project uses `class_weight="balanced"` in the Random Forest configuration to give additional weight to the minority class during training. Recall and F1 score are also considered when comparing models.

## 5. Model Evaluation

The reported model comparison is:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.05% | 13.90% | 82.35% | 23.78% |
| Decision Tree | 97.15% | 57.14% | 64.71% | 60.69% |
| Random Forest | 97.15% | 56.32% | 72.06% | 63.23% |

Random Forest was selected because it has the highest recall and F1 score among these evaluated models.

### Metric definitions

- **Accuracy:** Fraction of all predictions that are correct.
- **Precision:** Of the records predicted as failures, the fraction that are actual failures.
- **Recall:** Of all actual failures, the fraction detected by the model.
- **F1 score:** Harmonic mean of precision and recall.

### Random Forest confusion matrix

| Actual / Predicted | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | 1,894 | 38 |
| Actual Failure | 19 | 49 |

- True Negative (TN): 1,894
- False Positive (FP): 38
- False Negative (FN): 19
- True Positive (TP): 49

For predictive maintenance, false negatives matter because they represent failures that the model did not identify. This is one reason recall is an important evaluation metric.

## 6. Streamlit Application (`app.py`)

The application provides the user interface and the following main functions.

### Dashboard

The dashboard displays:
- Total records, normal records, failed records, and failure rate.
- Machine failure distribution.
- Failure-type analysis.
- Machine parameter analysis.
- Model comparison.
- Random Forest confusion matrix.
- Prediction history and CSV download.

### Prediction page

The user supplies six values:
1. Machine type
2. Air temperature
3. Process temperature
4. Rotational speed
5. Torque
6. Tool wear

The application encodes the selected machine type and assembles the six values into a Pandas DataFrame with the feature names expected by the model.

The model's `predict()` method returns the predicted class. The `predict_proba()` method returns class probability estimates, which the application displays as normal and failure probabilities.

These probability values are model estimates. They should not be described as guaranteed real-world failure probabilities.

### Maintenance recommendations

The application uses predefined checks for parameters including:
- Tool wear greater than 200 minutes.
- Torque greater than 60 Nm.
- Air temperature greater than 305 K.
- Process temperature greater than 315 K.
- A model prediction of machine failure.

These are prototype rules used to produce suggestions. They are not validated manufacturer limits or certified industrial safety thresholds.

### Prediction history

The application appends prediction records to `prediction_history.csv`. The documented fields include:
- Timestamp
- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Prediction

The dashboard provides a download option for this CSV history.

## 7. Supporting Python Files

### `src/preprocessing.py`
Responsible for preparing the dataset for downstream analysis and model development. Consult the current source file for the exact sequence of operations.

### `src/eda.py`
Responsible for exploratory analysis and visualizations of the dataset. Consult the current source file for the exact charts and transformations.

### `src/train_model.py`
Responsible for training the selected machine-learning model and saving the model artifact. The documented saved model is `src/maintenance_model.pkl`.

### `model_comparison.py`
Evaluates Logistic Regression, Decision Tree, and Random Forest and produces the model comparison results used in the dashboard.

## 8. Running the Application on Windows

From the project root folder in the VS Code terminal, use the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Alternatively, activate the virtual environment first:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

If PowerShell blocks activation, the execution policy can be changed for the current terminal process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Do not disable Windows security features to run the project.

## 9. Known Limitations

- The prototype uses historical dataset records and manual input; it does not read live data from industrial equipment.
- It has not been validated against a specific plant's machinery or operating limits.
- Maintenance suggestions are based on simple predefined rules.
- Predicted probabilities are model estimates and may not be calibrated to real-world failure rates.
- Before operational use, the model requires validation with representative plant-specific data, monitoring, and appropriate engineering review.

## 10. Suggested Future Improvements

- Integrate data through an industrial IoT gateway.
- Validate input ranges and handle missing or out-of-range values.
- Add model versioning and prediction audit logs.
- Monitor model performance as new labeled data becomes available.
- Evaluate probability calibration and decision thresholds with plant-specific validation data.
- Add maintenance scheduling after consultation with qualified maintenance personnel.
