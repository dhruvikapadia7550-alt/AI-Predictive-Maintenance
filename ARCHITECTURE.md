# Project Architecture and Workflow
## AI-Based Predictive Maintenance and Equipment Failure Prediction System

## 1. Architecture Overview

The application follows a simple data-processing and machine-learning workflow. A historical predictive-maintenance dataset is used to prepare and train classification models. The selected Random Forest model is saved and then loaded by the Streamlit application to predict machine failure from user-entered operating parameters.

## 2. High-Level Workflow

```text
             AI4I 2020 Dataset
                     |
                     v
             Data Preprocessing
                     |
                     v
          Exploratory Data Analysis
                     |
                     v
          Model Training and Testing
                     |
          +----------+-----------+
          |          |           |
          v          v           v
       Logistic   Decision    Random
      Regression   Tree       Forest
          |          |           |
          +----------+-----------+
                     |
                     v
             Model Evaluation
       (Accuracy, Precision, Recall,
                and F1 Score)
                     |
                     v
          Select Random Forest Model
                     |
                     v
          Save Trained Model (.pkl)
                     |
                     v
             Streamlit Application
                     |
          User enters machine values
                     |
                     v
            Prepare input DataFrame
                     |
                     v
             Load trained model
                     |
                     v
             Predict machine status
                     |
          +----------+-----------+
          |                      |
          v                      v
    Normal prediction      Failure prediction
          |                      |
          +----------+-----------+
                     |
                     v
       Display prediction probabilities
                     |
                     v
       Generate rule-based recommendations
                     |
                     v
         Save prediction to CSV history
                     |
                     v
          Display result in the app
```

## 3. Main Components

### A. Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset containing 10,000 records. The model uses these operating parameters:

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The target is `Machine failure`. Failure-type indicator columns are not used as model input features because they describe failure outcomes and could cause target leakage.

### B. Data Preprocessing — `src/preprocessing.py`

The preprocessing stage prepares the dataset for analysis and model training. The project checks the data, handles the required feature selection and encoding, and prepares the model input features.

### C. Exploratory Data Analysis — `src/eda.py`

The EDA stage examines the dataset and creates visualizations to understand:

- Normal and failed machine records.
- Failure-type distribution.
- Relationships between machine parameters and failure.
- Parameter patterns across failure classes.

### D. Model Training — `src/train_model.py`

The training stage evaluates classification algorithms using a training and test split. The compared models are:

- Logistic Regression
- Decision Tree
- Random Forest

The Random Forest model is saved as `src/maintenance_model.pkl` for use by the application.

### E. Model Comparison — `model_comparison.py`

The comparison stage evaluates the models using:

- Accuracy
- Precision
- Recall
- F1 score

The reported test results are:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.05% | 13.90% | 82.35% | 23.78% |
| Decision Tree | 97.15% | 57.14% | 64.71% | 60.69% |
| Random Forest | 97.15% | 56.32% | 72.06% | 63.23% |

Random Forest was selected because it had the highest recall and F1 score among the evaluated models.

### F. Streamlit Application — `app.py`

The Streamlit application provides the user interface. Its main functions include:

- Dashboard: displays dataset summaries, charts, model comparison, confusion matrix, and prediction history.
- Prediction: accepts six machine parameters and displays the model prediction.
- Prediction confidence: displays the model-estimated probabilities for normal and failure classes.
- Maintenance recommendations: applies predefined parameter checks and adds a recommendation when the model predicts failure.
- Prediction history: saves the entered parameters, timestamp, and prediction to `prediction_history.csv`.

## 4. Prediction Workflow

When a user opens the Prediction page and enters machine parameters:

1. The application reads the selected machine type and numeric operating values.
2. The machine type is encoded using the same mapping used during training: L = 0, M = 1, H = 2.
3. The six input features are assembled into a Pandas DataFrame using the model's expected feature names.
4. The saved Random Forest model is loaded from `src/maintenance_model.pkl`.
5. `predict()` produces the predicted class.
6. `predict_proba()` provides model-estimated class probabilities.
7. The application displays the predicted status and probability values.
8. Predefined rules check tool wear, torque, air temperature, and process temperature to generate maintenance suggestions.
9. The input values, timestamp, and predicted label are appended to `prediction_history.csv`.

## 5. Confusion Matrix

The Random Forest confusion matrix from the project is:

| Actual / Predicted | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | 1,894 | 38 |
| Actual Failure | 19 | 49 |

- True Negative (TN): 1,894
- False Positive (FP): 38
- False Negative (FN): 19
- True Positive (TP): 49

## 6. Important Design Notes and Limitations

- The current prototype accepts machine parameters manually through the application; it is not connected to physical machines or sensors.
- The model is trained using historical dataset records.
- Prediction probabilities are model estimates, not guaranteed real-world failure probabilities.
- Maintenance recommendations are based on predefined prototype rules. Their thresholds are not validated industrial operating limits.
- The application should be tested and validated with representative plant-specific data before operational use.

## 7. Future Architecture Extension

A future version could receive machine readings through an industrial IoT gateway, validate and store incoming data, and pass the prepared readings to a deployed prediction service. This is a proposed enhancement only; it is not part of the current prototype.
