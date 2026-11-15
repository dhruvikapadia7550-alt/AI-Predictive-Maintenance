# AI-Based Predictive Maintenance and Equipment Failure Prediction System

## 1. Project Overview

The AI-Based Predictive Maintenance System is a machine learning application designed to predict potential industrial machine failures using machine operating parameters.

The system analyzes air temperature, process temperature, rotational speed, torque, and tool wear to identify potential failure risks. It uses a Random Forest classification model and provides an interactive dashboard for machine performance analysis, failure prediction, prediction confidence, and maintenance recommendations.

## 2. Project Objectives

- Predict potential machine failures using machine learning.
- Analyze industrial machine operating conditions.
- Compare different machine learning algorithms.
- Visualize machine failure patterns and performance metrics.
- Provide maintenance recommendations based on machine parameters.
- Maintain a history of machine predictions.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| Streamlit | Web application |
| Joblib | Model persistence |

## 4. Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

- Total records: 10,000
- Normal machine records: 9,661
- Failure records: 339
- Failure rate: 3.39%

### Machine Parameters

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

The target variable is `Machine Failure`.

## 5. Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.05% | 13.90% | 82.35% | 23.78% |
| Decision Tree | 97.15% | 57.14% | 64.71% | 60.69% |
| Random Forest | 97.15% | 56.32% | 72.06% | 63.23% |

Random Forest was selected based on its recall and F1 score among the evaluated models.

## 6. Main Features

### Interactive Dashboard

- Machine performance overview
- Machine failure distribution
- Failure type analysis
- Machine parameter analysis
- Machine learning model comparison
- Random Forest confusion matrix

### Machine Failure Prediction

Users can enter machine operating parameters and receive a predicted machine status.

### Prediction Confidence

Displays model-estimated failure and normal probabilities. These are model estimates, not guarantees of real-world failure.

### Maintenance Recommendations

Provides rule-based maintenance suggestions based on machine operating parameters and model predictions. The thresholds are prototype rules and are not validated industrial operating limits.

### Prediction History

Stores prediction records with timestamps and machine parameters in a CSV file. The application also provides a CSV download option.

## 7. Project Structure

```text
AI_Predictive_Maintenance/
├── app.py
├── model_comparison.py
├── dataset/
│   └── ai4i2020.csv
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   └── maintenance_model.pkl
├── prediction_history.csv
├── model_comparison_results.csv
├── confusion_matrix.csv
└── README.md
```

The structure above lists the main project files discussed during development. Your local project may contain additional files and folders.

## 8. How to Run the Project

### Step 1: Open the project folder

Open `AI_Predictive_Maintenance` in VS Code.

### Step 2: Create a virtual environment (if one does not already exist)

```powershell
python -m venv venv
```

### Step 3: Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### Step 4: Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib plotly streamlit joblib
```

### Step 5: Start the application

```powershell
python -m streamlit run app.py
```

The Streamlit application should open in your browser. If you already have the project's virtual environment configured, activate it and run the application from the project root folder.

## 9. Future Enhancements

- Real-time machine data integration through an IoT gateway.
- Automated maintenance scheduling.
- Machine-specific failure prediction.
- Advanced failure diagnosis.
- Integration with industrial monitoring systems.

## 10. Project Limitations

- The current prototype uses a historical dataset and manual machine parameter inputs.
- It does not directly connect to physical industrial machines or sensors.
- Maintenance recommendations are based on predefined rules.
- Model probabilities are estimates and are not guarantees of actual machine failure.
- The prototype should be validated with representative industrial data before operational deployment.

## 11. Conclusion

This project demonstrates how machine learning can support predictive maintenance by analyzing industrial machine operating conditions and identifying potential failure risks.

It combines data analysis, machine learning, interactive visualization, and rule-based maintenance recommendations in a single software application.
