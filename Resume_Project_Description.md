# Resume Project Entry
## AI-Based Predictive Maintenance and Equipment Failure Prediction System

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Random Forest, Streamlit, Plotly, Matplotlib, Joblib

- Developed a software-based predictive maintenance application using the AI4I 2020 dataset (10,000 records) to classify potential machine failures from six operating parameters: machine type, air temperature, process temperature, rotational speed, torque, and tool wear.
- Trained and compared Logistic Regression, Decision Tree, and Random Forest classifiers; selected Random Forest based on its test-set recall (72.06%) and F1 score (63.23%).
- Built an interactive Streamlit dashboard to visualize machine failure distribution, parameter analysis, model performance, and a confusion matrix.
- Implemented manual machine-parameter input, model-based failure prediction, estimated class probabilities, rule-based maintenance suggestions, and timestamped CSV prediction history.

**Project scope:** Prototype using historical data and manual inputs; no direct connection to physical machines or sensors. Maintenance thresholds are illustrative prototype rules, not validated industrial limits.
