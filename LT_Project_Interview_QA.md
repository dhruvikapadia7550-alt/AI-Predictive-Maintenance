# L&T Heavy Engineering – Project-Based Technical Interview Questions and Answers

## Part A: Project Introduction

### 1. Explain your project.
**Answer:** I developed an AI-Based Predictive Maintenance and Equipment Failure Prediction System using Python and machine learning. It analyzes six machine operating parameters—machine type, air temperature, process temperature, rotational speed, torque, and tool wear—to predict whether a machine may fail. I compared Logistic Regression, Decision Tree, and Random Forest models, selected Random Forest based on its recall and F1 score, and built a Streamlit dashboard for predictions, visualizations, recommendations, and prediction history.

### 2. Why did you choose this project?
**Answer:** Predictive maintenance is relevant to industrial operations because unexpected equipment failure can cause downtime, repair costs, and production disruption. I wanted to apply software and machine learning concepts to an industrial engineering problem without requiring physical hardware for the prototype.

### 3. What problem does your project solve?
**Answer:** It provides an early risk indication based on machine operating parameters. A user can enter the current parameter values and receive a model prediction that can support further inspection. It is a decision-support prototype, not a replacement for engineering inspection.

### 4. Is your project connected to a real machine?
**Answer:** No. The current prototype uses a historical dataset and manual parameter inputs through the Streamlit application. Connecting live machine data through an industrial IoT gateway is a possible future enhancement.

## Part B: Dataset and Data Preparation

### 5. Which dataset did you use?
**Answer:** I used the AI4I 2020 Predictive Maintenance Dataset. It contains 10,000 records. In the project data, 9,661 records are normal and 339 records indicate machine failure.

### 6. Which features are used by your model?
**Answer:** Machine type, air temperature, process temperature, rotational speed, torque, and tool wear. The target variable is `Machine failure`.

### 7. Why did you remove UDI and Product ID?
**Answer:** They are identifiers rather than meaningful operating measurements for the prediction task. Keeping them could allow the model to learn patterns tied to record or product identifiers instead of general machine behaviour.

### 8. Why did you exclude TWF, HDF, PWF, OSF, and RNF?
**Answer:** These columns identify failure modes. They describe failure outcomes and are not available as ordinary input measurements when predicting whether a failure will occur. Including them could cause target leakage and make evaluation misleading.

### 9. How did you encode machine type?
**Answer:** I converted the categories to numeric values: L = 0, M = 1, and H = 2. I use the same mapping and feature order when preparing prediction inputs.

### 10. What is class imbalance in your dataset?
**Answer:** Class imbalance means that one target class has many more records than another. Here, there are 9,661 normal records and 339 failure records. A model could achieve high accuracy by mostly predicting the normal class, so I also considered precision, recall, and F1 score.

## Part C: Machine Learning

### 11. Which algorithms did you compare?
**Answer:** Logistic Regression, Decision Tree, and Random Forest.

### 12. Why did you select Random Forest?
**Answer:** In my evaluation, Random Forest achieved 97.15% accuracy, 72.06% recall, and a 63.23% F1 score. It had the highest recall and F1 score among the three models, which made it the preferred model for this prototype.

### 13. What is Random Forest?
**Answer:** Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions. For classification, the trees vote for a class, and the most-voted class becomes the final prediction. Combining trees can make predictions more robust than relying on a single decision tree.

### 14. What is a Decision Tree?
**Answer:** A Decision Tree is a supervised learning algorithm that makes predictions by applying a sequence of feature-based rules. Each split separates the data according to a feature and threshold, and the final leaf provides the predicted class.

### 15. What is Logistic Regression?
**Answer:** Logistic Regression is a supervised classification algorithm that estimates the probability of a class using a logistic function. Although its name includes regression, it is commonly used for classification tasks.

### 16. What is supervised learning?
**Answer:** Supervised learning is a machine learning approach in which a model learns from examples that contain input features and known target labels. In this project, machine parameters are the inputs and the machine-failure label is the target.

### 17. What is train-test splitting?
**Answer:** Train-test splitting separates the available labeled data into a training set and a test set. The model learns from the training data, and the test set is used to evaluate how it performs on records it did not train on. My project uses an 80/20 split, `random_state=42`, and stratification by the target.

### 18. Why use stratification?
**Answer:** Stratification preserves approximately the same proportion of normal and failure records in both the training and test sets. This is useful when the target classes are imbalanced.

### 19. What does `class_weight="balanced"` do?
**Answer:** It adjusts class weights inversely according to class frequencies, giving more weight to the minority class during training. In this project, that helps the model pay more attention to failure records.

### 20. What is overfitting?
**Answer:** Overfitting occurs when a model learns the training data too closely, including noise or accidental patterns, and performs less effectively on unseen data. A separate test set and appropriate model complexity help assess and reduce this risk.

## Part D: Evaluation Metrics

### 21. What is accuracy?
**Answer:** Accuracy is the proportion of all predictions that are correct. It is calculated as `(TP + TN) / (TP + TN + FP + FN)`.

### 22. What is precision?
**Answer:** Precision is the proportion of predicted failures that are actually failures. It is calculated as `TP / (TP + FP)`.

### 23. What is recall?
**Answer:** Recall is the proportion of actual failures that the model correctly identifies. It is calculated as `TP / (TP + FN)`. In predictive maintenance, recall is important because a false negative means an actual failure was missed by the model.

### 24. What is the F1 score?
**Answer:** The F1 score is the harmonic mean of precision and recall. It balances the two metrics and is useful when the classes are imbalanced.

### 25. Explain your confusion matrix.
**Answer:** The Random Forest confusion matrix is:

| Actual / Predicted | Predicted Normal | Predicted Failure |
|---|---:|---:|
| Actual Normal | 1,894 | 38 |
| Actual Failure | 19 | 49 |

There are 1,894 true negatives, 38 false positives, 19 false negatives, and 49 true positives. The 19 false negatives represent actual failure records that the model predicted as normal.

### 26. Why is accuracy alone not enough?
**Answer:** Because the dataset is imbalanced. A model could predict most records as normal and still have high accuracy while missing many failures. Recall, precision, F1 score, and the confusion matrix give a more complete view.

## Part E: Python and Application

### 27. What is Pandas used for?
**Answer:** Pandas is used to load and manipulate tabular data, select columns, prepare features, and work with CSV files and DataFrames.

### 28. What is a DataFrame?
**Answer:** A DataFrame is a two-dimensional labeled data structure in Pandas, similar to a table with rows and columns.

### 29. What is Scikit-learn?
**Answer:** Scikit-learn is a Python machine learning library. I used it for classification algorithms, splitting data, and evaluating model performance.

### 30. Why did you use Streamlit?
**Answer:** Streamlit allows me to build an interactive Python web application without developing a separate frontend and backend. I used it for the dashboard, machine input form, prediction results, charts, and CSV download.

### 31. What is Joblib used for?
**Answer:** Joblib is used to save and load Python objects, including the trained machine learning model. I save the Random Forest model and load it in the Streamlit application for predictions.

### 32. What is the difference between `predict()` and `predict_proba()`?
**Answer:** `predict()` returns the predicted class label. `predict_proba()` returns the model's estimated probability for each class. These probabilities are model estimates and are not automatically guaranteed to be calibrated real-world failure probabilities.

### 33. How does your application save prediction history?
**Answer:** After a prediction, the application appends the timestamp, machine type, entered machine parameters, and predicted label to `prediction_history.csv`. The dashboard displays the history and provides a CSV download option.

### 34. How are maintenance recommendations generated?
**Answer:** The application uses predefined checks for parameters such as tool wear, torque, air temperature, and process temperature. It also recommends inspection when the model predicts failure. These are prototype rules, not validated manufacturer limits or industrial safety thresholds.

## Part F: Industrial Understanding and Limitations

### 35. What is predictive maintenance?
**Answer:** Predictive maintenance uses data about equipment condition and operation to estimate when maintenance may be needed. Its purpose is to help plan inspection or maintenance before an unexpected breakdown occurs.

### 36. How is predictive maintenance different from preventive maintenance?
**Answer:** Preventive maintenance is generally performed at planned time or usage intervals. Predictive maintenance uses condition or operational data to estimate equipment risk and help decide when maintenance may be needed.

### 37. What could happen if the model predicts normal but the machine actually fails?
**Answer:** That is a false negative. It is a significant risk because the system may not flag a machine that later fails. This is why recall matters and why the prototype should not be used as the sole basis for operational safety or maintenance decisions.

### 38. What are the limitations of your project?
**Answer:** The current version uses historical data and manual inputs, is not connected to live equipment, and has not been validated for a particular plant or machine. The recommendations use simple predefined rules. Before operational use, it would need representative plant-specific data, engineering validation, and monitoring.

### 39. How could you improve the project?
**Answer:** I could add live data ingestion through an industrial IoT gateway, validate incoming readings, test the model on plant-specific data, monitor model performance, calibrate probabilities, and integrate maintenance scheduling after engineering review.

### 40. How could this project be useful in a heavy engineering environment?
**Answer:** As a prototype, it demonstrates how operating parameters can be analyzed to identify potential failure risk and support maintenance planning. In a real environment, it would need to be adapted and validated for the specific equipment, sensors, operating conditions, and maintenance procedures.

## Part G: Questions You Should Answer Honestly

### 41. Did you deploy this system in a factory?
**Answer:** No. It is an academic software prototype tested with a public dataset and manual inputs.

### 42. Are your maintenance thresholds approved by L&T?
**Answer:** No. They are predefined demonstration rules in my prototype. They are not L&T-approved or manufacturer-validated thresholds.

### 43. Does 89% failure probability mean the machine has an 89% real-world chance of failing?
**Answer:** Not necessarily. It is the probability estimate produced by the trained model for that input. To interpret it as a real-world probability, the model would need calibration and validation on representative operational data.

### 44. Why should we consider your project relevant if it has no hardware?
**Answer:** The project focuses on the software and machine learning part of predictive maintenance: data preparation, classification, evaluation, visualization, and decision support. Hardware or live sensor integration can be added as a separate future stage.

## Quick Revision: Remember These Project Facts

| Item | Project value |
|---|---|
| Dataset | AI4I 2020 Predictive Maintenance |
| Dataset size | 10,000 records |
| Normal records | 9,661 |
| Failure records | 339 |
| Input features | 6 |
| Target | Machine failure |
| Models compared | Logistic Regression, Decision Tree, Random Forest |
| Selected model | Random Forest |
| Random Forest accuracy | 97.15% |
| Random Forest precision | 56.32% |
| Random Forest recall | 72.06% |
| Random Forest F1 score | 63.23% |
| Application framework | Streamlit |
| Saved model | `src/maintenance_model.pkl` |
| Prediction history | `prediction_history.csv` |

## Interview Practice Advice

- Understand the project workflow rather than memorizing every sentence.
- Be able to explain why recall matters for failure detection.
- Know the difference between false positives and false negatives.
- Be honest that the current prototype uses manual inputs and historical data.
- Do not describe prototype thresholds as certified industrial limits.
- If asked about a detail you have not implemented, explain it as a possible future enhancement rather than claiming it is complete.
