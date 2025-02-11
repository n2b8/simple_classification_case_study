# Customer Churn Prediction Using Scikit-Learn

## Overview
This Jupyter Notebook provides an end-to-end implementation of **customer churn prediction** using machine learning models in **Scikit-Learn**. The dataset used is the **Telco Customer Churn Dataset**, which contains customer demographics, account details, and service usage information.

## Dataset
- Source: [Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- Contains **7,043** customer records
- Features include:
  - **Demographics** (e.g., gender, senior citizen, partner, dependents)
  - **Account Information** (e.g., contract type, payment method)
  - **Service Usage** (e.g., tenure, monthly charges, total charges)
  - **Target Variable**: `Churn` (1 = Yes, 0 = No)

## Steps in the Notebook
1. **Load and Preprocess Data**
   - Convert `TotalCharges` to numeric
   - Handle missing values
   - Encode categorical variables using **OneHotEncoding**
   - Standardize numerical features

2. **Train Three Classification Models**
   - **Logistic Regression** (baseline model)
   - **Decision Tree Classifier** (interpretable model)
   - **Random Forest Classifier** (ensemble method for better accuracy)

3. **Model Evaluation**
   - **Classification Reports** (Precision, Recall, F1-score)
   - **Confusion Matrices** (Visual representation of predictions)
   - **Comparison of Model Performance**

4. **Results and Findings**
   - Logistic Regression provides a **strong baseline** with high interpretability.
   - Decision Trees offer **better flexibility** but can overfit.
   - Random Forests improve **overall accuracy** but are computationally expensive.

## Outputs
- **Classification Reports**
  - `logistic_regression_report.csv`
  - `decision_tree_report.csv`
  - `random_forest_report.csv`
- **Confusion Matrix Plots** for each model

## How to Run the Notebook
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
2. Open Jupyter Notebook and run the cells step by step.
3. Modify the hyperparameters or try different models for experimentation.

## Future Improvements
- Add **Hyperparameter Tuning** with GridSearchCV.
- Explore **Other ML Algorithms** (e.g. Gradient Boosting, SVM).
- Implement **Feature Selection** for better model performance.
- Use a **Deep Learning Model** (e.g. TensorFlow, PyTorch) for comparison.

## Author
- Jake McCaig
- Medium Blog: (@mccaigjake)[https://medium.com/@mccaigjake]
- Contact: McCaigJake@gmail.com