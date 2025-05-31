# DEEP-LEARNING-PROJECT

COMPANY: CODETECH IT SOLUTION

NAME: MAYANK SAGAR

INTERN ID: CT04DN926

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Task Description:
Project Overview
The second task of my internship at CodTech was to build a machine learning pipeline for predicting employee churn—i.e., identifying employees who are likely to leave the company. Churn prediction is a critical use-case in HR analytics, as early identification of at-risk employees enables organizations to take timely action, design better retention policies, and save costs related to employee turnover.

After completing Task 1, which focused on data preprocessing and pipeline development, the next logical step was to use the clean, transformed data to develop, train, and evaluate a machine learning model. I selected the Random Forest Classifier due to its popularity and proven performance on structured/tabular datasets. The goal of the project was not only to achieve high predictive accuracy but also to create a repeatable workflow that can be adapted for other binary classification tasks in business.

Data Loading and Preparation
The first phase involved loading the processed data files generated from Task 1. These included separate CSV files for training and testing input features (X_train_transformed.csv and X_test_transformed.csv) and for the corresponding target labels (y_train.csv and y_test.csv). Using preprocessed data ensured consistency in feature engineering and avoided issues like missing values or inconsistent formats, which are common challenges in real-world data science projects.

I used the Pandas library for reading CSV files into DataFrame objects. Basic error handling was implemented to catch and gracefully report any data-loading issues, which is a good practice for robust and user-friendly scripts.

Model Selection and Training
I opted for a Random Forest Classifier from scikit-learn. Random Forest is an ensemble technique that builds multiple decision trees during training and aggregates their output to improve generalization and reduce the risk of overfitting. I initialized the model with a fixed random state for reproducibility so that results remain the same across runs.

The model was trained (fit) using the training set features and targets. Training a Random Forest involves automatically handling nonlinear feature interactions and provides built-in ways to estimate feature importance, making it highly suitable for HR churn data, which often has a mix of categorical and continuous features.

Evaluation and Interpretation
After training, the model's performance was evaluated on the test set. I used accuracy_score to measure the proportion of correct predictions and classification_report for a detailed breakdown of precision, recall, and F1-score for both classes (churned and not churned employees). This step is essential not only to assess whether the model generalizes well to unseen data but also to diagnose potential issues like class imbalance or bias in predictions.

Console print statements at each stage were added to improve user experience and transparency, giving instant feedback about data loading, training progress, prediction status, and evaluation outcomes.

Saving Predictions and Model
The script saves all predictions from the test set to a new CSV file (predictions.csv). This file allows further analysis, such as error inspection, sharing with stakeholders, or feeding into dashboards. Additionally, the trained Random Forest model is saved using joblib—so that it can be easily loaded and used for future predictions without needing to retrain. The script prints a friendly message upon successful saving, improving clarity for the next user (or reviewer).

Learning and Professional Skills Developed
This task strengthened my understanding of a typical machine learning workflow, from reading and validating data, through training and evaluation, to outputs and model persistence. Special attention was given to:

Structuring code for readability and maintainability
Handling data pipeline outputs for seamless workflow integration
Adding user-friendly messaging and basic error handling
The approach followed here is robust and industry-standard, and can be adapted to any supervised classification problem.

Conclusion
In summary, this task not only achieved the goal of building a high-performing, real-world machine learning model for employee churn prediction but also developed skills in professional coding practices and end-to-end workflow design. The deliverables—namely, the prediction csv, detailed performance report, and serializable model—are ready for deployment in an actual business setting or further extension (e.g., feature importance analysis, advanced ensemble methods).
