# DEEP-LEARNING-PROJECT

COMPANY: CODETECH IT SOLUTION

NAME: MAYANK SAGAR

INTERN ID: CT04DN926

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Task Description:
                  Introduction: This task focused on building a machine learning model to predict employee churn based on previously preprocessed data. Following the successful completion of Task 1, which involved creating a data pipeline and transforming raw employee data into clean, numeric-ready formats, this task extends the process by training a predictive model. The model aims to determine whether an employee is likely to leave the company (churn = 1) or stay (churn = 0). We employed a Random Forest Classifier, a powerful ensemble learning algorithm, due to its efficiency, accuracy, and ease of implementation.

Step 1: Loading the Preprocessed Data The first step was to load the transformed CSV files generated during Task 1. These included:

X_train_transformed.csv

X_test_transformed.csv

y_train.csv

y_test.csv


Pandas was used to load these files into DataFrame objects. The target column (target) was extracted separately from y_train.csv and y_test.csv to ensure compatibility with the training process.

Step 2: Initializing the Model We used the RandomForestClassifier from Scikit-learn. This model builds multiple decision trees and combines their output to improve prediction accuracy and avoid overfitting. The model was initialized with a fixed random state for reproducibility.

Step 3: Model Training The model was trained using the .fit() method with X_train and y_train. This step involves the model learning patterns and relationships from the data that can help it predict churn outcomes.

Step 4: Making Predictions After training, predictions were generated using the .predict() method on X_test. These predictions represented whether the model believed the employees in the test set would churn or not.

Step 5: Evaluating the Model The performance of the model was evaluated using:

accuracy_score: to check the overall percentage of correct predictions.

classification_report: to provide precision, recall, and f1-score for both churn (1) and non-churn (0) classes.


This evaluation ensured that we not only understood how often the model was right, but also how well it distinguished between employees likely to leave and those likely to stay.

Step 6: Saving the Output The predictions made by the model were stored in a CSV file named predictions.csv. This file included the actual labels and the predicted labels side by side, making it easier to review the model's performance manually if needed. Saving the predictions provides transparency and allows for further analysis, such as error inspection and result visualization.

Step 7: Exporting the Trained Model To support reusability and deployment, the trained Random Forest model was exported using the joblib.dump() method. This saved the model in a file called random_forest_model.pkl. The benefit of saving the model is that we can later load it directly to make predictions on new or real-time data without the need to retrain.

Conclusion: Task 2 represents a crucial step in the data science workflow, transitioning from data preprocessing to model development. By using an ensemble-based classifier and evaluating it using appropriate metrics, we ensured both robustness and reliability of our prediction system. This task not only enhances our understanding of model building but also demonstrates the ability to handle end-to-end data science workflows, aligning with industry practices and internship goals.

Ultimately, the approach taken in Task 2 showcases how a clean dataset can be effectively leveraged to build, evaluate, and save a machine learning model that contributes toward solving real-world business problems like employee retention.
