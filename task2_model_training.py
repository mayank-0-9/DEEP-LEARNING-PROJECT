"""
Task 2: Machine Learning Model for Employee Churn Prediction
Internship: CodTech
Author: Aniket

This script loads the preprocessed data from Task 1, trains a Random Forest Classifier,
evaluates its performance, and saves the predictions and the model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load Preprocessed Data
try:
    X_train = pd.read_csv('X_train_transformed.csv')
    X_test = pd.read_csv('X_test_transformed.csv')
    y_train = pd.read_csv('y_train.csv')['target']
    y_test = pd.read_csv('y_test.csv')['target']
    print("✅ Transformed data loaded successfully.")
except Exception as e:
    print("❌ Error loading data:", e)

# Step 2: Initialize Classifier
model = RandomForestClassifier(random_state=42)
print("📌 Random Forest Classifier initialized.")


# Step 3: Train the Model

model.fit(X_train, y_train)
print("✅ Model training completed.")

# ------------------------------
# Step 4: Make Predictions
# ------------------------------
y_pred = model.predict(X_test)
print("📈 Predictions generated.")

# ------------------------------
# Step 5: Evaluate the Model
# ------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy:.2f}\n")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 6: Save Predictions
# ------------------------------
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
predictions_df.to_csv('predictions.csv', index=False)
print("📁 Predictions saved to 'predictions.csv'")

# ------------------------------
# Step 7: Save Trained Model
# ------------------------------
joblib.dump(model, 'random_forest_model.pkl')
print("💾 Model saved as 'random_forest_model.pkl'")