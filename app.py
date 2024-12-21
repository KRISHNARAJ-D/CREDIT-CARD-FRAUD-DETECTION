from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Get the uploaded file from the request
        uploaded_file = request.files['file']
        
        if not uploaded_file or uploaded_file.filename == '':
            return "Error: No file uploaded."

        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(uploaded_file)
        
        # Clean the column names
        data.columns = data.columns.str.replace('"', '').str.strip()
        
        # Check if 'Class' column exists in the dataset
        if 'Class' not in data.columns:
            return "Error: Target column 'Class' not found in the dataset."

        # Separate features and target variable
        X = data.drop('Class', axis=1)
        Y = data['Class']
        
        # Check if the dataset contains at least two classes
        unique_classes = Y.unique()
        if len(unique_classes) < 2:
            return "Error: The dataset must contain at least two classes (e.g., legitimate and fraudulent transactions)."

        # Display class distribution
        class_distribution = Counter(Y)
        print(f"Class Distribution: {class_distribution}")
        
        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        # Compute class weights for handling imbalanced classes
        class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
        class_weight_dict = dict(zip(np.unique(Y), class_weights))

        # Initialize and train the Random Forest model with class weights
        model = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
        model.fit(X_train, Y_train)
        
        # Calculate feature importances
        feature_importances = model.feature_importances_

        # Map feature names to their importance values
        feature_importances_dict = dict(zip(X.columns, feature_importances))
        
        # Perform predictions and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        
        # Confusion Matrix
        cm = confusion_matrix(Y_test, predictions)
        print(f"Confusion Matrix:\n{cm}")
        
        # Count fraudulent and legitimate transactions
        fraudulent_count = (predictions == 1).sum()
        legitimate_count = (predictions == 0).sum()
        
        # Calculate fraud percentage
        fraud_percentage = (fraudulent_count / len(predictions)) * 100
        
        # Render the results to the result.html template
        return render_template(
            'result.html',
            total_transactions=len(predictions),
            fraudulent_count=fraudulent_count,
            legitimate_count=legitimate_count,
            fraud_percentage=fraud_percentage,
            accuracy=accuracy * 100,
            feature_importances=feature_importances_dict
        )

    except Exception as e:
        return f"Error processing the file: {e}"

if __name__ == "__main__":
    app.run(debug=True)
