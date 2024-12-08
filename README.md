# Optimizing Credit Card Fraud Detection with Random Forests and SMOTE

## 📌 Project Description
**Credit Card Fraud Detection** is a machine learning-based project designed to identify fraudulent transactions using transaction data. The project leverages advanced techniques like **Synthetic Minority Oversampling (SMOTE)** to address class imbalance and implements various machine learning models, including **Random Forest**, to achieve high accuracy and recall.

### 🚀 Key Features:
- **High Accuracy:** The Random Forest model achieved 99.5% accuracy in detecting fraudulent transactions.
- **Real-Time Predictions:** Integrated with a Flask-based web application for easy deployment.
- **Scalability:** Handles large datasets with minimal computational overhead.
- **User-Friendly Interface:** Simple and intuitive dashboard for uploading CSV files and viewing results.

## 🛠️ Methodology
1. **Data Preprocessing:**
   - Removed redundant values and standardized feature names.
   - Addressed class imbalance using **SMOTE**.
   - Applied **Principal Component Analysis (PCA)** to reduce dimensionality.

2. **Model Selection:**
   - Models evaluated: Logistic Regression, Decision Tree, Random Forest, and Neural Networks.
   - Hyperparameter tuning via grid search and cross-validation.

3. **Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

4. **Model Deployment:**
   - Saved the trained model and scaler using **joblib**.
   - Deployed as a Flask-based application for real-time fraud detection.

## 📊 Results
- **Model Performance:**
  - Random Forest achieved **99.5% accuracy**, precision, and recall.
  - High fraud detection rate with minimal false positives.

- **Visualizations:**
  - Bar charts and graphs for feature importance and performance metrics.

## 🖥️ Dashboard Features
### 🏠 Homepage:
- Simple interface inviting users to upload transaction data.
- File upload feature for analyzing CSV files.

### 📈 Results Page:
- Displays:
  - **Total Transactions**
  - **Fraudulent Transactions**
  - **Legitimate Transactions**
  - **Fraud Percentage**
- Option to upload a new dataset for analysis.

## ⚙️ Installation
### Prerequisites
- Python 3.7+
- Libraries:
  - Flask
  - scikit-learn
  - pandas
  - matplotlib
  - seaborn

### Steps
Steps

1. Clone the repository


2. Navigate to the project directory


3. Install dependencies:
```bash
pip install -r requirements.txt
```


4. Run the Flask application:
```
python app.py
```
## 💻 Usage

  1.Open the application in your web browser at http://localhost:5000/.

  2.Upload a CSV file with transaction data.

  3.View results on the dashboard.
## 🔧 Technologies Used

-  Programming Languages: Python

-  Frameworks: Flask

- Libraries:

  - **scikit-learn**

   - **pandas**

   - **matplotlib**

   - **seaborn**

- Algorithms: Logistic Regression, Decision Tree, Random Forest, Neural Networks

## 🌟 Future Enhancements

- **Integrate deep learning models like CNNs or RNNs.**

- **Include additional contextual features such as transaction history.**

- **Optimize for deployment in distributed systems.**

## 👥 Authors

- **Krishnaraj D Gmail: krishnakrishna4123@gmail.com**

- **Lokesh Rahul V V Gmail: lokeshrahulvl11@gmail.com**

- **Raja R Gmail: raja.rrasu@gmail.com**

