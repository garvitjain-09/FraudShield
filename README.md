FraudShield

Intelligent Transaction Fraud Detection System


About the Project

FraudShield is a machine learning project for detecting fraudulent financial transactions.

The project takes transaction details as input and uses a trained XGBoost model to predict whether a transaction may be fraudulent.


Project Objective

• Detect potentially fraudulent transactions
• Understand transaction patterns
• Handle highly imbalanced fraud data
• Train and compare machine learning models
• Improve the model using hyperparameter tuning
• Understand model predictions using SHAP
• Create a simple Streamlit application for prediction


Dataset

The project uses the PaySim Synthetic Financial Dataset.

The dataset contains information such as:

• Transaction type
• Transaction amount
• Origin account balance
• Destination account balance
• Transaction step
• Fraud label

The original dataset is not uploaded to GitHub because of its size.


Tools and Technologies

• Python
• Pandas
• NumPy
• Scikit-learn
• XGBoost
• SHAP
• Matplotlib
• Seaborn
• Joblib
• Streamlit
• Jupyter Notebook
• Git and GitHub


Project Steps

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Handling Class Imbalance
7. Model Training
8. Model Evaluation
9. Hyperparameter Tuning
10. SHAP Analysis
11. Model Saving
12. Streamlit Application


Feature Engineering

Two balance-based features were created:

• balance_diff_orig
• balance_diff_dest

A destination transaction count feature was also created:

• dest_transaction_count

Account IDs were not directly used as model features because they have very high cardinality.


Models Used

Logistic Regression

Used as the baseline model.

Random Forest

Used to test a tree-based model and understand nonlinear relationships.

XGBoost

Used as the main model for the final fraud detection system.


Model Evaluation

Because fraud transactions are much fewer than normal transactions, accuracy alone is not enough.

The project uses:

• Precision
• Recall
• F1-score
• Confusion Matrix
• ROC-AUC
• PR-AUC


Final XGBoost Results

The final XGBoost model was tested using a time-based validation approach.

ROC-AUC: 1.00

PR-AUC: 1.00

Fraud Recall: 99.07%

Fraud detected: 212 out of 214


Important Note

The dataset used in this project is synthetic.

The very high model performance on this dataset does not mean that the same performance would be achieved on real banking transactions.

Real-world fraud detection would require testing on real and different datasets, monitoring, and further validation.


SHAP Explainability

SHAP was used to understand which features affected the model predictions.

Some important features included:

• step
• balance_diff_orig
• newbalanceOrig
• Transaction type
• oldbalanceOrg
• amount

SHAP helps to make the model predictions easier to understand.


Streamlit Application

A Streamlit application was created for testing the trained model.

The application allows the user to enter transaction details and shows:

• Fraud prediction
• Fraud probability
• Risk level
• Transaction summary

Risk levels used in the application:

• LOW RISK — below 30%
• MEDIUM RISK — 30% to below 70%
• HIGH RISK — 70% or higher

These are application display thresholds and are not validated banking decision thresholds.


Project Structure

FraudShield/
│
├── app/
│   └── app.py
│
├── models/
│   ├── fraud_xgb_model.pkl
│   ├── type_encoder.pkl
│   └── dest_counts.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── .gitignore
├── requirements.txt
├── README.md
└── Dataset.csv


How to Run

Install the required libraries:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app/app.py


Model Files

The models folder contains:

• fraud_xgb_model.pkl — trained XGBoost model
• type_encoder.pkl — transaction type encoder
• dest_counts.pkl — destination transaction count information


Limitations

• PaySim is a synthetic dataset
• Results may not represent real-world fraud detection
• More datasets are needed to test generalization
• Real fraud systems need continuous monitoring
• Risk thresholds need proper validation before real-world use


Future Improvements

• Test the model on other fraud datasets
• Add more useful transaction features
• Improve probability calibration
• Add database integration
• Build a better fraud monitoring dashboard
• Deploy the application online
• Add model monitoring and retraining


Author

Garvit Jain

