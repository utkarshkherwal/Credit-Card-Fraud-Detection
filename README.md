# Credit-Card-Fraud-Detection
A machine learning-based system that detects fraudulent transactions using the imbalanced Credit Card dataset provided by Kaggle. This project utilizes data preprocessing, class imbalance handling (SMOTE), and classification models to accurately identify and flag fraud with high precision.

# Project Objective:
To build a predictive model that can classify whether a given transaction is fraudulent or not, based on anonymized features extracted from real credit card transactions.

# Dataset:
Source: Kaggle - Credit Card Fraud Detection
 Description: Contains transactions made by European cardholders in September 2013
 Total Rows: 284,807
 Fraudulent Transactions: ~0.17% (highly imbalanced)

 # Tech Stack:
# Language: Python
Libraries / Modules Used: pandas, numpy, matplotlib.pyplot, seaborn, matplotlib.colors, sklearn.preprocessing.StandardScaler, sklearn.linear_model.LogisticRegressionsklearn.ensemble.RandomForestClassifier, sklearn.metrics.classification_report, confusion_matrix, sklearn.model_selection.GridSearchCV, sklearn.model_selection.train_test_split

# Workflow of the Project
# 1. Data Importing and Preprocessing
- File Read: Load the dataset using pandas.
- Missing Values: Checked using data.isnull().sum() (good practice).
- Feature Scaling: Applied StandardScaler to the Amount column and created a new column normalizedAmount.


# 2. Exploratory Data Analysis (EDA)
A. Class Distribution Plot
- Used seaborn to visualize the imbalance between fraud (1) and non-fraud (0).
- Applied log scale to better show the rare fraud cases.
- Used custom color palette (green for non-fraud, red for fraud).

B. Correlation Heatmap
- Created a custom diverging colormap.
- Used .corr() to generate the correlation matrix.
- Plotted a beautiful annotated heatmap to understand feature relationships.

# 3. Logistic Regression Model
A. Preprocessing
- Dropped Class and Amount from features.
- Split the data using train_test_split.
- Scaled the features with StandardScaler.

B. Model Training
Trained LogisticRegression with:
- max_iter=10000 (to ensure convergence)
- class_weight='balanced' (to handle imbalance)

C. Evaluation
- Used classification_report to check precision, recall, F-score.

D. Grid Search for Hyperparameter Tuning
- Tuned C parameter using GridSearchCV.
- Again handled class imbalance and retrained best model.
- Printed best params and performance.

# 4. Random Forest Model (Advanced Model)
A. Training
- Repeated preprocessing (optional scaling).
- Trained RandomForestClassifier with class_weight='balanced'.

B. Evaluation
- Used classification_report again for results.

# 5. Model Evaluation and Visualization
A. Confusion Matrix (Random Forest)
- Used confusion_matrix.
- Normalized and plotted a heatmap showing true positives, false positives, etc., in percentage.

B. Feature Importance
- Extracted feature importances from the trained Random Forest.

- Plotted the top 10 important features using a viridis palette.



# Future Improvements:
* Add more classification models (e.g. XGBoost)

* Deploy the model using Streamlit or Flask

* Add real-time fraud detection simulation
