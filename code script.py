import pandas as pd


# data importing
data = pd.read_csv(r"D:\Credit Card Fraud Detection\creditcard.csv")


from sklearn.preprocessing import StandardScaler
   # check for missing values
print(data.isnull().sum())
   # normalize the 'amount' feature
scaler = StandardScaler()
data['normalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

#exploratory data analysis
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# plot with custom colors
sns.countplot(x='Class', data=data, palette=["#59ff00", "#ff0000"])

plt.yscale("log")  

plt.title('Distribution of Fraudulent Transactions (Log Scale)', fontsize=16)
plt.xlabel('Transaction Type', fontsize=12)
plt.ylabel('Count (Log Scale)', fontsize=12)
plt.xticks([0, 1], ['Non-Fraud (0)', 'Fraud (1)'], fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()

#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# optional: create a custom diverging color palette 
colors = ["#4575b4", "#ffffff", "#d73027"]
custom_cmap = LinearSegmentedColormap.from_list("beautiful_corr", colors)

# set the figure and style
plt.figure(figsize=(14, 12))
sns.set(style="white", font_scale=1.1)

# compute the correlation matrix
correlation_matrix = data.corr()

# plot the heatmap
sns.heatmap(correlation_matrix,
            cmap=custom_cmap,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},     
            linewidths=0.7,
            linecolor='lightgray',
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})

# title and ticks
plt.title("Correlation Heatmap", fontsize=18, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


# drop target and irrelevant features
X = data.drop(['Class', 'Amount'], axis=1)
y = data['Class']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create and train the logistic regression model with class imbalance handling
model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# predict and evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))




#gridSearch with class_weight and high max_iter
param_grid = {'C': [0.1, 1, 10]}
model = LogisticRegression(max_iter=10000, class_weight='balanced')  # Add fixes here
grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
grid.fit(X_train_scaled, y_train)

#evaluate
y_pred = grid.best_estimator_.predict(X_test_scaled)
print("Best Parameters:", grid.best_params_)
print(classification_report(y_test, y_pred))

#----------------------------
# ADVANCE MODELS
# including random forest, confusion matrix, feature importance for random forst
#----------------------------------------------------------------------------------
#model Training - Random Forest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#prepare features and target
X = data.drop(['Class', 'Amount'], axis=1)
y = data['Class']

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# (optional) Scale features — not needed for random forest, but we’ll keep it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train random forest with class_weight to handle imbalance
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_scaled, y_train)

# predict and evaluate
y_pred = rf_model.predict(X_test_scaled)
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred))







#----------------------------
# Confusion Matrix (Heatmap)
#----------------------------

cm = confusion_matrix(y_test, y_pred)

# normalize it by row (true label)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6, 5))
sns.set(font_scale=1.2)
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',  # Shows percentage with 2 decimal places
    cmap='Reds',
    linewidths=0.5,
    linecolor='black'
)

plt.title("Normalized Confusion Matrix (%) - Random Forest", fontsize=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'], fontsize=10)
plt.yticks([0.5, 1.5], ['Non-Fraud (0)', 'Fraud (1)'], fontsize=10, rotation=0)
plt.tight_layout()
plt.show()
#----------------------------
#feature importance
#----------------------------
import numpy as np

importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(9, 6))
sns.set_style("whitegrid")
sns.barplot(x=importances[indices],
            y=[X.columns[i] for i in indices],
            palette="viridis")  # Beautiful green-blue palette

plt.title('Top 10 Important Features - Random Forest', fontsize=16)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()