import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Load data file
data_file = pd.read_csv("data/StudentPerformanceFactors.csv")

#Handle any missing values
for col in data_file.select_dtypes(include="number").columns:
    data_file[col].fillna(data_file[col].median(), inplace=True)

#Fill categorical columns with the most common values
for col in data_file.select_dtypes(include="str").columns:
    data_file[col].fillna(data_file[col].mode()[0], inplace=True)

#Convert text to integers
le = LabelEncoder()
for col in data_file.select_dtypes(include="str").columns:
    data_file[col] = le.fit_transform(data_file[col])

# Split into target and features
X = data_file.drop(columns=["Exam_Score"])
Y = data_file["Exam_Score"]

#Train 80% Test 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Linear Regression: Model 1
lr = LinearRegression()
lr.fit(X_train, Y_train)
y_pred_lr = lr.predict(X_test)

print("=" * 50)
print("Linear Regression")
print(f" MAE  : {mean_absolute_error(Y_test, y_pred_lr):.2f}")
print(f" RMSE : {np.sqrt(mean_squared_error(Y_test, y_pred_lr)):.2f} ")
print(f" R²   : {r2_score(Y_test, y_pred_lr):.4f}")

#Random Forest Regressor: Model 2
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST REGRESSOR")
print(f"  MAE  : {mean_absolute_error(Y_test, y_pred_rf):.2f}")
print(f"  RMSE : {np.sqrt(mean_squared_error(Y_test, y_pred_rf)):.2f}")
print(f"  R²   : {r2_score(Y_test, y_pred_rf):.4f}")

#Feature Importance
feature_names = data_file.drop(columns=["Exam_Score"]).columns
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], color="steelblue")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha="right")
plt.title("Random Forest — Feature Importances (Regression)")
plt.tight_layout()
plt.show()

#Visual
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, y_pred_rf, alpha=0.3, color="steelblue")
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted — Random Forest")
plt.tight_layout()
plt.show()