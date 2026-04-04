import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score
)

#Load data file
data_file = pd.read_csv("data/StudentPerformanceFactors.csv")

#Cap scores
data_file["Exam_Score"] = data_file["Exam_Score"].clip(upper=100)

#Create the classification target
data_file["Grade"] = (data_file["Exam_Score"] >= 60).astype(int)
class_labels = ["Fail", "Pass"]

print("Class distribution:\n", data_file["Grade"].value_counts())

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
X = data_file.drop(columns=["Exam_Score", "Grade"])
Y = data_file["Grade"]

#Train 80% Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Logistic Regression: Model 1
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=class_labels))


# Random Forest Classifier: Model 2
rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)

print("RANDOM FOREST CLASSIFIER")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rfc):.4f}")
print(classification_report(y_test, y_pred_rfc, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rfc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Random Forest Classifier")
plt.tight_layout()
plt.show()

# Feature Importance
feature_names = data_file.drop(columns=["Exam_Score", "Grade"]).columns
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices], color="coral")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha="right")
plt.title("Random Forest — Feature Importances (Classification)")
plt.tight_layout()
plt.show()