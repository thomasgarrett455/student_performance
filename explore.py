import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
data_file = pd.read_csv("data/StudentPerformanceFactors.csv")

#Taking a look at the data
print("Shape:", data_file.shape)
print("\nColumns names:\n", data_file.columns.tolist())
print("\nFirst 5 rows:")
print(data_file.head())

#Finding data types and missing values
print("\nData Types:\n", data_file.dtypes)
print("\nMissing values per column:\n", data_file.isnull().sum())

#Statistics
print("\nNumeric summary:")
print(data_file.describe())

#Unique Values of the columns
cat_cols = data_file.select_dtypes(include="str").columns
for col in cat_cols:
    print(f"\n{col} -> {data_file[col].nunique()} unique values: {data_file[col].unique()}")

#Visualize Exam scores
plt.figure(figsize=(8, 4))
sns.histplot(data_file["Exam_Score"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of Exam Scores")
plt.xlabel('Exam Score')
plt.tight_layout()
plt.show()

#Correlation heatmap
plt.figure(figsize=(10,6))
numeric_data_file = data_file.select_dtypes(include="number")
sns.heatmap(numeric_data_file.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
plt.show()