# source: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
# Author: Siqi Dai
# Date: 2025-10-01
# Description: Data visualization of Medical Insurance Cost Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data (adjust the path if the file is in a subdirectory)
df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/insurance.csv')

print(df.info())
print("\nDescriptive Statistics:\n", df.describe())

# 1. Distribution of Charges (Histogram/KDE)
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, bins=30)
plt.title('1. Distribution of Medical Insurance Charges')

# 2. Charges vs. Age (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=df)
plt.title('2. Charges vs. Age')

# 3. Smoker vs. Non-Smoker Charges (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('3. Charges by Smoking Status')

# 4. Correlation Heatmap (Numerical Variables)
# Note: Use df.corr(numeric_only=True) for modern Pandas versions
numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr(numeric_only=True)

plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('4. Correlation Heatmap of Numerical Features')

# Show all generated plots
plt.tight_layout()
plt.show()
