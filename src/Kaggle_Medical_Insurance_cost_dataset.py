# source: https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset
# Author: Siqi Dai
# Date: 2025-10-01
# Description: Data visualization of Medical Insurance Cost Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/daisiqi/Machine-Learning-for-Thermodynamic-Property-dataset-URS-/data/insurance.csv')

print(df.info())
print("\nDescriptive Statistics:\n", df.describe())

# Identify numerical features excluding the target
numerical_features = ['age', 'bmi', 'children']
target_variable = 'charges'

# =================================================================
# 1. Scatter Plots for Each Numerical Variable vs. Charges (NEW SECTION)
# =================================================================
print("\n--- Generating Scatter Plots ---")
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y=target_variable, data=df, alpha=0.6)
    plt.title(f'Charges vs. {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Charges')
    # Saves the figure
    plt.savefig(f'scatter_charges_vs_{feature}.png')
    plt.close() # Close the figure to free memory

# =================================================================
# 2. Categorical Variable Analysis (Box Plots)
# =================================================================
print("\n--- Generating Box Plots ---")
categorical_features = ['smoker', 'region']

for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=feature, y=target_variable, data=df)
    plt.title(f'Charges by {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Charges')
    plt.savefig(f'boxplot_charges_by_{feature}.png')
    plt.close()

# =================================================================
# 3. Correlation Heatmap
# =================================================================
print("\n--- Generating Heatmap ---")
numerical_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_df.corr(numeric_only=True)

plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('heatmap_numerical_features.png')
plt.close()

print("\nAll visualization files have been generated and saved in your project root directory.")
