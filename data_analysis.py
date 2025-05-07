# Importing necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Importing necessary files
from importing_dataset import initial_dataset

# Dropping the 'id' column
initial_dataset.drop(columns='id', inplace=True)

# Numeric columns
COLUMNS = ['age', 'hypertension', 'heart_disease',
           'avg_glucose_level', 'bmi',
           'stroke']

# Displaying information about data types and non-null values
print("Dataset information about types and non-null values:\n",initial_dataset.info())

# Displaying the first 5 rows of the dataset
print("First 5 rows of the dataset:\n", initial_dataset.head() )

# Displaying statistics of the dataset
print("Dataset statistics:\n", initial_dataset.describe())

# Displaying skewness of the dataset
print("Skewness of the dataset:\n", initial_dataset[COLUMNS].skew())

# Displaying and visualizing the correlation matrix

correlation_matrix = initial_dataset[COLUMNS].corr()
print("Correlation matrix:\n", correlation_matrix)

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Encoding binary variables
initial_dataset['gender'] = initial_dataset['gender'].map({"Male": 1, "Female": 0})
initial_dataset['ever_married'] = initial_dataset['ever_married'].map({"Yes": 1, "No": 0})
initial_dataset['Residence_type'] = initial_dataset['Residence_type'].map({"Urban": 1, "Rural": 0})

# One-hot encoding for work_type and smoking_status
initial_dataset = pd.get_dummies(initial_dataset, columns=['work_type', 'smoking_status'], drop_first=True)

# Final dataset ready for modeling
prepared_dataset = initial_dataset.copy()

# Final check
print("Prepared dataset info:\n", prepared_dataset.info())
print("First 5 rows of prepared dataset:\n", prepared_dataset.head())
