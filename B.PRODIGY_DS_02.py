import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a synthetic dataset similar to Titanic with different values
np.random.seed(42)
n_samples = 891

synthetic_data = pd.DataFrame({
    'survived': np.random.randint(0, 2, size=n_samples),
    'pclass': np.random.randint(1, 4, size=n_samples),
    'sex': np.random.choice(['male', 'female'], size=n_samples),
    'age': np.random.uniform(2, 90, size=n_samples),  # Changed age range
    'sibsp': np.random.randint(0, 6, size=n_samples),  # Changed sibsp range
    'parch': np.random.randint(0, 7, size=n_samples),  # Changed parch range
    'fare': np.random.uniform(5, 500, size=n_samples),  # Changed fare range
    'embarked': np.random.choice(['C', 'Q', 'S', 'X'], size=n_samples)  # Added 'X' as new embark point
})

# Data Cleaning

# Check for missing values
missing_values = synthetic_data.isnull().sum()
print("Missing values:\n", missing_values)

# Check for duplicates
duplicates = synthetic_data.duplicated().sum()
print("Number of duplicates:", duplicates)

# Check data types
data_types = synthetic_data.dtypes
print("Data types:\n", data_types)

# Exploratory Data Analysis (EDA)

# Descriptive statistics for numerical columns
desc_stats = synthetic_data.describe()
print("Descriptive statistics:\n", desc_stats)

# Distribution of categorical variables
plt.figure(figsize=(20, 5))

# Distribution of sex
plt.subplot(1, 3, 1)
sns.countplot(data=synthetic_data, x='sex', palette='Set3')
plt.title('Distribution of Sex')

# Distribution of pclass
plt.subplot(1, 3, 2)
sns.countplot(data=synthetic_data, x='pclass', palette='Set1')
plt.title('Distribution of Pclass')

# Distribution of embarked
plt.subplot(1, 3, 3)
sns.countplot(data=synthetic_data, x='embarked', palette='Set2')
plt.title('Distribution of Embarked')

plt.tight_layout()
plt.show()

# Correlation analysis
numeric_data = synthetic_data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Spectral', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Survival analysis
survival_sex = synthetic_data.groupby('sex')['survived'].mean()
print("Survival rate by sex:\n", survival_sex)

survival_pclass = synthetic_data.groupby('pclass')['survived'].mean()
print("Survival rate by pclass:\n", survival_pclass)

survival_embarked = synthetic_data.groupby('embarked')['survived'].mean()
print("Survival rate by embarked:\n", survival_embarked)
