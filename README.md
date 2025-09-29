# task-5
It appears you're having trouble copying the content up to the **"Summary of Findings"** section from the provided Jupyter Notebook (`Titanic_EDA_from_task_5.ipynb`).

Here is the full text from the notebook, starting from the beginning up to and including the **Summary of Findings** section (Section 5), formatted as a single block of text for easier copying:

```
# 📘 Task 5: Exploratory Data Analysis (EDA) on the Titanic Dataset
# ----------------------------------------------------------------------------
# Task 5: Exploratory Data Analysis (EDA) on the TITANIC DATASET
# Objective: Extract insights using visual and statistical exploration.
# ----------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## 1️⃣ Data Loading and Initial Cleaning
df = pd.read_csv(r'"C:\Users\veera\OneDrive\Desktop\Documents\Desktop\elevate labs\titanic.csv"')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Pclass'] = df['Pclass'].astype('category')
df.head()

## 2️⃣ Statistical Exploration
# Dataset Info
df.info()

# Descriptive statistics
df.describe(include='all')

# Value counts
print("Sex Distribution (%)")
print(df['Sex'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\nSurvival Distribution (%)")
print(df['Survived'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\nPclass Counts")
print(df['Pclass'].value_counts(sort=False))

## 3️⃣ Visual Exploration

### Age and Fare Distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['Age'], bins=30, kde=True, color='#2c7bb6')
plt.title('Age Distribution (Histogram)')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'], color='#fdae61')
plt.title('Fare Distribution (Boxplot)')
plt.tight_layout()
plt.show()

**Observations:**
- Most passengers were between 20–40 years old.
- Fare distribution is highly skewed, with a few very high outliers.

### Survival Rate by Categorical Factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x='Sex', y='Survived', data=df, palette='Set1', ax=axes[0])
axes[0].set_title('Survival Rate by Sex')

sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis', ax=axes[1])
axes[1].set_title('Survival Rate by Pclass')

sns.barplot(x='Embarked', y='Survived', data=df, palette='cool', ax=axes[2])
axes[2].set_title('Survival Rate by Embarked Port')
plt.tight_layout()
plt.show()

**Observations:**
- Females had much higher survival chances than males.
- 1st-class > 2nd-class > 3rd-class survival hierarchy.
- Passengers from Cherbourg survived more than Southampton/Queenstown.

### Age vs Fare (Scatterplot by Survival)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, alpha=0.7, palette='Spectral')
plt.title('Age vs Fare, Grouped by Survival')
plt.legend(title='Survived', labels=['No (0)', 'Yes (1)'])
plt.show()

**Observations:**
- Survivors often paid higher fares.
- Children <10 had higher survival, regardless of fare.

## 4️⃣ Advanced Visualizations

### Pairplot
numerical_df = df[['Survived', 'Age', 'SibSp', 'Parch', 'Fare']]
sns.pairplot(numerical_df, hue='Survived', palette='coolwarm', diag_kind='kde')
plt.suptitle('Pairplot of Numerical Features by Survival', y=1.02)
plt.show()

### Correlation Heatmap
df['Sex_Numeric'] = df['Sex'].map({'male': 0, 'female': 1})
df['Pclass_Numeric'] = df['Pclass'].astype(int)

correlation_cols = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_Numeric', 'Pclass_Numeric']
correlation_matrix = df[correlation_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(
    correlation_matrix, annot=True, cmap='coolwarm',
    fmt='.2f', linewidths=.5, linecolor='black', cbar_kws={'label': 'Correlation Coefficient'}
)
plt.title('Correlation Matrix of Key Features')
plt.show()

**Observations:**
- Sex shows the strongest correlation with survival (+0.54).
- Pclass is negatively correlated (-0.34).
- Fare shows moderate positive correlation (+0.26).
- Age has a weak negative correlation.

## 5️⃣ 📌 Summary of Findings
1. **Gender** is the strongest predictor of survival → females had higher chances.
2. **Socio-economic status (Pclass & Fare)** significantly influenced outcomes.
3. **Children (Age < 10)** had higher survival rates → evacuation priority.
4. **Cherbourg (C) passengers** had better survival odds than other ports.
5. Titanic survival was shaped by both **social norms (women/children first)** and **economic status**.
```
