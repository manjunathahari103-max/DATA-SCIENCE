# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Shape of Dataset
print("\nDataset Shape:", df.shape)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Check Duplicate Rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Correlation Matrix
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(x=df["Fare"])
plt.title("Fare Box Plot")
plt.show()

# Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Distribution")
plt.show()

# Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class")
plt.show()

# Scatter Plot
plt.figure(figsize=(6,4))
plt.scatter(df["Age"], df["Fare"])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare")
plt.show()

print("\nEDA Completed Successfully!")
