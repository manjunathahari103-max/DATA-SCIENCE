import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("supermarket_sales.csv")

print(df.head())

print("\nDataset Info")
print(df.info())

print("\nTotal Sales:", df["Total"].sum())

print("Average Sales:", df["Total"].mean())

print("\nSales by Branch")
print(df.groupby("Branch")["Total"].sum())

print("\nSales by Product Line")
print(df.groupby("Product line")["Total"].sum())

print("\nPayment Method")
print(df["Payment"].value_counts())

df.groupby("Product line")["Total"].sum().plot(kind="bar")
plt.title("Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

df["Payment"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Payment Methods")
plt.ylabel("")
plt.show()

daily_sales = df.groupby("Date")["Total"].sum()

daily_sales.plot(figsize=(10,5))
plt.title("Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.show()

import seaborn as sns

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()
