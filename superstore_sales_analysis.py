```python
# ============================================================
# SALES ANALYSIS PROJECT - COMPLETE EDA
# Python | Pandas | NumPy | Matplotlib | Seaborn
# ============================================================

# STEP 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# STEP 2: LOAD DATASET
df = pd.read_csv(
    "Sample - Superstore.csv",
    encoding="windows-1252"
)

print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# STEP 3: UNDERSTAND THE DATA
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# STEP 4: DATA CLEANING
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Convert date columns
df["Order Date"] = pd.to_datetime(
    df["Order Date"], errors="coerce"
)
df["Ship Date"] = pd.to_datetime(
    df["Ship Date"], errors="coerce"
)

# Remove rows with invalid order dates
df = df.dropna(subset=["Order Date"]).copy()

# Remove rows with missing key analysis fields
df = df.dropna(
    subset=["Sales", "Profit", "Quantity"]
).copy()

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)

# STEP 5: FEATURE ENGINEERING
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Month_Name"] = df["Order Date"].dt.month_name()
df["Year_Month"] = df["Order Date"].dt.to_period("M")

# Profit margin (%)
df["Profit_Margin"] = np.where(
    df["Sales"] != 0,
    (df["Profit"] / df["Sales"]) * 100,
    np.nan
)

# STEP 6: KEY PERFORMANCE INDICATORS (KPIs)
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()
total_quantity = df["Quantity"].sum()
overall_margin = (
    total_profit / total_sales * 100
    if total_sales != 0 else np.nan
)

print("\nKEY PERFORMANCE INDICATORS")
print("=" * 50)
print(f"Total Sales       : ${total_sales:,.2f}")
print(f"Total Profit      : ${total_profit:,.2f}")
print(f"Total Orders      : {total_orders:,}")
print(f"Total Customers   : {total_customers:,}")
print(f"Total Quantity    : {total_quantity:,}")
print(f"Profit Margin     : {overall_margin:.2f}%")

# STEP 7: MONTHLY SALES ANALYSIS
monthly_sales = (
    df.groupby("Year_Month", observed=True)["Sales"]
      .sum()
      .sort_index()
)

print("\nMonthly Sales:")
print(monthly_sales)

monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Year-Month")
plt.ylabel("Sales ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# STEP 8: CATEGORY-WISE SALES AND PROFIT
category_analysis = (
    df.groupby("Category")[["Sales", "Profit", "Quantity"]]
      .sum()
      .sort_values("Sales", ascending=False)
)

print("\nCategory-Wise Analysis:")
print(category_analysis)

category_analysis[["Sales", "Profit"]].plot(
    kind="bar"
)
plt.title("Sales and Profit by Category")
plt.ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# STEP 9: TOP 10 PRODUCTS BY SALES
top_products = (
    df.groupby("Product Name")["Sales"]
      .sum()
      .nlargest(10)
)

print("\nTop 10 Products by Sales:")
print(top_products)

top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales ($)")
plt.tight_layout()
plt.show()

# STEP 10: REGION-WISE ANALYSIS
region_analysis = (
    df.groupby("Region")[["Sales", "Profit", "Quantity"]]
      .sum()
      .sort_values("Sales", ascending=False)
)

print("\nRegion-Wise Analysis:")
print(region_analysis)

region_analysis[["Sales", "Profit"]].plot(kind="bar")
plt.title("Sales and Profit by Region")
plt.ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# STEP 11: CUSTOMER SEGMENT ANALYSIS
segment_analysis = (
    df.groupby("Segment")[["Sales", "Profit"]]
      .sum()
      .sort_values("Sales", ascending=False)
)

print("\nCustomer Segment Analysis:")
print(segment_analysis)

segment_analysis.plot(kind="bar")
plt.title("Sales and Profit by Customer Segment")
plt.ylabel("Amount ($)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# STEP 12: SHIPPING MODE ANALYSIS
shipping_analysis = (
    df.groupby("Ship Mode")[["Sales", "Profit"]]
      .sum()
      .sort_values("Sales", ascending=False)
)

print("\nShipping Mode Analysis:")
print(shipping_analysis)

shipping_analysis.plot(kind="bar")
plt.title("Sales and Profit by Shipping Mode")
plt.ylabel("Amount ($)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# STEP 13: DISCOUNT ANALYSIS
print("\nDiscount Statistics:")
print(df["Discount"].describe())

discount_analysis = (
    df.groupby("Discount", as_index=False)
      .agg(
          Total_Sales=("Sales", "sum"),
          Total_Profit=("Profit", "sum"),
          Number_of_Records=("Order ID", "size")
      )
      .sort_values("Discount")
)

print("\nDiscount vs Sales and Profit:")
print(discount_analysis)

sns.scatterplot(
    data=df,
    x="Discount",
    y="Profit",
    hue="Category",
    alpha=0.6
)
plt.title("Discount vs Profit")
plt.tight_layout()
plt.show()

# STEP 14: SALES VS PROFIT ANALYSIS
sns.scatterplot(
    data=df,
    x="Sales",
    y="Profit",
    hue="Category",
    alpha=0.6
)
plt.title("Sales vs Profit")
plt.tight_layout()
plt.show()

# STEP 15: CORRELATION HEATMAP
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# STEP 16: BEST AND WORST PERFORMERS
best_category = category_analysis["Sales"].idxmax()
best_profit_category = category_analysis["Profit"].idxmax()
best_region = region_analysis["Sales"].idxmax()
best_month = monthly_sales.idxmax()
worst_profit_category = category_analysis["Profit"].idxmin()

print("\nBUSINESS INSIGHTS")
print("=" * 50)
print("Highest Sales Category:", best_category)
print("Highest Profit Category:", best_profit_category)
print("Highest Sales Region:", best_region)
print("Highest Sales Month:", best_month)
print("Lowest Profit Category:", worst_profit_category)

print("\nTOP 5 LOSS-MAKING PRODUCTS BY TOTAL PROFIT:")
product_profit = (
    df.groupby("Product Name")["Profit"]
      .sum()
      .sort_values()
)
print(product_profit.head(5))

# STEP 17: EXPORT CLEANED DATA
df.to_csv("Cleaned_Superstore.csv", index=False)

print("\nCleaned dataset saved as Cleaned_Superstore.csv")
print("\nSALES ANALYSIS EDA COMPLETED SUCCESSFULLY!")
```