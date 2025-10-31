# combined_code_safe.py
# E-Commerce Data Analysis Script — Robust for missing columns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import timedelta, date

# -----------------------------
# Visual settings
# -----------------------------
plt.rcParams['figure.figsize'] = (10,6)
sns.set(style='whitegrid')

# -----------------------------
# Dataset path
# -----------------------------
DATA_PATH = 'ecommerce_data.csv'

# -----------------------------
# Generate synthetic dataset
# -----------------------------
def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

n_customers = 1000
n_products = 200
n_orders = 5000

start_date = date(2023,1,1)
end_date = date(2024,12,31)

customers = [f'CUST_{i:05d}' for i in range(1, n_customers+1)]
products = [f'PROD_{i:04d}' for i in range(1, n_products+1)]
categories = ['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports', 'Toys']

rows = []
for order_id in range(1, n_orders+1):
    customer = random.choice(customers)
    product = random.choice(products)
    category = random.choice(categories)
    qty = random.choices([1,2,3,4], weights=[80,12,5,3])[0]
    price = round(random.uniform(5, 500), 2)
    order_date = random_date(start_date, end_date)
    total = round(qty * price, 2)
    payment = random.choice(['Credit Card', 'Debit Card', 'UPI', 'NetBanking', 'COD'])
    country = random.choice(['India','USA','UK','Germany','Canada','Australia'])
    rows.append({
        'order_id': f'ORDER_{order_id:06d}',
        'customer_id': customer,
        'product_id': product,
        'category': category,
        'quantity': qty,
        'price': price,
        'order_date': order_date.strftime('%Y-%m-%d'),
        'total_amount': total,
        'payment_method': payment,
        'country': country
    })

df_synth = pd.DataFrame(rows)
df_synth.to_csv(DATA_PATH, index=False)
print(f'Synthetic dataset saved to {DATA_PATH} — rows: {len(df_synth)}')
print(df_synth.head())

# -----------------------------
# Load dataset
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH, parse_dates=['order_date'])
except Exception:
    df = pd.read_csv(DATA_PATH)
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

print('Dataset loaded — shape:', df.shape)
print(df.head())

# -----------------------------
# Basic info & missing values
# -----------------------------
print(df.info())
print(df.describe(include='all').T)

missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct}))

# -----------------------------
# Data cleaning
# -----------------------------
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f'Dropped {before-after} duplicate rows')

# Fill missing total_amount
if 'total_amount' in df.columns and 'quantity' in df.columns and 'price' in df.columns:
    missing_total = df['total_amount'].isnull().sum()
    if missing_total > 0:
        df.loc[df['total_amount'].isnull(), 'total_amount'] = df['quantity'] * df['price']
        print('Filled', missing_total, 'missing total_amount')

# Drop rows with missing critical columns
critical_cols = ['order_id', 'customer_id', 'order_date', 'total_amount']
for c in critical_cols:
    if c in df.columns:
        missing = df[c].isnull().sum()
        if missing > 0:
            print(f'Dropping {missing} rows with missing {c}')
            df = df.dropna(subset=[c])

# Convert datatypes
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
if 'category' in df.columns:
    df['category'] = df['category'].astype('category')

print('After cleaning — shape:', df.shape)

# -----------------------------
# Feature engineering
# -----------------------------
if 'order_date' in df.columns:
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_dayofweek'] = df['order_date'].dt.day_name()

# -----------------------------
# Customer-level aggregates
# -----------------------------
if 'customer_id' in df.columns:
    cust_agg = df.groupby('customer_id').agg({
        'order_id': 'nunique',
        'total_amount': ['sum', 'mean'],
    }).reset_index()
    cust_agg.columns = ['customer_id', 'n_orders', 'total_spent', 'avg_order_value']
    print(cust_agg.sort_values('total_spent', ascending=False).head(10))
else:
    print("Column 'customer_id' missing — skipping customer aggregation.")

# -----------------------------
# Monthly sales
# -----------------------------
if 'order_date' in df.columns and 'total_amount' in df.columns:
    df_month = df.set_index('order_date').resample('M')['total_amount'].sum().reset_index()
    plt.figure(figsize=(12,5))
    sns.lineplot(data=df_month, x='order_date', y='total_amount', marker='o')
    plt.title('Monthly Sales (total_amount)')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.show()
else:
    print("Columns 'order_date' or 'total_amount' missing — skipping monthly sales plot.")

# -----------------------------
# Top products and categories
# -----------------------------
if 'product_id' in df.columns:
    top_products = df.groupby('product_id')['total_amount'].sum().sort_values(ascending=False).head(10)
    print(top_products.reset_index().rename(columns={'total_amount':'total_sales'}))
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Total Sales')
    plt.ylabel('Product ID')
    plt.tight_layout()
    plt.show()
else:
    print("Column 'product_id' missing — skipping top products plot.")

if 'category' in df.columns:
    cat_sales = df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=cat_sales.values, y=cat_sales.index)
    plt.title('Sales by Category')
    plt.xlabel('Total Sales')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.show()
else:
    print("Column 'category' missing — skipping category sales plot.")

# -----------------------------
# RFM analysis
# -----------------------------
if {'customer_id','order_date','total_amount'}.issubset(df.columns):
    snapshot_date = df['order_date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    rfm.columns = ['customer_id','recency','frequency','monetary']
    print(rfm.sort_values('monetary', ascending=False).head(10))

    fig, axes = plt.subplots(1,2, figsize=(14,5))
    sns.histplot(rfm['monetary'], ax=axes[0], bins=30, log_scale=(False, True))
    axes[0].set_title('Monetary Distribution (log scale y)')
    sns.histplot(rfm['frequency'], ax=axes[1], bins=30)
    axes[1].set_title('Frequency Distribution')
    plt.tight_layout()
    plt.show()
else:
    print("Columns 'customer_id', 'order_date', or 'total_amount' missing — skipping RFM analysis.")

# -----------------------------
# Payment method popularity
# -----------------------------
if 'payment_method' in df.columns and not df['payment_method'].isnull().all():
    pm = df['payment_method'].value_counts().reset_index()
    pm.columns = ['payment_method','count']
    print(pm)
    plt.figure(figsize=(8,4))
    order_values = pm['payment_method'].tolist() if not pm.empty else None
    sns.countplot(data=df, y='payment_method', order=order_values)
    plt.title('Orders by Payment Method')
    plt.tight_layout()
    plt.show()
else:
    print("Column 'payment_method' not found — skipping payment method analysis.")

# -----------------------------
# Sales by country
# -----------------------------
if 'country' in df.columns and not df['country'].isnull().all():
    country_sales = df.groupby('country')['total_amount'].sum().sort_values(ascending=False)
    print(country_sales.reset_index().rename(columns={'total_amount':'total_sales'}))
    plt.figure(figsize=(8,5))
    sns.barplot(x=country_sales.values, y=country_sales.index)
    plt.title('Sales by Country')
    plt.xlabel('Total Sales')
    plt.tight_layout()
    plt.show()
else:
    print("Column 'country' missing — skipping country sales plot.")

# -----------------------------
# Correlation matrix
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation matrix — numeric features')
    plt.show()
else:
    print('Not enough numeric columns for correlation matrix:', num_cols)

# -----------------------------
# Save cleaned dataset
# -----------------------------
cleaned_path = 'ecommerce_data_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print('Cleaned dataset saved to', cleaned_path)

# -----------------------------
# Suggested insights
# -----------------------------
print('\n---\nSuggested insights to report:')
print('- Monthly sales trend and seasonality')
print('- Top products and categories driving revenue')
print('- High-value customers (top monetary) and retention focus')
print('- Payment method and country-wise performance')
print('- Any data quality issues noticed (missing/incorrect dates, duplicate orders)')
