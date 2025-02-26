import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("sales_data.csv")

# Clean the data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract month and year
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Group sales by month
monthly_sales = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()

# Top selling products
top_products = df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)

# Profit by region
region_sales = df.groupby('Region')['Profit'].sum().reset_index()

# Create subplots to display all charts simultaneously
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

# Chart 1: Monthly Sales Trend
sns.lineplot(x=monthly_sales.index, y=monthly_sales['Sales'], marker='o', ax=axes[0])
axes[0].set_title('Monthly Sales Trend')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Total Sales')

# Chart 2: Top 5 Selling Products
sns.barplot(x=top_products.index, y=top_products.values, ax=axes[1])
axes[1].set_title('Top 5 Selling Products')
axes[1].set_xlabel('Product')
axes[1].set_ylabel('Total Sales')

# Chart 3: Profit by Region
sns.barplot(x='Region', y='Profit', data=region_sales, palette="coolwarm", ax=axes[2])
axes[2].set_title('Profit by Region')
axes[2].set_xlabel('Region')
axes[2].set_ylabel('Total Profit')

# Adjust layout and show all charts
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare data for prediction
df['Month'] = df['Date'].dt.month
X = df[['Month']]
y = df['Sales']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales
future_sales = model.predict([[6]])  # Predict for June
print(f"Predicted Sales for June: {future_sales[0]}")



