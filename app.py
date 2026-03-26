import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("b2b_pricing_dataset_1000_rows.xlsx")

st.title("B2B Pricing & Profitability Dashboard")

# KPIs
total_revenue = df["Revenue"].sum()
total_profit = df["Profit"].sum()
profit_margin = (total_profit / total_revenue) * 100
avg_discount = df["Discount"].mean()

st.subheader("Key Performance Indicators")
st.write(f"Total Revenue: {total_revenue:,.2f}")
st.write(f"Total Profit: {total_profit:,.2f}")
st.write(f"Profit Margin: {profit_margin:.2f}%")
st.write(f"Average Discount: {avg_discount:.2%}")

# Profit by Product
st.subheader("Profit by Product")
product_profit = df.groupby("Product_Name")["Profit"].sum().head(10)
st.bar_chart(product_profit)

# Region-wise Profit
st.subheader("Region-wise Profit")
region_profit = df.groupby("Region")["Profit"].sum()
st.bar_chart(region_profit)

# Price vs Sales Trend
st.subheader("Price vs Units Sold")
fig, ax = plt.subplots()
ax.scatter(df["Selling_Price"], df["Units_Sold"])
ax.set_xlabel("Selling Price")
ax.set_ylabel("Units Sold")
st.pyplot(fig)
