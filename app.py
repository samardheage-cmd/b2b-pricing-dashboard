import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(page_title="B2B Risk Dashboard", layout="wide")

# --------------------------
# TITLE
# --------------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>B2B Client Risk & Churn Intelligence Dashboard</h1>",
    unsafe_allow_html=True
)

# --------------------------
# LOAD DATA
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("B2B_Client_Churn_5000.csv")
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df

df = load_data()

# --------------------------
# RISK SCORE LOGIC
# --------------------------
df["Risk_Score"] = (
    df["Payment_Delay_Days"] * 0.4 +
    (100 - df["Monthly_Usage"]) * 0.3 +
    (12 - df["Contract_Length"]) * 0.2 +
    df["Support_Tickets"] * 0.1
)

def risk_category(score):
    if score < 40:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "High Risk"

df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# --------------------------
# SIDEBAR FILTERS
# --------------------------
st.sidebar.header("🔍 Filter Data")

region = st.sidebar.multiselect("Region", df["Region"].unique())
industry = st.sidebar.multiselect("Industry", df["Industry"].unique())
risk = st.sidebar.multiselect("Risk Category", df["Risk_Category"].unique())

filtered_df = df.copy()

if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]
if industry:
    filtered_df = filtered_df[filtered_df["Industry"].isin(industry)]
if risk:
    filtered_df = filtered_df[filtered_df["Risk_Category"].isin(risk)]

# --------------------------
# KPI CARDS
# --------------------------
st.markdown("### 📊 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Clients", len(filtered_df))
c2.metric("High Risk Clients", len(filtered_df[filtered_df["Risk_Category"]=="High Risk"]))
c3.metric("Avg Revenue", round(filtered_df["Revenue"].mean(),2))
c4.metric("Avg Risk Score", round(filtered_df["Risk_Score"].mean(),2))

# --------------------------
# MACHINE LEARNING
# --------------------------
df_ml = df.copy()
df_ml["Renewal_Status"] = df_ml["Renewal_Status"].map({"Yes":1,"No":0})

X = df_ml[["Monthly_Usage","Payment_Delay_Days","Contract_Length","Support_Tickets","Revenue"]]
y = df_ml["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train,y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test,pred)

st.markdown(f"### 🤖 Model Accuracy: **{round(accuracy*100,2)}%**")

# --------------------------
# VISUALS
# --------------------------
col1, col2 = st.columns(2)

# Risk Distribution
with col1:
    st.subheader("Risk Category Distribution")
    fig, ax = plt.subplots()
    filtered_df["Risk_Category"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# Industry Risk
with col2:
    st.subheader("Industry-wise Risk")
    fig, ax = plt.subplots()
    pd.crosstab(filtered_df["Industry"], filtered_df["Risk_Category"]).plot(kind="bar", ax=ax)
    st.pyplot(fig)

# Scatter
st.subheader("Revenue vs Risk Score")
fig, ax = plt.subplots()
ax.scatter(filtered_df["Revenue"], filtered_df["Risk_Score"])
ax.set_xlabel("Revenue")
ax.set_ylabel("Risk Score")
st.pyplot(fig)

# Contract vs Churn
st.subheader("Contract Length vs Churn")
fig, ax = plt.subplots()
ax.scatter(df["Contract_Length"], df_ml["Renewal_Status"])
st.pyplot(fig)

# --------------------------
# CONFUSION MATRIX
# --------------------------
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
cm = confusion_matrix(y_test,pred)
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# --------------------------
# FEATURE IMPORTANCE
# --------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.dataframe(importance)

# --------------------------
# TOP 20 CLIENTS
# --------------------------
st.subheader("Top 20 High Risk Clients")

top20 = filtered_df.sort_values(by="Risk_Score", ascending=False).head(20)
st.dataframe(top20)

# --------------------------
# RETENTION STRATEGY
# --------------------------
st.subheader("💡 AI-Based Retention Strategy")

if st.button("Generate Retention Strategy"):
    st.success("Recommended Actions:")
    st.write("• Offer discounts for clients with high payment delays")
    st.write("• Assign dedicated account managers")
    st.write("• Provide long-term contract incentives")
    st.write("• Improve engagement for low usage clients")
    st.write("• Prioritize high revenue clients for retention")

# --------------------------
# RESPONSIBLE AI
# --------------------------
st.subheader("⚖️ Ethical Implications")

st.write("""
• Bias in model predictions may affect certain industries unfairly  
• Labeling clients as 'High Risk' may impact business relationships  
• Ensure client data privacy and protection  
• Use AI as decision-support, not final authority  
""")
