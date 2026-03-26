import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="B2B Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>B2B Client Risk & Churn Dashboard</h1>",
    unsafe_allow_html=True
)

# ----------------------------
# LOAD DATA (CSV FIXED)
# ----------------------------
@st.cache_data
def load_data():
    file_path = "b2b_pricing_dataset_1000_rows.csv"
    
    if not os.path.exists(file_path):
        st.error("❌ CSV file not found. Upload it to GitHub.")
        st.stop()
        
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    
    return df

df = load_data()

st.success("Dataset Loaded Successfully")

# ----------------------------
# AUTO DETECT COLUMNS
# ----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ----------------------------
# RISK SCORE (DYNAMIC)
# ----------------------------
if len(numeric_cols) >= 3:
    df["Risk_Score"] = (
        df[numeric_cols[0]] * 0.4 +
        df[numeric_cols[1]] * 0.3 +
        df[numeric_cols[2]] * 0.3
    )
else:
    st.error("Not enough numeric columns for risk calculation")
    st.stop()

# Risk Category
df["Risk_Category"] = pd.qcut(
    df["Risk_Score"], q=3,
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔍 Filters")

filtered_df = df.copy()

for col in cat_cols[:2]:
    selected = st.sidebar.multiselect(col, df[col].unique())
    if selected:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

risk_filter = st.sidebar.multiselect(
    "Risk Category", df["Risk_Category"].unique()
)

if risk_filter:
    filtered_df = filtered_df[filtered_df["Risk_Category"].isin(risk_filter)]

# ----------------------------
# KPI CARDS
# ----------------------------
st.subheader("📊 Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Clients", len(filtered_df))
c2.metric("High Risk Clients",
          len(filtered_df[filtered_df["Risk_Category"]=="High Risk"]))
c3.metric("Average Risk Score",
          round(filtered_df["Risk_Score"].mean(),2))

# ----------------------------
# MACHINE LEARNING
# ----------------------------
if "Renewal_Status" in df.columns:

    df_ml = df.copy()

    if df_ml["Renewal_Status"].dtype == "object":
        df_ml["Renewal_Status"] = df_ml["Renewal_Status"].map({"Yes":1,"No":0})

    X = df_ml[numeric_cols]
    y = df_ml["Renewal_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    c4.metric("Model Accuracy", str(round(accuracy*100,2)) + "%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Feature Importance")
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(imp)

else:
    c4.metric("Model Accuracy", "N/A")

# ----------------------------
# VISUALS
# ----------------------------
st.subheader("📊 Risk Distribution")

fig1, ax1 = plt.subplots()
filtered_df["Risk_Category"].value_counts().plot(kind="bar", ax=ax1)
st.pyplot(fig1)

st.subheader("📈 Risk vs Feature")

fig2, ax2 = plt.subplots()
ax2.scatter(filtered_df[numeric_cols[0]], filtered_df["Risk_Score"])
ax2.set_xlabel(numeric_cols[0])
ax2.set_ylabel("Risk Score")
st.pyplot(fig2)

# ----------------------------
# TOP CLIENTS
# ----------------------------
st.subheader("🔥 Top 20 High Risk Clients")

top20 = filtered_df.sort_values(
    by="Risk_Score", ascending=False).head(20)

st.dataframe(top20)

# ----------------------------
# RETENTION STRATEGY
# ----------------------------
st.subheader("💡 AI Retention Strategy")

if st.button("Generate Strategy"):
    st.success("Recommended Actions:")
    st.write("• Offer discounts to high-risk clients")
    st.write("• Improve engagement")
    st.write("• Provide contract incentives")
    st.write("• Assign account managers")

# ----------------------------
# ETHICS
# ----------------------------
st.subheader("⚖️ Ethical Considerations")

st.write("""
• Bias in predictions may affect clients  
• Risk labeling must be used carefully  
• Ensure data privacy  
• AI supports decisions, not replaces them  
""")
