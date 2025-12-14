# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score

st.set_page_config(
    page_title="Inventory Shrinkage Dashboard",
    layout="wide"
)

st.title("📦 Inventory Shrinkage Analytics & Risk Classification")

@st.cache_data
def load_data():
    return pd.read_csv("lp_shrinkage_project_data.csv")

df = load_data()
df["date"] = pd.to_datetime(df["date"])

# ======================
# CREATE CLASS LABEL
# ======================
threshold = df["shrinkage"].median()
df["risk_label"] = (df["shrinkage"] > threshold).astype(int)
# 0 = Low Risk, 1 = High Risk

# ======================
# SIDEBAR FILTER
# ======================
st.sidebar.header("🔍 Filter Data")

store = st.sidebar.selectbox(
    "Select Store",
    ["All"] + sorted(df["store_id"].unique())
)

department = st.sidebar.selectbox(
    "Select Department",
    ["All"] + sorted(df["department"].unique())
)

filtered_df = df.copy()
if store != "All":
    filtered_df = filtered_df[filtered_df["store_id"] == store]
if department != "All":
    filtered_df = filtered_df[filtered_df["department"] == department]

# ======================
# KPI
# ======================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric("High Risk (%)",
            f"{filtered_df['risk_label'].mean() * 100:.1f}%")
col3.metric("Total Sales", f"{filtered_df['sales'].sum():,}")
col4.metric("Avg Inventory", f"{filtered_df['inventory'].mean():.0f}")

# ======================
# VISUALIZATION
# ======================
st.subheader("📈 Data Visualization")

colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots()
    ax.hist(filtered_df["shrinkage"], bins=20)
    ax.set_title("Shrinkage Distribution")
    st.pyplot(fig)

with colB:
    risk_by_dept = filtered_df.groupby("department")["risk_label"].mean()
    fig, ax = plt.subplots()
    risk_by_dept.plot(kind="bar", ax=ax)
    ax.set_title("High Risk Ratio by Department")
    st.pyplot(fig)

# ======================
# MODEL (CLASSIFICATION)
# ======================
st.subheader("🤖 Shrinkage Risk Classification")

features = ["sales", "returns", "inventory", "promo", "staff_on_duty"]
X = df[features]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
recall = recall_score(y_test, preds)

st.info(f"Model Recall: {recall:.2f}")

# ======================
# PREDICTION INPUT
# ======================
st.subheader("🧪 Try Risk Prediction")

c1, c2, c3 = st.columns(3)
sales = c1.number_input("Sales", min_value=0)
returns = c2.number_input("Returns", min_value=0)
inventory = c3.number_input("Inventory", min_value=0)

promo = st.selectbox("Promo Active?", [0, 1])
staff = st.slider("Staff on Duty", 1, 20, 5)

if st.button("Predict Risk"):
    pred = model.predict([[sales, returns, inventory, promo, staff]])[0]

    if pred == 1:
        st.error("🚨 HIGH SHRINKAGE RISK")
    else:
        st.success("✅ LOW SHRINKAGE RISK")
