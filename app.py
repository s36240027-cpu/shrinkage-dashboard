# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, confusion_matrix

st.set_page_config(
    page_title="Inventory Shrinkage Dashboard",
    layout="wide"
)

st.title("📦 Inventory Shrinkage Analytics & Risk Classification")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_csv("lp_shrinkage_project_data.csv")

df = load_data()
df["date"] = pd.to_datetime(df["date"])

# ======================
# PREPROCESSING (MATCH NOTEBOOK)
# ======================

# Create label (fixed threshold)
df["High_Risk"] = (df["shrinkage"] > 400).astype(int)

# One-hot encoding categorical features
cat_cols = ["store_id", "department"]
df_ohe = pd.get_dummies(df[cat_cols], prefix=cat_cols)

df_final = pd.concat(
    [df.drop(columns=cat_cols), df_ohe],
    axis=1
)

# Feature & target
X = df_final.drop(columns=["shrinkage", "date", "High_Risk"])
y = df_final["High_Risk"]

# ======================
# KPI
# ======================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric(
    "High Risk (%)",
    f"{filtered_df['High_Risk'].mean() * 100:.1f}%"
)
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
    ax.axvline(400, linestyle="--", label="Threshold 400")
    ax.legend()
    ax.set_title("Shrinkage Distribution")
    st.pyplot(fig)

with colB:
    risk_by_dept = filtered_df.groupby("department")["High_Risk"].mean()
    fig, ax = plt.subplots()
    risk_by_dept.plot(kind="bar", ax=ax)
    ax.set_title("High Risk Ratio by Department")
    st.pyplot(fig)

# ======================
# MODEL (NO SCALING – CORRECT)
# ======================
st.subheader("🤖 Shrinkage Risk Classification")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ======================
# EVALUATION (OPTION C)
# ======================
threshold = 0.37

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= threshold).astype(int)

recall = recall_score(y_test, preds)
st.info(f"Model Recall (threshold={threshold}): {recall:.2f}")

# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_test, preds)
st.write("Confusion Matrix")
st.dataframe(
    pd.DataFrame(
        cm,
        index=["Actual Low", "Actual High"],
        columns=["Pred Low", "Pred High"]
    )
)

# ======================
# PREDICTION INPUT (FIXED)
# ======================
st.subheader("🧪 Try Risk Prediction")

c1, c2, c3 = st.columns(3)
sales = c1.number_input("Sales", min_value=0)
returns = c2.number_input("Returns", min_value=0)
inventory = c3.number_input("Inventory", min_value=0)

promo = st.selectbox("Promo Active?", [0, 1])
staff = st.slider("Staff on Duty", 1, 20, 5)

store_input = st.selectbox(
    "Store ID",
    sorted(df["store_id"].unique())
)

department_input = st.selectbox(
    "Department",
    sorted(df["department"].unique())
)

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        "sales": sales,
        "returns": returns,
        "inventory": inventory,
        "promo": promo,
        "staff_on_duty": staff,
        "store_id": store_input,
        "department": department_input
    }])

    input_ohe = pd.get_dummies(
        input_df[["store_id", "department"]],
        prefix=["store_id", "department"]
    )

    input_final = pd.concat(
        [input_df.drop(columns=["store_id", "department"]), input_ohe],
        axis=1
    )

    input_final = input_final.reindex(
        columns=X.columns,
        fill_value=0
    )

    proba_input = model.predict_proba(input_final)[0, 1]

    if proba_input >= threshold:
        st.error(f"🚨 HIGH SHRINKAGE RISK (prob={proba_input:.2f})")
    else:
        st.success(f"✅ LOW SHRINKAGE RISK (prob={proba_input:.2f})")
