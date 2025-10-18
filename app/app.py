# app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ───────────────────────────────────────────────
# Streamlit Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="ER Admission Risk Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────
# Load Model and Data
# ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Random Forest model from /models directory."""
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_rf.joblib")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    return joblib.load(model_path)


@st.cache_data
def load_data():
    """Load synthetic and processed datasets."""
    base_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    raw_path = os.path.join(base_dir, "synthetic_er_data.csv")
    x_path = os.path.join(base_dir, "processed", "X_test.csv")
    y_path = os.path.join(base_dir, "processed", "y_test.csv")

    if not all(os.path.exists(p) for p in [raw_path, x_path, y_path]):
        st.error("❌ One or more data files are missing in the /data folder.")
        st.stop()

    raw = pd.read_csv(raw_path, parse_dates=["arrival_time"])
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).squeeze()
    return raw, X_test, y_test


# Load assets
model = load_model()
raw, X_test, y_test = load_data()

# ───────────────────────────────────────────────
# App Header
# ───────────────────────────────────────────────
st.title("🏥 Emergency Room Admission Risk Prediction")

st.markdown("""
This interactive app predicts the **probability that a patient arriving at the Emergency Department will be admitted** to the hospital.

It uses demographics, vitals, triage score, and wait time — evaluated through a trained Random Forest model.
Fill in patient details on the left and click **Predict** to see the admission probability and visual analytics.
""")

# ───────────────────────────────────────────────
# Sidebar Input Form
# ───────────────────────────────────────────────
st.sidebar.header("👤 Patient Details")

with st.sidebar.form("patient_form"):
    age = st.number_input("Age", 0, 120, 50)
    hr = st.number_input("Heart Rate", 30, 200, 85)
    bp_sys = st.number_input("Systolic BP", 50, 200, 130)
    bp_dia = st.number_input("Diastolic BP", 30, 120, 85)
    triage = st.slider("Triage Score", 1, 5, 3)
    wait = st.number_input("Wait Time (min)", 0, 120, 15)
    arrival_hour = st.slider("Arrival Hour (0–23)", 0, 23, 12)
    visit_reason = st.selectbox("Visit Reason", ["Chest Pain", "Fever", "Injury", "Shortness of Breath", "Dizziness"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    submit = st.form_submit_button("▶️ Predict")

if not submit:
    st.info("Enter patient details in the sidebar and click **Predict**.")
    st.stop()

# ───────────────────────────────────────────────
# Prediction Logic
# ───────────────────────────────────────────────
bp_delta = bp_sys - bp_dia

patient = pd.DataFrame([{
    "age": age,
    "arrival_hour": arrival_hour,
    "bp_delta": bp_delta,
    "high_hr_flag": int(hr > 100),
    "triage_urgent": int(triage <= 2),
    "visit_count": 1,
    "frequent_visitor_flag": 0,
    **{f"reason_{r}": int(r == visit_reason)
       for r in ["Chest Pain", "Fever", "Injury", "Shortness of Breath", "Dizziness"]},
    "gender_Male": int(gender == "Male"),
    "wait_cat_medium": int(10 <= wait <= 30),
    "wait_cat_long": int(wait > 30),
}])

patient = patient.reindex(columns=X_test.columns, fill_value=0)
proba = model.predict_proba(patient)[:, 1][0]
label = "Admit" if proba >= 0.5 else "No Admit"

# ───────────────────────────────────────────────
# Display Prediction
# ───────────────────────────────────────────────
st.subheader("🎯 Admission Probability")
st.metric(label=label, value=f"{proba:.1%}")

# ───────────────────────────────────────────────
# 1️⃣ Probability Distribution
# ───────────────────────────────────────────────
st.subheader("Population Admission Probability Distribution")
probs = model.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(probs, bins=30, kde=True, ax=ax, color="skyblue")
ax.axvline(proba, color="red", linestyle="--", label="Your Patient")
ax.set_xlabel("Predicted Admission Probability")
ax.legend()
st.pyplot(fig)

# ───────────────────────────────────────────────
# 2️⃣ Feature Comparison
# ───────────────────────────────────────────────
st.subheader("Feature Comparison")
fig2, axes = plt.subplots(2, 2, figsize=(9, 6))

compare = [
    ("age", age, "Age"),
    ("bp_delta", bp_delta, "BP Delta"),
    ("high_hr_flag", int(hr > 100), "High HR Flag"),
    ("wait_cat_long", int(wait > 30), "Long Wait Flag"),
]

for ax, (col, val, xlbl) in zip(axes.flatten(), compare):
    sns.histplot(X_test[col], bins=20, ax=ax, color="lightgray")
    ax.axvline(val, color="red", linestyle="--")
    ax.set_xlabel(xlbl)

plt.tight_layout()
st.pyplot(fig2)

# ───────────────────────────────────────────────
# 3️⃣ Model Evaluation
# ───────────────────────────────────────────────
st.subheader("Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix (0.5 Threshold)**")
    cm = confusion_matrix(y_test, (probs >= 0.5).astype(int))
    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

with col2:
    st.markdown("**Top 10 Feature Importances**")
    fi = pd.Series(model.feature_importances_, index=X_test.columns)
    fi = fi.sort_values(ascending=False).head(10)
    fig_fi, ax_fi = plt.subplots(figsize=(4, 4))
    fi.plot.barh(ax=ax_fi, color="steelblue")
    ax_fi.invert_yaxis()
    ax_fi.set_xlabel("Importance")
    st.pyplot(fig_fi)
