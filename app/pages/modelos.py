import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from back.models import load_artifacts
from back.chart_models import confusion_matrix_plot, roc_curve_plot



model, y_test, y_pred, y_proba, metrics,  X_train, X_test = load_artifacts()

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

st.set_page_config(
    page_title="Credit Risk Model",
    layout="wide"
)

st.title(" Credit Risk Model & Explainability")

st.header(" Model Overview")

st.markdown("""
**Objective:** Predict probability of loan default.

**Model Type:** Supervised binary classification.

**Target:** `loan_status` (1 = Default, 0 = Non-default)

**Key Features:**
- Credit score & credit maturity
- Financial stress indicators
- Income and debt ratios
""")

st.header(" Model Performance")

threshold = st.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2f}")

with col2:
    st.metric("F1 Score", f"{metrics['f1_score']:.2f}")

with col3:
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba):.2f}")

#ROC
st.subheader("ROC Curve")

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
st.pyplot(fig)


# CONFUSION MATRIX

st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, ax=ax, cmap="Blues"
)
st.pyplot(fig)

st.markdown("""
📌 **Business note:**  
Threshold can be adjusted to prioritize capturing high-risk customers, 
reducing financial losses at the cost of rejecting more loans.
""")


# SHAP

st.header("🧠 Model Explainability (SHAP)")

@st.cache_resource
def load_shap_explainer():
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return shap_values

shap_values = load_shap_explainer()

# GLOBAL IMPORTANCE

st.subheader("Global Feature Importance")

fig = plt.figure()
shap.plots.bar(shap_values, max_display=10, show=False)
st.pyplot(fig)

st.markdown("""
The model relies primarily on **financial stress indicators**, 
**credit score**, and **income-related ratios**, confirming consistency
with the exploratory data analysis.
""")


# BEESWARM

st.subheader("Global Feature Impact")

fig = plt.figure()
shap.plots.beeswarm(shap_values, max_display=10, show=False)
st.pyplot(fig)

# this two plots are not in because financial_stress_score is not in the dataset that was trained
# but can be added back if needed
# if needed to add back, make sure to run back/feature_engineering.py in training.py to include financial_stress_score feature

#st.subheader("Key Feature Relationships")

#col1, col2 = st.columns(2)
#feature_idx = shap_values.feature_names.index("financial_stress_score")
#with col1:
#    fig = plt.figure()
#    shap.plots.scatter(
#        shap_values[:, feature_idx],
        
#    )
#    st.pyplot(fig)

#with col2:
#    fig = plt.figure()
#    shap.plots.scatter(
#        shap_values[:, "credit_score"],
#        color=shap_values[:, "financial_stress_score"],
#        show=False
#    )
#    st.pyplot(fig)


st.header(" Individual Customer Explanation")
#this can be changed using st.number_input too (with the col ID that was taknen from the dataset)
idx = st.slider(
    "Select customer index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0
)

prob = y_proba[idx]

st.metric("Default Probability", f"{prob:.2%}")

fig = plt.figure()
shap.plots.waterfall(shap_values[idx], show=False)
st.pyplot(fig)

st.markdown("""
This explanation shows how each feature contributed to the individual prediction,
supporting transparent and auditable credit decisions.
""")


st.header("💰 Business Impact (Preview)")

expected_loss = st.number_input(
    "Average Loss per Default ($)",
    value=5000
)

avoided_defaults = np.sum((y_proba >= threshold) & (y_test == 1))
estimated_savings = avoided_defaults * expected_loss

st.metric(
    "Estimated Avoided Loss",
    f"${estimated_savings:,.0f}"
)

st.markdown("""
This is a simplified estimate to illustrate the potential financial impact.
A full simulation is available in the Business Impact section.
""")
