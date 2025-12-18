import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix
from back.models import load_artifacts


model, y_test, y_pred, y_proba, metrics,  X_train, X_test = load_artifacts()
#can be changed for that
# model, y_test, _, _, _, _, X_test,  = load_artifacts() 

st.title(" Business Impact Analysis")
st.markdown("""
This section translates model performance into **financial impact**, 
supporting data-driven credit policy decisions.
""")

# ======================================
# side bar simulation parameters
st.sidebar.header("Simulation Parameters")

cost_per_default = st.sidebar.number_input(
    " Cost per Default (R$)",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
)

threshold = st.sidebar.slider(
    " Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
)

# ======================================
# PAGE CONTENT
# this page can be more cleaned up, in the future, by moving some calculations to back/.py
probas = model.predict_proba(X_test)[:, 1]
y_pred = (probas >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

loss_without_model = y_test.sum() * cost_per_default
loss_with_model = fn * cost_per_default
savings = loss_without_model - loss_with_model

approval_rate = (y_pred == 0).mean()
default_capture_rate = tp / (tp + fn)


st.subheader(" Key Financial Metrics")

col1, col2, col3 = st.columns(3)
col1.metric(" Estimated Savings", f"R$ {savings:,.0f}")
col2.metric(" Default Capture Rate", f"{default_capture_rate:.2%}")
col3.metric(" Approval Rate", f"{approval_rate:.2%}")


st.subheader(" Decision Breakdown")

cm_df = pd.DataFrame(
    {
        "Outcome": ["True Negatives", "False Positives", "False Negatives", "True Positives"],
        "Count": [tn, fp, fn, tp],
        "Financial Impact (R$)": [
            0,
            0,
            -fn * cost_per_default / max(fn, 1),
            tp * cost_per_default / max(tp, 1),
        ],
    }
)

st.dataframe(cm_df, use_container_width=True)


st.subheader(" Savings vs Threshold")

results = []

for t in np.arange(0.1, 0.9, 0.05):
    preds = (probas >= t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, preds).ravel()
    savings_ = (y_test.sum() - fn_) * cost_per_default

    results.append({
        "threshold": t,
        "savings": savings_,
        "approval_rate": (preds == 0).mean(),
        "default_capture": tp_ / (tp_ + fn_)
    })

results_df = pd.DataFrame(results)

fig = px.line(
    results_df,
    x="threshold",
    y="savings",
    title="Estimated Savings by Decision Threshold",
    markers=True,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader(" Business Interpretation")

st.markdown(f"""
- At a threshold of **{threshold:.2f}**, the model captures **{default_capture_rate:.1%}** of defaults.
- Estimated savings reach **R$ {savings:,.0f}**, compared to no model usage.
- Increasing the threshold reduces approvals but increases loss prevention.
""")
