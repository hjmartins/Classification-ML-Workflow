import streamlit as st
import sys
import os

# Ensure project root (machine_learning/) is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

st.title(" Credit Risk Intelligence Dashboard")

st.markdown("""
### Data-driven, explainable and business-oriented credit risk analysis
""")

st.divider()

st.markdown("""
##  Context & Problem

Financial institutions constantly balance **risk mitigation** and **business growth**.
Approving risky loans increases default losses, while conservative policies reduce revenue.

This dashboard demonstrates how **machine learning and explainable AI** can support
**better, transparent and financially optimized credit decisions**.
""")

st.markdown("""
##  Solution Overview

This project delivers an **end-to-end credit risk solution**, including:

- Data exploration and feature engineering  
- Predictive modeling  
- Explainability with SHAP  
- Business impact simulation  
""")

st.markdown("""
##  What You’ll Find in This App

###  Exploratory Data Analysis (EDA)
Understand customer behavior, risk distribution and financial patterns.

###  Model & Explainability
Evaluate model performance and understand *why* decisions are made.

###  Business Impact
Simulate financial outcomes and optimize decision thresholds.
""")

st.markdown("""
##  Tech Stack

- Python (pandas, numpy, scikit-learn)
- Streamlit & Plotly
- SHAP Explainability
""")

st.markdown("""
##  How to Navigate

Use the sidebar to access:
-  **EDA**
-  **Model**
-  **Business Impact**
""")

st.markdown("""
##  Key Takeaways

- Credit risk can be **modeled and explained**
- Explainability increases **trust and accountability**
- Optimal thresholds maximize **financial efficiency**
""")
