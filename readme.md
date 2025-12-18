# 🏦 Credit Risk Intelligence Dashboard

An end-to-end **credit risk analysis and decision-support system** that combines **machine learning** and **financial impact simulation** to enable **transparent and business-driven lending decisions**.

---

## 📌 Project Overview

Financial institutions face a constant trade-off between **risk mitigation** and **customer approval**.
Approving high-risk loans leads to financial losses, while overly conservative credit policies reduce growth.

This project demonstrates how **data science and explainable machine learning** can be used to:

* Predict loan default risk
* Explain model decisions transparently
* Quantify financial impact through interactive simulations

---

## 🎯 Objectives

* Perform structured **Exploratory Data Analysis (EDA)**
* Engineer meaningful **risk-related features**
* Train and evaluate a **credit default prediction model**
* Apply **SHAP** for model explainability
* Translate predictions into **financial outcomes**

---

## 🧠 Solution Architecture

```
EDA
 ├─ Data understanding & feature engineering
 └─ Risk segmentation

Modeling
 ├─ Training & evaluation
 └─ Threshold-based decision logic

Explainability
 ├─ Global feature importance (SHAP)
 ├─ Feature impact & interactions
 └─ Individual predictions

Business Impact
 ├─ Financial simulation
 ├─ Threshold optimization
 └─ Estimated savings analysis
```

---

##  Application Pages

###  Home

Project overview, context, solution explanation and navigation guide.

###  EDA

* Distribution analysis
* Bivariate and multivariate relationships
* Engineered features such as:

  * Financial stress score
  * Savings rate
  * Credit maturity
  * Income and score bands

###  Model & Explainability

* Model performance metrics (ROC, Confusion Matrix)
* Adjustable decision threshold
* SHAP explainability:

  * Global feature importance
  * Feature impact (beeswarm)
  * Feature interactions
  * Individual prediction explanations

###  Business Impact

* Interactive financial simulation
* User-defined cost per default
* Threshold-based savings estimation
* Approval rate vs risk trade-off analysis

---

##  Feature Engineering Highlights

* **Financial Stress Score**
  Weighted combination of debt and income ratios

* **Credit Maturity**
  Credit history normalized by age

* **Savings Rate**
  Savings relative to income

* **Log-transformed financial variables**
  To handle skewed distributions

* **Risk segmentation bands**
  Income, age and credit score groups

---

##  Explainability (SHAP)

The project applies SHAP to ensure:

* Transparency in model decisions
* Alignment between EDA insights and model behavior
* Explainable decisions at both **global** and **individual** levels

This enables trust, accountability and regulatory-friendly analysis.

---

##  Business-Oriented Decision Making

Rather than optimizing only technical metrics, the system focuses on:

* **Loss prevention**
* **Approval rate optimization**
* **Threshold-based credit policies**
* **Estimated financial savings**

The Business Impact page translates model outputs into **monetary value**.

---

##  Tech Stack

* **Python**

  * pandas, numpy
  * scikit-learn
* **Visualization**

  * Plotly
  * Streamlit
* **Explainable AI**

  * SHAP
* **Modeling**

  * Classification models (logistic / tree-based)

---

##  How to Run the App

```bash
# create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run streamlit app
streamlit run app.py
```

---

##  Project Structure

```
machine_learning/
│
├── app/
│   ├── app.py
│   └── pages/
│       ├── eda.py
│       ├── modelos.py
│       └── business_impact.py
│
├── back/
│   ├── loan_eda.py
│   └── models.py
│
├── artifacts/
│   ├── model.pkl
│   ├── X_train.pkl
│   ├── X_test.pkl
│   └── y_test.pkl
│
└── README.md
```

---

##  Key Takeaways

* Credit risk can be **modeled, explained and optimized**
* Explainability bridges the gap between data science and business
* Optimal thresholds maximize **financial efficiency**, not just accuracy
* Data-driven decisions reduce losses while preserving growth

---

##  Author

**Henrique Martins**
Bachelor in Computer Science
Experience in Data Analysis, Machine Learning and Data Visualization

---

##  Disclaimer

This project is for **educational and portfolio purposes** and does not represent a real financial institution’s production system.



