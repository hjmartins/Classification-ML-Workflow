Here is the **professional, polished README in English**, including a clear description of the **problem being solved** (credit risk assessment):

---

# 📘 Lending Club — Exploratory Analysis + Credit Risk Classification

## 📌 **Overview**

This project performs a complete analysis and machine learning workflow using the **Lending Club Loan Data**.
It includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature engineering
* Encoding of categorical variables
* Handling class imbalance
* Training and comparing multiple classification models
* Interpreting the results

The final goal is to build a model capable of predicting the **loan status** and assessing the probability of **credit default**.

---

# ❗ **Problem Definition**

Financial institutions face significant challenges when granting loans, especially in distinguishing **low-risk** from **high-risk** applicants.
Incorrect classification can lead to:

* Financial losses from loan defaults
* Incorrect approval of risky applicants
* Rejection of good customers

This project addresses the question:

> **“Given a customer’s financial profile, employment information, credit score, and loan details, can we accurately predict their loan status?”**

The target variable is:

```
loan_status
```

The model aims to support **credit risk assessment** and improve decision-making in loan approval.

---

# 📂 **Project Structure**

```
📁 lending-club-loan-analysis
│
├── data/
│   └── lending_club.csv
│   └── loan_eda_cleaned.csv
│
├── notebooks/
│   └── 01_EDA_LendingClub.ipynb
│   └── main.ipynb
│
├── models/
│   └── best_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

# 🧼 **1. Data Cleaning & Preprocessing**

Main preprocessing steps include:

### ✔ Handling missing values

### ✔ Fixing incorrect data types

### ✔ Removing or capping outliers such as:

* `loan_amount`
* `annual_income`
* `dti`
* `interest_rate`

### ✔ Feature creation:

* Income-to-loan ratio
* Credit score grouping
* Risk categories

---

# 🔍 **2. Exploratory Data Analysis (EDA)**

The EDA includes:

* Univariate and bivariate visualizations
* Distribution analysis
* Category frequency plots
* Correlation matrix
* Boxplots for detecting outliers
* Relationship between:

  * loan purpose × loan status
  * employment length × default probability
  * interest rate × default
  * income × default

### **Key Insights**

* Lower-income groups show higher default likelihood
* Some loan purposes (e.g., debt consolidation) carry higher risk
* The dataset is **highly imbalanced**, requiring balancing techniques


---

# 🧪 **4. Categorical Encoding**

Three encoding strategies were tested:

### ✔ One-Hot Encoding

### ✔ Label Encoding (best for tree-based models)


---

# 🤖 **5. Trained Models**

Multiple algorithms were compared:

| Model               | Accuracy | ROC AUC  | F1-Score |
| ------------------- | -------- | -------- | -------- |
| Logistic Regression | ~        | ~        | ~        |
| Random Forest       | ~        | ~        | ~        |
| XGBoost             | **Best** | **Best** | **Best** |
| Gradient Boosting   | ~        | ~        | ~        |
| KNN                 | ~        | ~        | ~        |

> Exact metrics are available in the modeling notebook.

The models is stored in:

```
models/best_model.pkl
```

---

# 🧠 **6. Model Interpretation**

Using **SHAP values**, the notebook explains:

* Which features impact loan status prediction
* How each feature influences individual predictions

### Most important features:

* `interest_rate`
* `annual_income`
* `credit_score`
* `dti`
* `loan_amount`

---

# 🚀 **7. How to Run Locally**

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/lending-club-loan-analysis.git
cd lending-club-loan-analysis
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Open the notebooks

```
notebooks/01_EDA_LendingClub.ipynb
notebooks/02_Modeling_LendingClub.ipynb
```

---

# 📊 **8. Tech Stack**

* Python
* pandas, numpy
* matplotlib, seaborn, plotly
* scikit-learn
* xgboost
* imbalanced-learn
* shap
* Jupyter Notebook

---

# 📝 **9. Future Improvements**

* Deployment using FastAPI or Flask
* Interactive dashboard with Streamlit


---

# 🧑‍💼 **Author**

**Henrique Martins**
🔗 LinkedIn: *[https://www.linkedin.com/in/henrique-jos%C3%A9-dos-santos-martins-a14235236/](https://www.linkedin.com/in/henrique-jos%C3%A9-dos-santos-martins-a14235236/)*
📧 Email: *hjmartins88@gmail.com*
