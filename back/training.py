
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from pathlib import Path
#from feature_engineering import preprocess_data


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(
    BASE_DIR / "credit_loan" / "loan_eda_cleaned.csv"
)

df = df.drop(columns=["customer_id"], errors="ignore")

#if needed perform preprocessing
#caution with the news features created in feature_engineering.py they can turn the model too optimistic
# myadvice is to comment this line unless you want to test different preprocessing techniques or add new features
# financial_stress_score and the logs one can be a good add but the other ones may lead to data leakage
#df = preprocess_data(df)


TARGET = "loan_status"


cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
}

results = []

for name, model in models.items():
    if name in ["Logistic Regression", "SVM", "Naive Bayes"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append((name, acc, f1, model, y_pred, y_proba))


# see and save the BEST MODEL

best = sorted(results, key=lambda x: x[2], reverse=True)[0]

best_name, acc, f1, best_model, y_pred, y_proba = best

joblib.dump(best_model, ARTIFACTS_DIR / "model.pkl")

pd.Series(y_test).to_csv(ARTIFACTS_DIR / "y_test.csv", index=False)
pd.Series(y_pred).to_csv(ARTIFACTS_DIR / "y_pred.csv", index=False)
pd.Series(y_proba).to_csv(ARTIFACTS_DIR / "y_proba.csv", index=False)
if best_name in ["Logistic Regression", "SVM", "Naive Bayes"]:
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(ARTIFACTS_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(ARTIFACTS_DIR / "X_test.csv", index=False)
else:
    pd.DataFrame(X_train).to_csv(ARTIFACTS_DIR / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(ARTIFACTS_DIR / "X_test.csv", index=False)

with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(
        {"model": best_name, "accuracy": acc, "f1_score": f1},
        f,
        indent=4,
    )

print(f"Best model: {best_name}")
