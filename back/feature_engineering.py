import pandas as pd
import numpy as  np

def feature_engineering(df):
    df = df.copy()
    df['financial_stress_score'] = (df['debt_to_income_ratio'] * 0.5 + 
                                    df['loan_to_income_ratio'] * 0.3 + df['payment_to_income_ratio'] * 0.2)

    df['has_credit_issue'] = ((df['defaults_on_file'] > 0) | (df['derogatory_marks'] > 0)).astype(int) 
    # not putting deliquancies, but can be added later if needed

    df['savings_rate'] = df['savings_assets'] / df['annual_income']
    df['credity_maturity'] = df['credit_history_years'] / df['age']

    for col in ["annual_income", "loan_amount", "current_debt", "savings_assets"]:
        df[f"log_{col}"] = np.log1p(df[col])

    df['income_band'] = pd.qcut(df['annual_income'], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])
    df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 50, 65, 100], labels=["18-25", "26-35", "36-45", "46-60", "60+"])
    df['score_group'] = pd.cut(df['credit_score'], bins=[300, 580, 670, 740, 800, 850], labels=["Poor/Subprime", "Fair", "Good", "Very Good", "Exceptional/Excellent"])

    ''' Typical Score Ranges (FICO/VantageScore)
    Exceptional/Excellent: 800-850
    Very Good: 740-799
    Good: 670-739
    Fair: 580-669
    Poor/Subprime: 300-579 '''

    return df