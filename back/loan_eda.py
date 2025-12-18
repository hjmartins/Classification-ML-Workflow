# eda_loan.py
import pandas as pd
import numpy as np
import plotly.express as px
from back.feature_engineering import feature_engineering
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
### this page is for exploratory data analysis (EDA) functions ###
def load_data():
    df = pd.read_csv(BASE_DIR / "credit_loan" / "loan_eda_cleaned.csv")

    return feature_engineering(df)

def hist_plot(df, column, title=None, nbins=40):
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title or f"Distribution of {column}",
    )
    fig.update_layout(bargap=0.05)
    return fig


def count_plot(df, column, title=None):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]

    fig = px.bar(
        counts,
        x=column,
        y="count",
        title=title or f"{column} Distribution",
        text_auto=True,
    )
    return fig

def rate_by_category(df, category, target="loan_status", title=None):
    rates = (
        df.groupby(category)[target]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        rates,
        x=category,
        y=target,
        title=title or f"Default Rate by {category}",
        text_auto=".2%",
    )
    return fig
def box_plot(df, x, y, title=None):
    fig = px.box(
        df,
        x=x,
        y=y,
        title=title or f"{y} Distribution by {x}",
    )
    return fig
def density_relationship(df, x, y, title=None):
    fig = px.density_heatmap(
        df,
        x=x,
        y=y,
        nbinsx=30,
        nbinsy=30,
        title=title or f"{x} vs {y} (Density)",
    )
    return fig

def risk_heatmap(df, x, y, target="loan_status", title=None):
    df[y] = pd.qcut(
    df["financial_stress_score"],
    q=5,
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
    )
    agg = (
        df.groupby([x, y])[target]
        .mean()
        .reset_index()
    )

    fig = px.density_heatmap(
        agg,
        x=x,
        y=y,
        z=target,
        histfunc="avg",
        title=title or f"Default Rate: {x} vs {y}",
    )
    return fig

def correlation_heatmap(df):
    features = [
        "credit_score",
        "financial_stress_score",
        "savings_rate",
        "debt_to_income_ratio",
        "loan_to_income_ratio",
    ]

    fig = px.imshow(
        df[features].corr(),
        text_auto=".2f",
        aspect="auto",
        title="Correlation Matrix (Selected Features)",
        color_continuous_scale="RdBu",
    )
    return fig
def univariate_figures(df):
    return {
        "Loan Status Distribution": count_plot(df, "loan_status"),
        "Log Annual Income": hist_plot(df, "log_annual_income", nbins=50),
        "Log Loan Amount": hist_plot(df, "log_loan_amount", nbins=50),
        "Credit Score Distribution": hist_plot(df, "credit_score"),
    }



def bivariate_figures(df):
    return {
        
        "Financial Stress vs Default": box_plot(df, "loan_status", "financial_stress_score"),
        "Savings Rate vs Default": box_plot(df, "loan_status", "savings_rate"),
        "Credit Maturity vs Default": box_plot(df, "loan_status", "credity_maturity"),

        
        "Default Rate by Score Group": rate_by_category(df, "score_group"),
        "Default Rate by Income Band": rate_by_category(df, "income_band"),
        "Default Rate by Age Group": rate_by_category(df, "age_group"),
        "Default Rate by Occupation": rate_by_category(df, "occupation_status"),

       
        "Income vs Loan Amount (Density)": density_relationship(
            df, "log_annual_income", "log_loan_amount"
        ),

        
        "Score Group vs Stress Level": risk_heatmap(
            df, "score_group", "stress_bin"
        ),

       
        "Correlation Heatmap": correlation_heatmap(df),
    }
