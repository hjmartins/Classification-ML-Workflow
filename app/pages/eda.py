import streamlit as st

from back.loan_eda import load_data, univariate_figures, bivariate_figures

st.title("📊 Exploratory Data Analysis")

df = load_data()

st.header("Univariate Analysis")
for fig in univariate_figures(df).values():
    st.plotly_chart(fig, use_container_width=True)

st.header("Bivariate Analysis")
for i, fig in enumerate(bivariate_figures(df).values()):
    st.plotly_chart(fig, use_container_width=True, key=f"bivariate_{i}")
