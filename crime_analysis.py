import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


knn = joblib.load("knn_model.pkl")
pca = joblib.load("pca_model.pkl")
scaler = joblib.load("scaler.pkl")
state_year = pd.read_csv("processed_state_year.csv")

st.set_page_config(page_title="Crime Analysis - India", layout="wide")

st.title("Crime Analysis and Prediction Dashboard")
st.markdown("### Analyze and visualize crime trends across Indian states (2001–2020)")



st.sidebar.header("Input Parameters")
state_input = st.sidebar.selectbox("Select State/UT", sorted(state_year["STATE/UT"].unique()))
year_input = st.sidebar.slider("Select Year", 2001, 2020, 2020)


row = state_year[(state_year["STATE/UT"] == state_input) & (state_year["YEAR"] == year_input)]

if row.empty:
    st.error("No data available for that state and year.")
else:
    
    
    X_single = row.drop(columns=["STATE/UT", "YEAR", "TOTAL_CRIMES", "CRIME_ZONE", "ZONE_CODE"])
    X_single_scaled = scaler.transform(X_single)
    X_single_pca = pca.transform(X_single_scaled)
    zone_pred = knn.predict(X_single_pca)[0]
    zone_label = {0: "Low", 1: "Medium", 2: "High"}[zone_pred]

    st.subheader(f"Risk Zone for {state_input} ({year_input}): **{zone_label}**")

    
    
    crimes_only = row.drop(columns=["STATE/UT", "YEAR", "TOTAL_CRIMES", "CRIME_ZONE", "ZONE_CODE"])
    top3 = crimes_only.T.sort_values(by=row.index[0], ascending=False).head(3)
    st.markdown("### Top 3 Crimes:")
    for crime, val in zip(top3.index, top3[row.index[0]].values):
        st.write(f"- {crime}: {int(val)} cases")

    # --- Correlation among top 3 crimes ---
    sub = state_year[state_year["STATE/UT"] == state_input]
    corr = sub[top3.index].corr()

    st.markdown("### Correlation among Top 3 Crimes:")
    st.dataframe(corr.style.background_gradient(cmap="YlGnBu").format("{:.2f}"))

    # --- Trend visualization ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub["YEAR"], sub["TOTAL_CRIMES"], marker='o', linewidth=2, color='royalblue')
    ax.set_title(f"Crime Trend in {state_input} (2001–2020)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Crimes")
    ax.grid(True)
    st.pyplot(fig)

# --- National Analysis Section ---
st.markdown("---")
st.header("📊 National Crime Evolution (2001–2020)")

# 1. Year-wise total crimes across India
yearly_total = state_year.groupby("YEAR")["TOTAL_CRIMES"].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(yearly_total["YEAR"], yearly_total["TOTAL_CRIMES"], marker='o', linewidth=2, color='teal')
ax.set_title("Total Crimes Against Women in India (2001–2020)")
ax.set_xlabel("Year")
ax.set_ylabel("Total Crimes")
ax.grid(True)
st.pyplot(fig)

# 2. Top 5 states by average total crimes
state_avg = state_year.groupby("STATE/UT")["TOTAL_CRIMES"].mean().sort_values(ascending=False).head(5)
st.subheader("Top 5 States with Highest Average Crimes (2001–2020)")
st.bar_chart(state_avg)

# 3. Correlation among major crime categories
key_crimes = [
    "RAPE",
    "KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS",
    "CRUELTY BY HUSBAND OR HIS RELATIVES",
    "ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY",
]
corr_matrix = state_year[key_crimes].corr()

st.subheader("Correlation Matrix – Major Crime Categories")
st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm").format("{:.2f}"))
