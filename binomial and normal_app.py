# probability_app.py

import streamlit as st
import pandas as pd
import numpy as np
import math

st.title("📊 Binomial & Normal Probability Calculator")

# Choose distribution
dist_type = st.radio("Choose Distribution", ["Binomial", "Normal"])

# -----------------------------
# BINOMIAL DISTRIBUTION
# -----------------------------
if dist_type == "Binomial":
    st.header("🎲 Binomial Distribution")

    # User input
    n = st.number_input("Enter number of trials (n)", min_value=1, value=10)
    p = st.number_input("Enter probability of success (p)", min_value=0.0, max_value=1.0, value=0.5)
    k = st.number_input("Enter number of successes (k)", min_value=0, value=5)

    # Binomial PMF calculation
    def binomial_pmf(n, k, p):
        comb = math.comb(n, k)  # nCk
        return comb * (p ** k) * ((1 - p) ** (n - k))

    probability = binomial_pmf(n, k, p)

    st.success(f"➡ Probability of getting exactly {k} successes in {n} trials = **{probability:.6f}**")

# -----------------------------
# NORMAL DISTRIBUTION
# -----------------------------
elif dist_type == "Normal":
    st.header("📈 Normal Distribution")

    # User input
    mu = st.number_input("Enter mean (μ)", value=0.0)
    sigma = st.number_input("Enter standard deviation (σ)", min_value=0.1, value=1.0)
    x = st.number_input("Enter value of x", value=0.0)

    # Normal PDF calculation
    def normal_pdf(x, mu, sigma):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    density = normal_pdf(x, mu, sigma)

    st.success(f"➡ Probability density at x = {x} (μ={mu}, σ={sigma}) = **{density:.6f}**")
