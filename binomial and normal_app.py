# binomial_and_normal_app.py

import streamlit as st
import pandas as pd
import numpy as np

st.title("📊 Binomial and Normal Distribution Explorer")

# Sidebar for parameters
st.sidebar.header("Distribution Parameters")

# Distribution selection
dist_type = st.sidebar.radio("Choose Distribution", ["Binomial", "Normal"])

# Binomial Distribution
if dist_type == "Binomial":
    n = st.sidebar.slider("Number of Trials (n)", min_value=1, max_value=100, value=10)
    p = st.sidebar.slider("Probability of Success (p)", min_value=0.0, max_value=1.0, value=0.5)

    # Generate data
    x = np.arange(0, n+1)
    pmf = [pd.Series(np.random.binomial(n, p, 10000)).value_counts(normalize=True).get(i, 0) for i in x]

    # Create dataframe
    df = pd.DataFrame({"Successes": x, "PMF": pmf})

    st.subheader(f"Binomial Distribution (n={n}, p={p})")
    st.bar_chart(df.set_index("Successes"))

    st.write("📌 This chart shows the probability distribution of successes in a binomial experiment.")

# Normal Distribution
elif dist_type == "Normal":
    mu = st.sidebar.slider("Mean (μ)", min_value=-10, max_value=10, value=0)
    sigma = st.sidebar.slider("Standard Deviation (σ)", min_value=1, max_value=10, value=2)

    # Generate data
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # Create dataframe
    df = pd.DataFrame({"x": x, "PDF": pdf})

    st.subheader(f"Normal Distribution (μ={mu}, σ={sigma})")
    st.line_chart(df.set_index("x"))

    st.write("📌 This chart shows the bell curve of a normal distribution.")
