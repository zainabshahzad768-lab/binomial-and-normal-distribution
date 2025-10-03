# distributions.py
"""
Distributions Module
--------------------
This script demonstrates Binomial and Normal distributions
using NumPy, SciPy, and Matplotlib.

You can use it for learning, teaching, or visualization purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm


def plot_binomial_distribution(n=10, p=0.5):
    """
    Plot a Binomial distribution PMF and CDF.

    Parameters:
        n (int): number of trials
        p (float): probability of success
    """
    x = np.arange(0, n+1)
    pmf = binom.pmf(x, n, p)
    cdf = binom.cdf(x, n, p)

    # Plot PMF
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.stem(x, pmf, basefmt=" ")
    plt.title(f"Binomial PMF (n={n}, p={p})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")

    # Plot CDF
    plt.subplot(1, 2, 2)
    plt.step(x, cdf, where="post")
    plt.title(f"Binomial CDF (n={n}, p={p})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Cumulative Probability")

    plt.tight_layout()
    plt.show()


def plot_normal_distribution(mu=0, sigma=1):
    """
    Plot a Normal distribution PDF and CDF.

    Parameters:
        mu (float): mean
        sigma (float): standard deviation
    """
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)
    cdf = norm.cdf(x, mu, sigma)

    # Plot PDF
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, pdf, label="PDF")
    plt.title(f"Normal PDF (μ={mu}, σ={sigma})")
    plt.xlabel("x")
    plt.ylabel("Density")

    # Plot CDF
    plt.subplot(1, 2, 2)
    plt.plot(x, cdf, label="CDF", color="orange")
    plt.title(f"Normal CDF (μ={mu}, σ={sigma})")
    plt.xlabel("x")
    plt.ylabel("Cumulative Probability")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example Usage
    plot_binomial_distribution(n=10, p=0.5)
    plot_normal_distribution(mu=0, sigma=1)
