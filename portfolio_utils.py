import numpy as np
import cvxpy as cp
from scipy.stats import norm
import streamlit as st  # Added for sidebar warnings
import matplotlib.pyplot as plt

def generate_data(n_assets=3, n_samples=1000, seed=42):
    """
    Generate synthetic asset returns assuming a multivariate normal distribution.
    
    Parameters:
    - n_assets: Number of assets.
    - n_samples: Number of return samples.
    - seed: Random seed for reproducibility.
    
    Returns:
    - returns: (n_samples, n_assets) array of asset returns.
    """
    np.random.seed(seed)
    # Define true mean returns and volatilities
    true_mu = np.linspace(0.04, 0.08, n_assets)
    true_vol = np.linspace(0.05, 0.15, n_assets)
    # Create a correlation matrix (identity for independence)
    corr = np.eye(n_assets)
    # Covariance matrix
    true_cov = np.diag(true_vol) @ corr @ np.diag(true_vol)
    # Generate returns
    returns = np.random.multivariate_normal(true_mu, true_cov, n_samples)
    return returns

def estimate_parameters(returns):
    """
    Estimate mean vector and covariance matrix from return data.
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns.
    
    Returns:
    - mu_hat: Estimated mean vector.
    - Sigma_hat: Estimated covariance matrix.
    """
    # Remove rows with any NaN values
    returns_clean = returns[~np.isnan(returns).any(axis=1)]
    n_samples, n_assets = returns_clean.shape
    
    # Check for sufficient data
    if n_samples < 2:
        raise ValueError("Insufficient data: at least 2 samples are required after removing NaNs.")
    if n_samples < 30 * n_assets:
        st.sidebar.warning(f"Warning: Only {n_samples} samples for {n_assets} assets. Low sample size may lead to unreliable estimates (recommend ≥ {30 * n_assets} samples).")
    
    # Estimate mean and covariance
    mu_hat = np.mean(returns_clean, axis=0)
    Sigma_hat = np.cov(returns_clean, rowvar=False)
    
    # Apply shrinkage to covariance matrix for robustness
    shrinkage = 0.1  # Shrinkage intensity (0 to 1)
    trace_Sigma = np.trace(Sigma_hat)
    Sigma_shrunk = (1 - shrinkage) * Sigma_hat + shrinkage * np.eye(n_assets) * (trace_Sigma / n_assets)
    
    # Ensure symmetry
    Sigma_shrunk = (Sigma_shrunk + Sigma_shrunk.T) / 2
    
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvals(Sigma_shrunk))
    if min_eig < 1e-6:
        Sigma_shrunk += np.eye(n_assets) * (1e-6 - min_eig)
    
    return mu_hat, Sigma_shrunk

def mean_variance_optimization(mu, Sigma, gamma, no_short=True):
    """
    Solve traditional mean-variance portfolio optimization.
    
    Parameters:
    - mu: Mean return vector.
    - Sigma: Covariance matrix.
    - gamma: Risk aversion parameter.
    - no_short: If True, enforce no short-selling (weights >= 0).
    
    Returns:
    - w: Optimal portfolio weights.
    """
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    objective = cp.Minimize(-mu.T @ w + (gamma / 2) * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1]
    if no_short:
        constraints.append(w >= 0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value

def wasserstein_robust_optimization(mu, Sigma, gamma, epsilon, no_short=True):
    """
    Solve Wasserstein-robust portfolio optimization.
    
    Parameters:
    - mu: Mean return vector.
    - Sigma: Covariance matrix.
    - gamma: Risk aversion parameter.
    - epsilon: Wasserstein radius.
    - no_short: If True, enforce no short-selling (weights >= 0).
    
    Returns:
    - w: Optimal robust portfolio weights.
    """
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    # Wasserstein-robust objective: -mu^T w + (gamma/2) w^T Sigma w + epsilon ||w||_2
    objective = cp.Minimize(-mu.T @ w + (gamma / 2) * cp.quad_form(w, Sigma) + epsilon * cp.norm(w, 2))
    constraints = [cp.sum(w) == 1]
    if no_short:
        constraints.append(w >= 0)
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return w.value

def expected_shortfall(returns, weights, alpha=0.95):
    """
    Compute the Expected Shortfall (ES) at confidence level alpha.
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns.
    - weights: Portfolio weights.
    - alpha: Confidence level.
    
    Returns:
    - es: Expected Shortfall.
    """
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, (1 - alpha) * 100)
    es = -np.mean(portfolio_returns[portfolio_returns <= var])
    return es

def estimate_sdf(returns, risk_aversion=1.0, risk_free_rate=0.0):
    """
    Estimate the Stochastic Discount Factor (SDF) from historical returns.
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns
    - risk_aversion: Risk aversion coefficient (default: 1.0)
    - risk_free_rate: Risk-free rate (default: 0.0)
    
    Returns:
    - sdf: (n_samples,) array of SDF values
    """
    # Market return is the equally weighted average of all asset returns
    market_return = np.mean(returns, axis=1)
    
    # Linear factor model: M = 1 - b * (market_return - risk_free_rate)
    sdf = 1 - risk_aversion * (market_return - risk_free_rate)
    
    # Ensure SDF is positive (theoretical requirement)
    sdf = np.maximum(sdf, 1e-6)
    
    # Normalize to have mean 1 (for proper discounting)
    sdf = sdf / np.mean(sdf)
    
    return sdf

def simplified_weds(returns, weights, alpha=0.95, risk_aversion=1.0):
    """
    Compute simplified Wasserstein Expected Discounted Shortfall (WEDS).
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns
    - weights: Portfolio weights
    - alpha: Confidence level
    - risk_aversion: Risk aversion for SDF estimation
    
    Returns:
    - weds: Simplified WEDS value
    - sdf: Estimated SDF values
    - discounted_returns: Portfolio returns discounted by SDF
    """
    # Estimate SDF
    sdf = estimate_sdf(returns, risk_aversion)
    
    # Calculate portfolio returns
    portfolio_returns = returns @ weights
    
    # Apply SDF to get discounted returns
    discounted_returns = portfolio_returns * sdf
    
    # Calculate VaR at confidence level alpha
    var = np.percentile(discounted_returns, (1 - alpha) * 100)
    
    # Calculate WEDS as the average of discounted returns below VaR
    weds = -np.mean(discounted_returns[discounted_returns <= var])
    
    return weds, sdf, discounted_returns

def evaluate_portfolio(returns, weights, alpha=0.95, include_sdf=True):
    """
    Evaluate portfolio performance metrics including SDF-based measures.
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns.
    - weights: Portfolio weights.
    - alpha: Confidence level for ES and WEDS.
    - include_sdf: Whether to include SDF-based metrics.
    
    Returns:
    - metrics: Dictionary with performance metrics.
    """
    portfolio_returns = returns @ weights
    exp_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    es = expected_shortfall(returns, weights, alpha)
    
    metrics = {
        "Expected Return": exp_return,
        "Volatility": volatility,
        "ES": es
    }
    
    if include_sdf:
        weds, sdf, discounted_returns = simplified_weds(returns, weights, alpha)
        metrics.update({
            "WEDS": weds,
            "SDF": sdf,
            "Discounted Returns": discounted_returns,
            "Portfolio Returns": portfolio_returns
        })
    
    return metrics

def plot_sdf(sdf, fig_width=10, fig_height=6):
    """
    Plot the Stochastic Discount Factor over time/scenarios.
    
    Parameters:
    - sdf: Array of SDF values
    - fig_width: Figure width
    - fig_height: Figure height
    
    Returns:
    - fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(sdf, color='purple', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("SDF Value", fontsize=12)
    ax.set_title("Stochastic Discount Factor (SDF)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig

def plot_returns_distribution(portfolio_returns, discounted_returns, fig_width=10, fig_height=6):
    """
    Plot distribution of original vs. discounted returns.
    
    Parameters:
    - portfolio_returns: Original portfolio returns
    - discounted_returns: SDF-discounted returns
    - fig_width: Figure width
    - fig_height: Figure height
    
    Returns:
    - fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot histograms
    bins = np.linspace(min(min(portfolio_returns), min(discounted_returns)),
                       max(max(portfolio_returns), max(discounted_returns)),
                       30)
    ax.hist(portfolio_returns, bins=bins, alpha=0.5, label="Original Returns", color='blue')
    ax.hist(discounted_returns, bins=bins, alpha=0.5, label="SDF-Discounted Returns", color='orange')
    
    # Add vertical lines for means
    ax.axvline(x=np.mean(portfolio_returns), color='blue', linestyle='--', linewidth=2)
    ax.axvline(x=np.mean(discounted_returns), color='orange', linestyle='--', linewidth=2)
    
    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Returns: Original vs. SDF-Discounted", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig

def plot_risk_comparison(es, weds, strategy_names, fig_width=10, fig_height=6):
    """
    Plot comparison of ES vs. WEDS for different strategies.
    
    Parameters:
    - es: List of ES values for different strategies
    - weds: List of WEDS values for different strategies
    - strategy_names: Names of the strategies
    - fig_width: Figure width
    - fig_height: Figure height
    
    Returns:
    - fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    ax.bar(x - width/2, es, width, label='Expected Shortfall (ES)', color='blue', alpha=0.7)
    ax.bar(x + width/2, weds, width, label='Wasserstein Expected Discounted Shortfall (WEDS)', color='red', alpha=0.7)
    
    ax.set_ylabel('Risk Measure Value', fontsize=12)
    ax.set_title('Risk Measure Comparison: ES vs. WEDS', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage increase labels
    for i, (es_val, weds_val) in enumerate(zip(es, weds)):
        pct_increase = (weds_val - es_val) / es_val * 100
        ax.annotate(f"+{pct_increase:.1f}%",
                   xy=(i + width/2, weds_val),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold')
    
    return fig