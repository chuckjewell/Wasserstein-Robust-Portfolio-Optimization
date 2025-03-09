import numpy as np
import cvxpy as cp
from scipy.stats import norm
import streamlit as st  # Added for sidebar warnings

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

def evaluate_portfolio(returns, weights, alpha=0.95):
    """
    Evaluate portfolio performance metrics.
    
    Parameters:
    - returns: (n_samples, n_assets) array of asset returns.
    - weights: Portfolio weights.
    - alpha: Confidence level for ES.
    
    Returns:
    - metrics: Dictionary with 'Expected Return', 'Volatility', and 'ES'.
    """
    portfolio_returns = returns @ weights
    exp_return = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    es = expected_shortfall(returns, weights, alpha)
    return {"Expected Return": exp_return, "Volatility": volatility, "ES": es}