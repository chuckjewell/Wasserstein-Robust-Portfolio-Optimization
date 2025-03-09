import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_t
from stockdex import Ticker
from portfolio_utils import (generate_data, estimate_parameters,
                             mean_variance_optimization,
                             wasserstein_robust_optimization,
                             evaluate_portfolio)

# List of S&P 500 stocks (top 100 by market cap for simplicity)
SP500_STOCKS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH",
    "JPM", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "AVGO", "MRK",
    "LLY", "COST", "ABBV", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC",
    "TMO", "CSCO", "ACN", "ABT", "CMCSA", "NFLX", "DIS", "VZ", "PFE", "DHR",
    "TXN", "NEE", "PM", "INTC", "AMD", "WFC", "COP", "UPS", "ORCL", "CAT",
    "HON", "IBM", "QCOM", "LOW", "INTU", "SPGI", "BA", "GE", "RTX", "AMAT",
    "DE", "LIN", "GS", "MS", "BLK", "SBUX", "GILD", "C", "ADI", "MDLZ",
    "AXP", "AMGN", "BKNG", "TJX", "MMC", "PLD", "SYK", "ISRG", "CVS", "ETN",
    "VRTX", "ZTS", "REGN", "TMUS", "SCHW", "EOG", "SO", "MO", "BDX", "CME",
    "CI", "CB", "PGR", "DUK", "SLB", "ITW", "APD", "BSX", "TGT", "AON"
]

# Initialize session state for preserving UI state
if 'no_short_selling' not in st.session_state:
    st.session_state.no_short_selling = True
if 'gamma' not in st.session_state:
    st.session_state.gamma = 5.0
if 'epsilon' not in st.session_state:
    st.session_state.epsilon = 0.01
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.95
if 'train_ratio' not in st.session_state:
    st.session_state.train_ratio = 0.5
if 'test_dist' not in st.session_state:
    st.session_state.test_dist = "Normal"

def fetch_stockdex_data(tickers, period='1y'):
    """Fetch stock data using stockdex library"""
    # Add debug container to show API response details
    debug_container = st.sidebar.expander("API Debug Info", expanded=False)
    
    try:
        debug_container.write(f"Fetching data for tickers: {tickers}")
        debug_container.write(f"Period: {period}")
        
        # Split tickers string into a list
        ticker_list = tickers.split()
        
        # Create a DataFrame to store all returns
        all_returns = pd.DataFrame()
        
        # Process each ticker
        for ticker_symbol in ticker_list:
            debug_container.write(f"Processing ticker: {ticker_symbol}")
            
            try:
                # Create a ticker object
                ticker = Ticker(ticker=ticker_symbol)
                
                # Get price data
                price_data = ticker.yahoo_api_price(range=period, dataGranularity='1d')
                
                # Check if we got valid data
                if price_data is not None and not price_data.empty:
                    debug_container.write(f"Successfully fetched data for {ticker_symbol}")
                    debug_container.write(f"Data shape: {price_data.shape}")
                    
                    # Calculate returns
                    returns = price_data['close'].pct_change().dropna()
                    
                    # Add to the DataFrame
                    all_returns[ticker_symbol] = returns
                else:
                    debug_container.error(f"No data received for {ticker_symbol}")
            except Exception as ticker_error:
                debug_container.error(f"Error processing {ticker_symbol}: {str(ticker_error)}")
        
        # Check if we have enough valid tickers
        if len(all_returns.columns) < 2:
            debug_container.error("Need at least 2 assets with valid data")
            st.sidebar.error("Not enough valid tickers with data. Need at least 2.")
            return pd.DataFrame()
        
        # Drop rows with NaN values
        all_returns = all_returns.dropna()
        
        # Show sample of returns
        debug_container.write("Sample of calculated returns:")
        debug_container.write(all_returns.head())
        
        debug_container.success(f"Successfully processed data for {all_returns.shape[1]} assets")
        return all_returns
        
    except Exception as e:
        debug_container.exception(f"Error in fetch_stockdex_data: {str(e)}")
        st.sidebar.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def main():
    # Set page to wide mode
    st.set_page_config(layout="wide")
    
    st.title("Wasserstein-Robust Portfolio Optimization")
    
    # Add a brief introduction with usage instructions
    st.markdown("""
    This app helps you compare traditional and robust portfolio optimization strategies.
    
    **How to use this app:**
    1. Select a data source (Stock Data, Upload CSV, or Generate Synthetic)
    2. Adjust the parameters to see how they affect the optimal portfolio
    3. Compare the performance metrics and portfolio weights
    4. Examine the efficient frontier graph to understand the risk-return tradeoff
    """)

    # --- Sidebar: User Inputs ---
    st.sidebar.header("Input Data")
    data_option = st.sidebar.radio("Data Source:", ["Stock Data", "Upload CSV", "Generate Synthetic"])

    if data_option == "Stock Data":
        st.sidebar.subheader("Stock Data")
        
        # Add info about the data source
        st.sidebar.info("Using StockDex library to fetch stock data.")
        
        # Add a button to randomize stocks
        if st.sidebar.button("🎲 Randomize (5 S&P 500 Stocks)"):
            random_tickers = " ".join(random.sample(SP500_STOCKS, 5))
            st.session_state.tickers_input = random_tickers
        
        # Default to ETFs that are more likely to work, but use session state if available
        tickers_input = st.sidebar.text_area(
            "Enter ticker symbols (space separated)",
            value=st.session_state.get('tickers_input', "SPY QQQ DIA"),
            help="Example: SPY QQQ DIA IWM XLF (ETFs tend to be more reliable)",
            key="tickers_input"
        )
        
        period = st.sidebar.selectbox(
            "Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=2,
            help="Data period to download",
            key="period"
        )
        
        if st.sidebar.button("Fetch Data"):
            with st.spinner("Fetching stock data..."):
                # Show data preview section
                data_preview = st.empty()
                
                # Fetch data with detailed debugging
                returns_df = fetch_stockdex_data(tickers_input, period)
                
                # Check if we got valid data
                if returns_df.empty:
                    st.error("Failed to fetch valid data. Please check the API Debug Info in the sidebar for details.")
                    
                    # Provide clear suggestions in a more structured format
                    suggestion_col1, suggestion_col2 = st.columns(2)
                    with suggestion_col1:
                        st.info("### Troubleshooting Suggestions:")
                        st.markdown("""
                        - Try using ETFs like SPY, QQQ, DIA instead of individual stocks
                        - Use a shorter time period (1mo or 3mo)
                        - Check your internet connection
                        """)
                    
                    with suggestion_col2:
                        st.info("### Alternative Option:")
                        st.markdown("""
                        If fetching stock data continues to fail, select the **Generate Synthetic** 
                        option from the Data Source dropdown instead. This will create synthetic 
                        market data for testing your portfolio optimization.
                        """)
                    return
                
                # Success case
                st.sidebar.success(f"Successfully fetched data for {len(returns_df.columns)} assets")
                
                # Show data preview
                data_preview.subheader("Data Preview (Returns)")
                data_preview.dataframe(returns_df.head(), use_container_width=True)
                
                # Convert to numpy array for the optimization
                returns = returns_df.values
        else:
            st.info("Please enter ticker symbols and click 'Fetch Data'")
            return
            
    elif data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload returns (CSV)", type="csv")
        if uploaded_file:
            returns = pd.read_csv(uploaded_file).values
            if returns.shape[1] < 2:
                st.sidebar.error("CSV must have at least 2 assets.")
                return
            if returns.shape[0] < 30 * returns.shape[1]:
                st.sidebar.warning("Low sample size may lead to unreliable results.")
        else:
            st.sidebar.warning("Please upload a CSV file.")
            return
    else:  # Generate Synthetic
        st.sidebar.subheader("Synthetic Data")
        n_assets = st.sidebar.slider("Number of assets", 2, 10, 3)
        n_samples = st.sidebar.slider("Number of samples", 100, 10000, 1000)
        returns = generate_data(n_assets, n_samples)

    st.sidebar.header("Parameters")
    gamma = st.sidebar.slider("Risk aversion (γ)", 1.0, 10.0, st.session_state.gamma, 0.1,
                              help="Controls the trade-off between risk and return. Higher values (e.g., 10.0) prioritize lower risk at the expense of return. Lower values (e.g., 1.0) prioritize higher returns while accepting more risk.",
                              key="gamma")
    epsilon = st.sidebar.slider("Wasserstein radius (ε)", 0.0, 0.1, st.session_state.epsilon, 0.001,
                                help="Controls the level of robustness against uncertainty in return estimates. Higher values create more conservative portfolios that perform better in volatile markets. A value of 0 is equivalent to traditional optimization.",
                                key="epsilon")
    alpha = st.sidebar.slider("Confidence level (α)", 0.9, 0.99, st.session_state.alpha, 0.01,
                              help="Used for Expected Shortfall calculation, which measures the average loss in the worst α% of scenarios. Higher values (e.g., 0.99) focus on more extreme tail risks.",
                              key="alpha")

    st.sidebar.header("Constraints")
    no_short_selling = st.sidebar.checkbox("No short selling", st.session_state.no_short_selling,
                                           help="When checked, all portfolio weights must be non-negative (≥ 0), meaning you can only buy assets, not short sell them. This is a common constraint for many investors.",
                                           key="no_short_selling")

    st.sidebar.header("Scenario Configuration")
    train_ratio = st.sidebar.slider("Training data ratio", 0.1, 0.9, st.session_state.train_ratio, 0.1,
                                    help="Fraction of data used for estimating parameters (training). The remaining data is used for out-of-sample testing. A value of 0.5 means 50% for training, 50% for testing.",
                                    key="train_ratio")
    test_dist = st.sidebar.selectbox("Test data distribution", ["Normal", "T-Distribution"],
                                     index=0 if st.session_state.test_dist == "Normal" else 1,
                                     help="Distribution used for test data. 'Normal' assumes returns follow a normal distribution. 'T-Distribution' has fatter tails, better representing extreme market events like crashes or rallies.",
                                     key="test_dist")

    # --- Process Data ---
    train_size = int(len(returns) * train_ratio)
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]

    if test_dist == "T-Distribution":
        # Simulate fat-tailed test data
        df = 3  # Degrees of freedom
        test_returns = multivariate_t.rvs(shape=np.cov(train_returns, rowvar=False), 
                                          df=df, size=len(test_returns))

    mu_hat, Sigma_hat = estimate_parameters(train_returns)

    # --- Optimize Portfolios ---
    w_mv = mean_variance_optimization(mu_hat, Sigma_hat, gamma, no_short_selling)
    w_robust = wasserstein_robust_optimization(mu_hat, Sigma_hat, gamma, epsilon, no_short_selling)

    # --- Evaluate Performance ---
    mv_metrics = evaluate_portfolio(test_returns, w_mv, alpha)
    robust_metrics = evaluate_portfolio(test_returns, w_robust, alpha)

    # --- Display Results ---
    # Create a container for all explanations
    with st.expander("📚 Understanding Portfolio Optimization Concepts", expanded=False):
        explanation_tab1, explanation_tab2 = st.tabs(["Optimization Strategies", "Efficient Frontier"])
        
        with explanation_tab1:
            st.markdown("""
            ### Portfolio Optimization Strategies
            
            This app implements two portfolio optimization strategies:
            
            1. **Traditional Mean-Variance Optimization**: Developed by Harry Markowitz (Nobel Prize winner), this approach finds the optimal balance between expected return and risk (measured by variance).
            
            2. **Wasserstein-Robust Optimization**: An advanced approach that accounts for uncertainty in the estimation of asset returns. It creates portfolios that are more robust to market fluctuations and estimation errors.
            
            ### When to use each approach?
            
            - **Traditional Mean-Variance**: Works well in stable markets when you have high confidence in your return estimates.
            - **Wasserstein-Robust**: Better in volatile markets or when you have less confidence in your return estimates.
            """)
            
        with explanation_tab2:
            st.markdown("""
            ### What is the Efficient Frontier?
            
            The **Efficient Frontier** is a key concept in portfolio theory that shows the set of optimal portfolios that offer the highest expected return for a given level of risk.
            
            In this graph:
            - **X-axis (Volatility)**: Represents the risk of the portfolio
            - **Y-axis (Expected Return)**: Represents the expected return of the portfolio
            - **Blue points**: Traditional Mean-Variance portfolios with different risk aversion levels
            - **Orange points**: Wasserstein-Robust portfolios with different risk aversion levels
            
            ### How to interpret this graph:
            
            - **Upper left is better**: Portfolios in the upper left corner have higher returns for lower risk
            - **The curve shape**: Shows the trade-off between risk and return
            - **Comparing the two curves**: The gap between the blue and orange curves shows the "cost of robustness" - how much expected return you sacrifice for greater stability
            """)
    
    # Generate efficient frontier data
    gammas = np.linspace(1, 10, 20)
    mv_returns, mv_vols = [], []
    robust_returns, robust_vols = [], []

    for g in gammas:
        w_mv = mean_variance_optimization(mu_hat, Sigma_hat, g, no_short_selling)
        w_robust = wasserstein_robust_optimization(mu_hat, Sigma_hat, g, epsilon, no_short_selling)
        mv_m = evaluate_portfolio(test_returns, w_mv, alpha)
        robust_m = evaluate_portfolio(test_returns, w_robust, alpha)
        mv_returns.append(mv_m["Expected Return"])
        mv_vols.append(mv_m["Volatility"])
        robust_returns.append(robust_m["Expected Return"])
        robust_vols.append(robust_m["Volatility"])
    
    # Calculate evaluation metrics
    sharpe_mv = mv_metrics['Expected Return'] / mv_metrics['Volatility']
    sharpe_robust = robust_metrics['Expected Return'] / robust_metrics['Volatility']
    
    # Calculate return-to-ES ratio (similar to Sortino ratio)
    sortino_mv = mv_metrics['Expected Return'] / mv_metrics['ES']
    sortino_robust = robust_metrics['Expected Return'] / robust_metrics['ES']
    
    # Determine which strategy is better based on the metrics
    if sharpe_robust > sharpe_mv and sortino_robust > sortino_mv:
        recommendation = "The **Wasserstein-Robust strategy** appears superior in this scenario, offering better risk-adjusted returns."
    elif sharpe_mv > sharpe_robust and sortino_mv > sortino_robust:
        recommendation = "The **Traditional Mean-Variance strategy** appears superior in this scenario, offering better risk-adjusted returns."
    else:
        if (sharpe_robust/sharpe_mv + sortino_robust/sortino_mv)/2 > 1:
            recommendation = "The **Wasserstein-Robust strategy** has a slight edge in this scenario, particularly for risk-conscious investors."
        else:
            recommendation = "The **Traditional Mean-Variance strategy** has a slight edge in this scenario, but may be less stable in volatile markets."
    
    # Main content in a two-column layout with better space utilization
    left_col, right_col = st.columns([1, 1.5])
    
    with left_col:
        # Portfolio weights
        st.subheader("Portfolio Weights")
        weights_df = pd.DataFrame({
            "Traditional MV": w_mv.round(4),
            "Wasserstein-Robust": w_robust.round(4)
        })
        
        if data_option == "Stock Data":
            # Add asset names if using Stock Data
            tickers_list = tickers_input.split()
            if len(tickers_list) == weights_df.shape[0]:
                weights_df.index = tickers_list
            
        st.dataframe(weights_df, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility", f"Expected Shortfall (α={alpha})"],
            "Traditional MV": [
                f"{mv_metrics['Expected Return']:.4f}",
                f"{mv_metrics['Volatility']:.4f}",
                f"{mv_metrics['ES']:.4f}"
            ],
            "Wasserstein-Robust": [
                f"{robust_metrics['Expected Return']:.4f}",
                f"{robust_metrics['Volatility']:.4f}",
                f"{robust_metrics['ES']:.4f}"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Evaluation metrics
        eval_df = pd.DataFrame({
            "Metric": ["Sharpe Ratio", "Return-to-ES Ratio"],
            "Traditional MV": [f"{sharpe_mv:.4f}", f"{sortino_mv:.4f}"],
            "Wasserstein-Robust": [f"{sharpe_robust:.4f}", f"{sortino_robust:.4f}"]
        })
        
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        
        # Key Insights (stacked vertically)
        st.subheader("Key Insights")
        st.markdown("""
        ### Performance Metrics Explained
        
        - **Sharpe Ratio**: Higher is better, measures return per unit of risk
        - **Return-to-ES Ratio**: Higher is better, measures return per unit of tail risk
        
        ### Strategy Characteristics
        
        - **Traditional optimization** typically shows higher expected returns but may underestimate risks
        - **Robust optimization** produces more conservative portfolios that may perform better in volatile markets
        - The **gap between the curves** on the efficient frontier represents the "cost of robustness"
        """)
        
        # Strategy Evaluation (stacked vertically)
        st.subheader("Strategy Evaluation")
        st.markdown(f"""
        ### Overall Assessment
        
        {recommendation}
        
        **For your specific investment goals:**
        - If you prioritize maximizing returns in stable markets: Consider the Traditional approach
        - If you prioritize stability and protection against market uncertainty: Consider the Robust approach
        - For a balanced portfolio: Consider allocating to both strategies
        """)
    
    with right_col:
        # Efficient Frontier Plot
        st.subheader("Efficient Frontier")
        
        # Create a larger figure for better visualization
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(mv_vols, mv_returns, label="Traditional MV", alpha=0.7, s=50)
        ax.scatter(robust_vols, robust_returns, label="Wasserstein-Robust", alpha=0.7, s=50)
        
        # Add lines connecting the points for better visualization
        ax.plot(mv_vols, mv_returns, 'b--', alpha=0.4)
        ax.plot(robust_vols, robust_returns, 'r--', alpha=0.4)
        
        ax.set_xlabel("Volatility", fontsize=12)
        ax.set_ylabel("Expected Return", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        st.pyplot(fig, use_container_width=True)

if __name__ == "__main__":
    main()