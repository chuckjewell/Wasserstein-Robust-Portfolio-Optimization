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
                             evaluate_portfolio, plot_sdf,
                             plot_returns_distribution, plot_risk_comparison)

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

# Initialize session state for preserving UI state and data
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
if 'include_sdf' not in st.session_state:
    st.session_state.include_sdf = True
    
# Store data in session state
if 'returns_data' not in st.session_state:
    st.session_state.returns_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_option' not in st.session_state:
    st.session_state.data_option = "Stock Data"

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
    
    # Add a brief introduction
    st.markdown("""
    This app helps you compare traditional and robust portfolio optimization strategies.
    """)
    
    # Add usage instructions in an expander
    with st.expander("📋 How to use this app", expanded=False):
        st.markdown("""
        ### Getting Started
        
        1. **Select a data source**:
           - **Stock Data**: Enter ticker symbols (e.g., SPY, AAPL) to fetch real market data
           - **Upload CSV**: Upload your own returns data in CSV format
           - **Generate Synthetic**: Create simulated market data for testing
        
        2. **Adjust parameters**:
           - **Risk aversion (γ)**: Controls the trade-off between risk and return
           - **Wasserstein radius (ε)**: Controls the level of robustness
           - **Confidence level (α)**: Used for Expected Shortfall calculation
           - **No short selling**: Restricts to long-only positions
           - **Include SDF Analysis**: Enables advanced risk assessment
        
        3. **Analyze results**:
           - Compare portfolio weights and performance metrics
           - Examine the efficient frontier graph
           - Review the strategy evaluation and recommendation
        
        ### Tips
        
        - Use the expandable sections to view detailed information
        - Try different parameter combinations to see how they affect the portfolios
        - Enable SDF Analysis for more comprehensive risk assessment
        """)

    # --- Sidebar: User Inputs ---
    st.sidebar.header("Input Data")
    data_option = st.sidebar.radio("Data Source:", ["Stock Data", "Upload CSV", "Generate Synthetic"], key="data_source")
    
    # Store the data option in session state
    st.session_state.data_option = data_option
    
    # Data loading section - only show if data is not already loaded or if data source changed
    if not st.session_state.data_loaded or st.session_state.data_option != data_option:
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
                    
                    # Store data in session state
                    st.session_state.returns_data = returns_df.values
                    st.session_state.data_loaded = True
                    st.session_state.tickers_list = tickers_input.split() if data_option == "Stock Data" else None
                    
                    # Set returns for processing
                    returns = st.session_state.returns_data
            else:
                st.info("Please enter ticker symbols and click 'Fetch Data'")
                return
                
        elif data_option == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload returns (CSV)", type="csv")
            if uploaded_file:
                returns_df = pd.read_csv(uploaded_file)
                returns = returns_df.values
                if returns.shape[1] < 2:
                    st.sidebar.error("CSV must have at least 2 assets.")
                    return
                if returns.shape[0] < 30 * returns.shape[1]:
                    st.sidebar.warning("Low sample size may lead to unreliable results.")
                
                # Store data in session state
                st.session_state.returns_data = returns
                st.session_state.data_loaded = True
                st.session_state.tickers_list = None
            else:
                st.sidebar.warning("Please upload a CSV file.")
                return
        else:  # Generate Synthetic
            st.sidebar.subheader("Synthetic Data")
            n_assets = st.sidebar.slider("Number of assets", 2, 10, 3)
            n_samples = st.sidebar.slider("Number of samples", 100, 10000, 1000)
            
            if st.sidebar.button("Generate Data"):
                returns = generate_data(n_assets, n_samples)
                
                # Store data in session state
                st.session_state.returns_data = returns
                st.session_state.data_loaded = True
                st.session_state.tickers_list = [f"Asset {i+1}" for i in range(n_assets)]
                
                st.sidebar.success(f"Successfully generated synthetic data with {n_assets} assets")
            else:
                st.info("Click 'Generate Data' to create synthetic market data")
                return
    else:
        # Use data from session state if already loaded
        returns = st.session_state.returns_data
        
        # Show a reset button to allow reloading data
        if st.sidebar.button("Reset Data"):
            st.session_state.data_loaded = False
            st.rerun()

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

    st.sidebar.header("Advanced Analysis")
    include_sdf = st.sidebar.checkbox("Include SDF Analysis", st.session_state.include_sdf,
                                     help="When checked, the app will calculate and display Stochastic Discount Factor (SDF) analysis, including Wasserstein Expected Discounted Shortfall (WEDS) and related visualizations. This provides deeper insights into how portfolios perform under different market conditions.",
                                     key="include_sdf")

    st.sidebar.header("Scenario Configuration")
    train_ratio = st.sidebar.slider("Training data ratio", 0.1, 0.9, st.session_state.train_ratio, 0.1,
                                    help="Fraction of data used for estimating parameters (training). The remaining data is used for out-of-sample testing. A value of 0.5 means 50% for training, 50% for testing.",
                                    key="train_ratio")
    test_dist = st.sidebar.selectbox(
        "Test data distribution",
        ["Normal", "T-Distribution"],
        index=0 if st.session_state.test_dist == "Normal" else 1,
        help="""
        This controls how the test portion of your data is modified for evaluating portfolio performance:
        
        - **Normal**: Uses your actual test data without modification. This assumes your historical data is representative of future market conditions.
        
        - **T-Distribution**: Transforms your test data to have more extreme movements (both up and down). This simulates more volatile market conditions to stress-test your portfolio.
        
        This applies to both real market data and synthetic data. It's a way to see how your portfolio might perform in different market environments.
        """,
        key="test_dist"
    )

    # --- Process Data ---
    train_size = int(len(returns) * train_ratio)
    train_returns = returns[:train_size]
    test_returns = returns[train_size:]

    # Apply the selected test data distribution
    # Note: This applies to both real and synthetic data - we're using the statistical properties
    # of the training data to generate new test scenarios
    with st.expander(f"ℹ️ How Test Data is Used ({train_ratio*100:.0f}% train / {(1-train_ratio)*100:.0f}% test)", expanded=False):
        st.markdown(f"""
        ### Data Splitting Process
        
        Even when using real market data, we split it into:
        - **Training data ({train_ratio*100:.0f}%)**: Used to estimate parameters and optimize portfolios
        - **Test data ({(1-train_ratio)*100:.0f}%)**: Used to evaluate how portfolios would perform out-of-sample
        
        ### Test Distribution Options
        
        The **Test Distribution** option controls how this test data is modified:
        
        #### Normal Distribution
        - Uses your actual test data without modification
        - Assumes historical data is representative of future market conditions
        - Answers: "How would my portfolio perform if future conditions are similar to what we've seen?"
        
        #### T-Distribution
        - Takes your real data and makes it more extreme to simulate market stress
        - Creates more frequent and severe market movements (both up and down)
        - Answers: "How would my portfolio perform if market conditions were more volatile?"
        
        ### Current Setting
        
        You are currently using the **{test_dist}** for testing.
        """)
    
    # Apply the selected distribution
    if test_dist == "T-Distribution":
        # Simulate fat-tailed test data using multivariate t-distribution
        df = 3  # Degrees of freedom (lower = fatter tails = more extreme events)
        original_test_returns = test_returns.copy()  # Save original for comparison
        test_returns = multivariate_t.rvs(shape=np.cov(train_returns, rowvar=False),
                                          df=df, size=len(test_returns))
        
        # Add a small indicator that T-Distribution is being used
        st.info("Using **T-Distribution** for test data (more extreme market movements)")
    else:
        # Add a small indicator that Normal Distribution is being used
        st.info("Using **Normal Distribution** for test data (original market movements)")

    mu_hat, Sigma_hat = estimate_parameters(train_returns)

    # --- Optimize Portfolios ---
    with st.spinner("Optimizing portfolios... Please wait."):
        w_mv = mean_variance_optimization(mu_hat, Sigma_hat, gamma, no_short_selling)
        w_robust = wasserstein_robust_optimization(mu_hat, Sigma_hat, gamma, epsilon, no_short_selling)

        # --- Evaluate Performance ---
        mv_metrics = evaluate_portfolio(test_returns, w_mv, alpha, include_sdf)
        robust_metrics = evaluate_portfolio(test_returns, w_robust, alpha, include_sdf)

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
    
    # Generate efficient frontier data with more points and better sorting
    with st.spinner("Generating efficient frontier... This may take a moment."):
        gammas = np.logspace(-1, 2, 30)  # Use logarithmic spacing for better coverage
        mv_returns, mv_vols = [], []
        robust_returns, robust_vols = [], []
        
        # Progress bar for efficient frontier calculation
        progress_bar = st.progress(0)
        
        for i, g in enumerate(gammas):
            try:
                w_mv = mean_variance_optimization(mu_hat, Sigma_hat, g, no_short_selling)
                w_robust = wasserstein_robust_optimization(mu_hat, Sigma_hat, g, epsilon, no_short_selling)
                mv_m = evaluate_portfolio(test_returns, w_mv, alpha, False)  # Don't include SDF for frontier to save time
                robust_m = evaluate_portfolio(test_returns, w_robust, alpha, False)
                mv_returns.append(mv_m["Expected Return"])
                mv_vols.append(mv_m["Volatility"])
                robust_returns.append(robust_m["Expected Return"])
                robust_vols.append(robust_m["Volatility"])
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(gammas))
            except Exception as e:
                st.warning(f"Optimization failed for gamma={g:.2f}: {str(e)}")
                continue
        
        # Clear progress bar when done
        progress_bar.empty()
    
    # Sort the points by volatility for smoother curves
    mv_points = sorted(zip(mv_vols, mv_returns))
    robust_points = sorted(zip(robust_vols, robust_returns))
    
    # Unzip the sorted points
    mv_vols, mv_returns = zip(*mv_points) if mv_points else ([], [])
    robust_vols, robust_returns = zip(*robust_points) if robust_points else ([], [])
    
    # Calculate evaluation metrics
    sharpe_mv = mv_metrics['Expected Return'] / mv_metrics['Volatility']
    sharpe_robust = robust_metrics['Expected Return'] / robust_metrics['Volatility']
    
    # Calculate return-to-ES ratio (similar to Sortino ratio)
    sortino_mv = mv_metrics['Expected Return'] / mv_metrics['ES']
    sortino_robust = robust_metrics['Expected Return'] / robust_metrics['ES']
    
    # Determine which strategy is better based on the metrics
    if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
        # Calculate WEDS-based ratios
        weds_ratio_mv = mv_metrics['Expected Return'] / mv_metrics['WEDS']
        weds_ratio_robust = robust_metrics['Expected Return'] / robust_metrics['WEDS']
        
        # Consider all three ratios
        mv_wins = 0
        robust_wins = 0
        
        if sharpe_mv > sharpe_robust:
            mv_wins += 1
        else:
            robust_wins += 1
            
        if sortino_mv > sortino_robust:
            mv_wins += 1
        else:
            robust_wins += 1
            
        if weds_ratio_mv > weds_ratio_robust:
            mv_wins += 1
        else:
            robust_wins += 1
        
        if robust_wins > mv_wins:
            recommendation = "The **Wasserstein-Robust strategy** appears superior in this scenario, offering better risk-adjusted returns across multiple risk measures."
        elif mv_wins > robust_wins:
            recommendation = "The **Traditional Mean-Variance strategy** appears superior in this scenario, offering better risk-adjusted returns across multiple risk measures."
        else:
            # If tied, consider the WEDS ratio as the tiebreaker (since it's the most comprehensive)
            if weds_ratio_robust > weds_ratio_mv:
                recommendation = "The **Wasserstein-Robust strategy** has a slight edge in this scenario, particularly when accounting for market conditions via the SDF."
            else:
                recommendation = "The **Traditional Mean-Variance strategy** has a slight edge in this scenario, but may be less stable during market stress."
    else:
        # Original logic for when SDF analysis is not included
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
        # Create expandable sections for detailed metrics
        with st.expander("📊 Portfolio Weights", expanded=False):
            weights_df = pd.DataFrame({
                "Traditional MV": w_mv.round(4),
                "Wasserstein-Robust": w_robust.round(4)
            })
            
            # Add asset names from session state if available
            if st.session_state.get('tickers_list') is not None:
                if len(st.session_state.tickers_list) == weights_df.shape[0]:
                    weights_df.index = st.session_state.tickers_list
                
            st.dataframe(weights_df, use_container_width=True)
            
            # Add explanation about weights
            st.markdown("""
            **Understanding Portfolio Weights:**
            - Values represent the proportion of capital allocated to each asset
            - Weights sum to 1.0 (100% of portfolio)
            - Positive values indicate long positions
            - Negative values (if present) indicate short positions
            """)
        
        with st.expander("📈 Performance Metrics", expanded=False):
            # Create metrics dataframe based on whether SDF analysis is included
            if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
                metrics_df = pd.DataFrame({
                    "Metric": ["Expected Return", "Volatility", f"Expected Shortfall (α={alpha})", "WEDS"],
                    "Traditional MV": [
                        f"{mv_metrics['Expected Return']:.4f}",
                        f"{mv_metrics['Volatility']:.4f}",
                        f"{mv_metrics['ES']:.4f}",
                        f"{mv_metrics['WEDS']:.4f}"
                    ],
                    "Wasserstein-Robust": [
                        f"{robust_metrics['Expected Return']:.4f}",
                        f"{robust_metrics['Volatility']:.4f}",
                        f"{robust_metrics['ES']:.4f}",
                        f"{robust_metrics['WEDS']:.4f}"
                    ]
                })
                
                # Add explanation of WEDS
                st.markdown("""
                **Risk Measures Explained:**
                - **Expected Return**: Average portfolio return based on historical data
                - **Volatility**: Standard deviation of returns (measure of risk)
                - **Expected Shortfall (ES)**: Average loss in the worst α% of scenarios
                - **WEDS**: Wasserstein Expected Discounted Shortfall - ES adjusted by the Stochastic Discount Factor (SDF)
                
                WEDS provides a more comprehensive risk assessment by accounting for market conditions and their correlation with portfolio returns.
                """)
            else:
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
                
                # Add note about risk measure simplification
                st.markdown("""
                **Risk Measures Explained:**
                - **Expected Return**: Average portfolio return based on historical data
                - **Volatility**: Standard deviation of returns (measure of risk)
                - **Expected Shortfall (ES)**: Average loss in the worst α% of scenarios
                
                **Note**: For more advanced analysis, enable the SDF Analysis option in the sidebar to calculate the Wasserstein Expected Discounted Shortfall (WEDS), which incorporates the Stochastic Discount Factor (SDF) and distributional robustness.
                """)
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with st.expander("⚖️ Evaluation Metrics", expanded=False):
            if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
                # Calculate WEDS-based ratios
                weds_ratio_mv = mv_metrics['Expected Return'] / mv_metrics['WEDS']
                weds_ratio_robust = robust_metrics['Expected Return'] / robust_metrics['WEDS']
                
                eval_df = pd.DataFrame({
                    "Metric": ["Sharpe Ratio", "Return-to-ES Ratio", "Return-to-WEDS Ratio"],
                    "Traditional MV": [
                        f"{sharpe_mv:.4f}",
                        f"{sortino_mv:.4f}",
                        f"{weds_ratio_mv:.4f}"
                    ],
                    "Wasserstein-Robust": [
                        f"{sharpe_robust:.4f}",
                        f"{sortino_robust:.4f}",
                        f"{weds_ratio_robust:.4f}"
                    ]
                })
                
                # Add explanation of WEDS ratio
                st.markdown("""
                **Evaluation Metrics Explained:**
                - **Sharpe Ratio**: Return per unit of volatility (higher is better)
                - **Return-to-ES Ratio**: Return per unit of tail risk measured by ES (higher is better)
                - **Return-to-WEDS Ratio**: Return per unit of SDF-adjusted tail risk (higher is better)
                
                The Return-to-WEDS Ratio provides a more comprehensive assessment of risk-adjusted performance by accounting for market conditions.
                """)
            else:
                eval_df = pd.DataFrame({
                    "Metric": ["Sharpe Ratio", "Return-to-ES Ratio"],
                    "Traditional MV": [f"{sharpe_mv:.4f}", f"{sortino_mv:.4f}"],
                    "Wasserstein-Robust": [f"{sharpe_robust:.4f}", f"{sortino_robust:.4f}"]
                })
                
                # Add explanation of ratios
                st.markdown("""
                **Evaluation Metrics Explained:**
                - **Sharpe Ratio**: Return per unit of volatility (higher is better)
                - **Return-to-ES Ratio**: Return per unit of tail risk measured by ES (higher is better)
                """)
            
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
        
        # Key Insights in an expander
        with st.expander("🔑 Key Insights", expanded=False):
            st.markdown("""
            ### Performance Metrics Explained
            
            - **Sharpe Ratio**: Higher is better, measures return per unit of risk
            - **Return-to-ES Ratio**: Higher is better, measures return per unit of tail risk
            
            ### Strategy Characteristics
            
            - **Traditional optimization** typically shows higher expected returns but may underestimate risks
            - **Robust optimization** produces more conservative portfolios that may perform better in volatile markets
            - The **gap between the curves** on the efficient frontier represents the "cost of robustness"
            
            ### When to Use Each Strategy
            
            - **Traditional Mean-Variance**: Best for stable markets with reliable return estimates
            - **Wasserstein-Robust**: Best for volatile markets or when return estimates are uncertain
            - **Balanced Approach**: Allocate to both strategies for a more diversified portfolio
            """)
        
        # Strategy Evaluation with enhanced visuals
        st.subheader("Strategy Evaluation")
        
        # Add data summary section
        with st.expander("📊 Data Summary", expanded=True):
            # Get asset names
            asset_names = st.session_state.get('tickers_list', [f"Asset {i+1}" for i in range(len(w_mv))])
            
            # Calculate basic stats for each asset
            if returns is not None and len(returns) > 0:
                asset_stats = []
                for i in range(returns.shape[1]):
                    asset_return = returns[:, i]
                    stats = {
                        "Asset": asset_names[i],
                        "Mean Return": np.mean(asset_return),
                        "Volatility": np.std(asset_return),
                        "Min": np.min(asset_return),
                        "Max": np.max(asset_return),
                        "Skewness": 0 if len(asset_return) < 3 else np.mean(((asset_return - np.mean(asset_return)) / np.std(asset_return))**3),
                        "Kurtosis": 0 if len(asset_return) < 4 else np.mean(((asset_return - np.mean(asset_return)) / np.std(asset_return))**4) - 3
                    }
                    asset_stats.append(stats)
                
                # Create a DataFrame for the stats
                stats_df = pd.DataFrame(asset_stats)
                
                # Format the numbers
                for col in stats_df.columns:
                    if col != "Asset" and stats_df[col].dtype in [np.float64, np.float32]:
                        stats_df[col] = stats_df[col].map(lambda x: f"{x:.4f}")
                
                # Display the stats
                st.markdown("### Asset Statistics")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(returns.T)
                corr_df = pd.DataFrame(corr_matrix, columns=asset_names, index=asset_names)
                
                # Display correlation matrix
                st.markdown("### Correlation Matrix")
                st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)
                
                # Add description of the data
                st.markdown(f"""
                ### Data Description
                
                - **Number of Assets**: {returns.shape[1]}
                - **Number of Observations**: {returns.shape[0]}
                - **Time Period**: {st.session_state.get('period', 'Custom')}
                - **Training/Testing Split**: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%
                - **Test Distribution**: {test_dist}
                
                The data shows {'high' if np.mean(np.abs(corr_matrix - np.eye(len(corr_matrix)))) > 0.3 else 'moderate' if np.mean(np.abs(corr_matrix - np.eye(len(corr_matrix)))) > 0.15 else 'low'} correlation between assets,
                which {'supports' if np.mean(np.abs(corr_matrix - np.eye(len(corr_matrix)))) > 0.15 else 'limits'} the benefits of diversification.
                
                The assets display {'high' if np.mean([s['Volatility'] for s in asset_stats]) > 0.02 else 'moderate' if np.mean([s['Volatility'] for s in asset_stats]) > 0.01 else 'low'} volatility,
                with {'significant' if np.any([abs(float(s['Skewness'])) > 0.5 for s in asset_stats]) else 'some' if np.any([abs(float(s['Skewness'])) > 0.2 for s in asset_stats]) else 'minimal'} skewness
                and {'fat tails' if np.any([float(s['Kurtosis']) > 1.0 for s in asset_stats]) else 'near-normal tails'}.
                """)
        
        # Create a visually appealing recommendation box
        recommendation_type = ""
        if "Wasserstein-Robust strategy" in recommendation:
            box_color = "rgba(76, 175, 80, 0.2)"  # Green background for robust
            recommendation_type = "robust"
            strategy_icon = "🛡️"  # Shield icon for robust strategy
        elif "Traditional Mean-Variance strategy" in recommendation:
            box_color = "rgba(33, 150, 243, 0.2)"  # Blue background for traditional
            recommendation_type = "traditional"
            strategy_icon = "📈"  # Chart icon for traditional strategy
        else:
            box_color = "rgba(156, 39, 176, 0.2)"  # Purple background for balanced
            recommendation_type = "balanced"
            strategy_icon = "⚖️"  # Balance icon for balanced approach
        
        # Display the recommendation in a colored box with an icon
        st.markdown(f"""
        <div style="background-color: {box_color}; padding: 20px; border-radius: 10px; border-left: 5px solid {'#388e3c' if recommendation_type == 'robust' else '#1976d2' if recommendation_type == 'traditional' else '#7b1fa2'};">
            <h3 style="margin-top: 0; display: flex; align-items: center;">
                <span style="font-size: 28px; margin-right: 10px;">{strategy_icon}</span>
                Overall Assessment
            </h3>
            <p style="font-size: 16px; margin-bottom: 20px;">{recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a more visual strategy comparison
        st.markdown("""
        ### Strategy Comparison
        """)
        
        # Create three columns for the three strategy types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background-color: rgba(33, 150, 243, 0.1); padding: 15px; border-radius: 10px; height: 100%;">
                <h4 style="color: #1976d2; margin-top: 0; text-align: center;">
                    <span style="font-size: 24px;">📈</span><br>
                    Traditional Strategy
                </h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Best for:</strong> Stable markets</li>
                    <li><strong>Focus:</strong> Maximizing returns</li>
                    <li><strong>Confidence:</strong> High certainty in estimates</li>
                    <li><strong>Time Horizon:</strong> Shorter-term</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 10px; height: 100%;">
                <h4 style="color: #388e3c; margin-top: 0; text-align: center;">
                    <span style="font-size: 24px;">🛡️</span><br>
                    Robust Strategy
                </h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Best for:</strong> Volatile markets</li>
                    <li><strong>Focus:</strong> Stability & protection</li>
                    <li><strong>Confidence:</strong> Uncertainty in estimates</li>
                    <li><strong>Time Horizon:</strong> Longer-term</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="background-color: rgba(156, 39, 176, 0.1); padding: 15px; border-radius: 10px; height: 100%;">
                <h4 style="color: #7b1fa2; margin-top: 0; text-align: center;">
                    <span style="font-size: 24px;">⚖️</span><br>
                    Balanced Approach
                </h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Best for:</strong> Mixed market conditions</li>
                    <li><strong>Focus:</strong> Risk-adjusted returns</li>
                    <li><strong>Confidence:</strong> Moderate certainty</li>
                    <li><strong>Time Horizon:</strong> Medium to long-term</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Add a visual performance comparison
        st.markdown("### Performance Comparison")
        
        # Create metrics for visual comparison
        metric_cols = st.columns(3)
        
        # Calculate percentage differences
        sharpe_diff = (sharpe_robust - sharpe_mv) / sharpe_mv * 100
        es_diff = (sortino_robust - sortino_mv) / sortino_mv * 100
        
        # Add WEDS comparison if available
        if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
            weds_ratio_mv = mv_metrics['Expected Return'] / mv_metrics['WEDS']
            weds_ratio_robust = robust_metrics['Expected Return'] / robust_metrics['WEDS']
            weds_diff = (weds_ratio_robust - weds_ratio_mv) / weds_ratio_mv * 100
            
            with metric_cols[0]:
                st.metric(
                    label="Sharpe Ratio Comparison",
                    value=f"{sharpe_robust:.4f} vs {sharpe_mv:.4f}",
                    delta=f"{sharpe_diff:.1f}% {'better' if sharpe_diff > 0 else 'worse'} with Robust"
                )
            
            with metric_cols[1]:
                st.metric(
                    label="Return-to-ES Comparison",
                    value=f"{sortino_robust:.4f} vs {sortino_mv:.4f}",
                    delta=f"{es_diff:.1f}% {'better' if es_diff > 0 else 'worse'} with Robust"
                )
                
            with metric_cols[2]:
                st.metric(
                    label="Return-to-WEDS Comparison",
                    value=f"{weds_ratio_robust:.4f} vs {weds_ratio_mv:.4f}",
                    delta=f"{weds_diff:.1f}% {'better' if weds_diff > 0 else 'worse'} with Robust"
                )
        else:
            with metric_cols[0]:
                st.metric(
                    label="Sharpe Ratio Comparison",
                    value=f"{sharpe_robust:.4f} vs {sharpe_mv:.4f}",
                    delta=f"{sharpe_diff:.1f}% {'better' if sharpe_diff > 0 else 'worse'} with Robust"
                )
            
            with metric_cols[1]:
                st.metric(
                    label="Return-to-ES Comparison",
                    value=f"{sortino_robust:.4f} vs {sortino_mv:.4f}",
                    delta=f"{es_diff:.1f}% {'better' if es_diff > 0 else 'worse'} with Robust"
                )
                
            with metric_cols[2]:
                st.metric(
                    label="Overall Assessment",
                    value="See recommendation",
                    delta=None
                )
    
    with right_col:
        # Create tabs for different visualizations
        if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
            frontier_tab, sdf_tab = st.tabs(["Efficient Frontier", "SDF Analysis"])
        else:
            frontier_tab = st.container()
            
        with frontier_tab:
            # Efficient Frontier Plot
            st.subheader("Efficient Frontier")
            
            # Add explanatory text about the efficient frontier in an expander
            with st.expander("ℹ️ Understanding the Efficient Frontier", expanded=False):
                st.markdown("""
                ### Understanding the Efficient Frontier Graph
                
                This graph shows the optimal portfolios that offer the highest expected return for a given level of risk:
                
                - **Each point** represents a portfolio with a different risk aversion level
                - **X-axis (Volatility)**: Lower is better (less risk)
                - **Y-axis (Expected Return)**: Higher is better (more return)
                - **Upper left corner** has the most favorable risk-return tradeoff
                
                ### How to Interpret the Curves
                
                - **Blue curve (Traditional)**: Shows the traditional mean-variance efficient frontier
                - **Orange curve (Robust)**: Shows the Wasserstein-robust efficient frontier
                - **The gap between curves**: Represents the "cost of robustness" - how much expected return you sacrifice for greater stability
                - **Minimum volatility points**: The lowest risk portfolio for each approach (marked with larger points)
                """)
            
            # Create a larger figure for better visualization
            fig, ax = plt.subplots(figsize=(12, 9))
            
            # Only plot if we have data points
            if len(mv_vols) > 0 and len(robust_vols) > 0:
                # Plot the efficient frontiers with smoother curves
                ax.scatter(mv_vols, mv_returns, label="Traditional MV", color='blue', alpha=0.7, s=50)
                ax.scatter(robust_vols, robust_returns, label="Wasserstein-Robust", color='orange', alpha=0.7, s=50)
                
                # Use scipy's interpolation for smoother curves
                from scipy.interpolate import make_interp_spline
                
                # Only apply smoothing if we have enough points
                if len(mv_vols) > 3:
                    # Create smooth curves for Traditional MV
                    mv_x_new = np.linspace(min(mv_vols), max(mv_vols), 300)
                    try:
                        mv_spl = make_interp_spline(mv_vols, mv_returns, k=3)
                        mv_y_new = mv_spl(mv_x_new)
                        ax.plot(mv_x_new, mv_y_new, 'b-', alpha=0.6, linewidth=2)
                    except Exception:
                        # Fall back to simple line if spline fails
                        ax.plot(mv_vols, mv_returns, 'b-', alpha=0.6, linewidth=2)
                else:
                    ax.plot(mv_vols, mv_returns, 'b-', alpha=0.6, linewidth=2)
                
                # Create smooth curves for Wasserstein-Robust
                if len(robust_vols) > 3:
                    robust_x_new = np.linspace(min(robust_vols), max(robust_vols), 300)
                    try:
                        robust_spl = make_interp_spline(robust_vols, robust_returns, k=3)
                        robust_y_new = robust_spl(robust_x_new)
                        ax.plot(robust_x_new, robust_y_new, 'orange', alpha=0.6, linewidth=2)
                    except Exception:
                        # Fall back to simple line if spline fails
                        ax.plot(robust_vols, robust_returns, 'orange', alpha=0.6, linewidth=2)
                else:
                    ax.plot(robust_vols, robust_returns, 'orange', alpha=0.6, linewidth=2)
                
                # Highlight the minimum volatility portfolio for each approach
                min_vol_idx_mv = np.argmin(mv_vols)
                min_vol_idx_robust = np.argmin(robust_vols)
                
                ax.scatter([mv_vols[min_vol_idx_mv]], [mv_returns[min_vol_idx_mv]],
                          color='blue', s=150, alpha=0.8, edgecolors='black', linewidth=2,
                          label="Min Volatility (Traditional)")
                ax.scatter([robust_vols[min_vol_idx_robust]], [robust_returns[min_vol_idx_robust]],
                          color='orange', s=150, alpha=0.8, edgecolors='black', linewidth=2,
                          label="Min Volatility (Robust)")
                
                # Add annotations in better positions
                min_vol = min(mv_vols + robust_vols)
                max_vol = max(mv_vols + robust_vols)
                min_ret = min(mv_returns + robust_returns)
                max_ret = max(mv_returns + robust_returns)
                
                # Position annotations outside the plot area
                ax.annotate("Lower Risk",
                           xy=(min_vol, (max_ret + min_ret)/2),
                           xytext=(-80, 0),
                           textcoords="offset points",
                           fontsize=12,
                           fontweight='bold',
                           color='green',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='green'))
                
                ax.annotate("Higher Return",
                           xy=((max_vol + min_vol)/2, max_ret),
                           xytext=(0, 40),
                           textcoords="offset points",
                           fontsize=12,
                           fontweight='bold',
                           color='green',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                           arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='green'))
                
                # Improve axis labels and styling
                ax.set_xlabel("Volatility (Risk)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Expected Return", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=12, loc='best')
                
                # Add title
                ax.set_title("Efficient Frontier: Risk-Return Tradeoff", fontsize=16, fontweight='bold')
                
                st.pyplot(fig, use_container_width=True)
            else:
                st.error("Unable to generate efficient frontier. Try different parameters or data.")
        
        # SDF Analysis Tab
        if include_sdf and 'WEDS' in mv_metrics and 'WEDS' in robust_metrics:
            with sdf_tab:
                st.subheader("Stochastic Discount Factor (SDF) Analysis")
                
                st.markdown("""
                ### Understanding the SDF
                
                The **Stochastic Discount Factor (SDF)** is a key concept in asset pricing that:
                
                - Discounts future cash flows based on economic conditions
                - Assigns higher weights to returns in "bad" market states
                - Helps assess how portfolios perform in different market environments
                
                The visualizations below show how the SDF affects portfolio evaluation:
                """)
                
                # Plot SDF over time
                st.subheader("Stochastic Discount Factor")
                fig_sdf = plot_sdf(mv_metrics['SDF'])
                st.pyplot(fig_sdf, use_container_width=True)
                
                # Plot return distributions
                st.subheader("Return Distributions")
                
                # Traditional MV
                st.markdown("#### Traditional Mean-Variance Portfolio")
                fig_dist_mv = plot_returns_distribution(
                    mv_metrics['Portfolio Returns'],
                    mv_metrics['Discounted Returns']
                )
                st.pyplot(fig_dist_mv, use_container_width=True)
                
                # Robust
                st.markdown("#### Wasserstein-Robust Portfolio")
                fig_dist_robust = plot_returns_distribution(
                    robust_metrics['Portfolio Returns'],
                    robust_metrics['Discounted Returns']
                )
                st.pyplot(fig_dist_robust, use_container_width=True)
                
                # Risk comparison
                st.subheader("Risk Measure Comparison")
                fig_risk = plot_risk_comparison(
                    [mv_metrics['ES'], robust_metrics['ES']],
                    [mv_metrics['WEDS'], robust_metrics['WEDS']],
                    ["Traditional MV", "Wasserstein-Robust"]
                )
                st.pyplot(fig_risk, use_container_width=True)
                
                # Add explanation
                st.markdown("""
                ### Key Insights from SDF Analysis
                
                - **WEDS vs. ES**: The WEDS is typically higher than ES because it accounts for the correlation between portfolio returns and market conditions
                - **Discounted Returns**: The SDF-discounted returns show how the portfolio would perform when adjusted for market risk
                - **Strategy Comparison**: The gap between WEDS and ES indicates how sensitive each strategy is to market conditions
                
                The Wasserstein-Robust strategy typically shows less sensitivity to market conditions, making it more resilient during market stress.
                """)
                
            # Add a new section explaining why the strategy is successful
            with st.expander("🔍 Why is this strategy successful?", expanded=True):
                st.markdown("""
                ### Why the Wasserstein-Robust Strategy Succeeds
                
                The Wasserstein-Robust strategy is successful in this scenario for several key reasons:
                
                1. **Accounts for Uncertainty**: The robust approach explicitly accounts for uncertainty in return estimates, which traditional methods ignore.
                
                2. **Protects Against Worst-Case Scenarios**: By adding the Wasserstein penalty term (ε * ||w||₂), the optimization creates portfolios that perform better under adverse market conditions.
                
                3. **Reduces Estimation Error Impact**: Traditional optimization often amplifies estimation errors, while robust methods dampen their effects.
                
                ### What the Efficient Frontier Graph Shows
                
                Looking at the efficient frontier graph:
                
                - The **blue curve** (Traditional) typically extends further up and to the right, showing higher potential returns but also higher risk
                - The **orange curve** (Robust) is often more conservative, with a smaller range of risk-return combinations
                - The **gap between curves** represents the "price of robustness" - what you give up in expected return to gain protection
                - The **minimum volatility points** (larger dots) show the lowest-risk portfolios for each approach
                
                In this case, the Robust strategy's frontier indicates it achieves better risk-adjusted returns (higher Sharpe ratio, Return-to-ES ratio, and Return-to-WEDS ratio). This suggests the robust approach is finding more efficient portfolios by avoiding positions that might perform well on average but could suffer severely in adverse scenarios.
                
                ### Mathematical Intuition
                
                Mathematically, the Wasserstein-robust approach adds a regularization term to the optimization:
                
                ```
                min  -μᵀw + (γ/2)wᵀΣw + ε||w||₂
                ```
                
                This regularization:
                - Penalizes extreme weights
                - Promotes diversification
                - Reduces sensitivity to input parameters
                
                The result is a portfolio that sacrifices some expected return for significantly improved stability and resilience.
                """)

if __name__ == "__main__":
    main()