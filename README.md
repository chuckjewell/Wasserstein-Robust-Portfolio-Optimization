# Wasserstein-Robust Portfolio Optimization

A Streamlit application for comparing traditional Mean-Variance portfolio optimization with Wasserstein-Robust optimization strategies.

![Portfolio Optimization App](https://github.com/your-username/portfolio-optimization/raw/main/screenshot.png)

## Features

- **Multiple Data Sources**:
  - Fetch real stock data using the StockDex library
  - Upload your own returns data via CSV
  - Generate synthetic market data for testing

- **Randomize Feature**:
  - Quickly test with 5 random S&P 500 stocks

- **Interactive Parameters**:
  - Adjust risk aversion, robustness, and confidence levels
  - Toggle short selling constraints
  - Configure training/testing scenarios

- **Comprehensive Analysis**:
  - Portfolio weights visualization
  - Performance metrics comparison
  - Efficient frontier visualization
  - Strategy evaluation and recommendations

- **Educational Content**:
  - Explanations of portfolio optimization concepts
  - Interpretation guides for the efficient frontier
  - Performance metrics explanations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/portfolio-optimization.git
   cd portfolio-optimization
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Or use the provided launch script:
   ```bash
   chmod +x launch_portfolio_app.sh
   ./launch_portfolio_app.sh
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app_stockdex.py
   ```

2. Select a data source:
   - **Stock Data**: Enter ticker symbols (e.g., "SPY QQQ DIA") or use the randomize button
   - **Upload CSV**: Upload a CSV file with returns data
   - **Generate Synthetic**: Create simulated market data

3. Adjust parameters:
   - **Risk aversion (γ)**: Controls the trade-off between risk and return
   - **Wasserstein radius (ε)**: Controls the level of robustness
   - **Confidence level (α)**: Used for Expected Shortfall calculation

4. Analyze results:
   - Compare portfolio weights
   - Examine performance metrics
   - Study the efficient frontier
   - Review the strategy evaluation

## Dependencies

- streamlit
- numpy
- pandas
- matplotlib
- scipy
- stockdex
- cvxpy (for optimization)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on research in robust portfolio optimization using Wasserstein distance
- Utilizes the StockDex library for fetching financial data