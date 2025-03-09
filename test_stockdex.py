import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stockdex import Ticker

def main():
    st.set_page_config(layout="wide")
    st.title("StockDex Test App")
    
    st.sidebar.header("Settings")
    ticker_symbol = st.sidebar.text_input("Enter ticker symbol", "SPY")
    
    if st.sidebar.button("Fetch Data"):
        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            try:
                # Create a ticker object
                ticker = Ticker(ticker=ticker_symbol)
                
                # Display basic information
                st.subheader(f"Data for {ticker_symbol}")
                
                # Create tabs for different data sources
                tab1, tab2, tab3 = st.tabs(["Yahoo API Data", "Price Data", "Financial Data"])
                
                with tab1:
                    st.subheader("Yahoo API Data")
                    try:
                        # Try to get price data
                        price_data = ticker.yahoo_api_price(range='1y', dataGranularity='1d')
                        st.write("Price Data (Last 5 rows):")
                        st.dataframe(price_data.tail(), use_container_width=True)
                        
                        # Plot the price data
                        st.subheader("Price Chart")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(price_data.index, price_data['close'], label='Close Price')
                        ax.set_title(f"{ticker_symbol} Price (Last Year)")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price")
                        ax.grid(True)
                        ax.legend()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error fetching Yahoo API data: {str(e)}")
                        st.info("This could be due to API limitations or connectivity issues.")
                
                with tab2:
                    st.subheader("Price Data from Digrin")
                    try:
                        # Try to get price data from Digrin
                        digrin_price = ticker.digrin_price
                        st.write("Historical Price Data (Last 5 rows):")
                        st.dataframe(digrin_price.tail(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error fetching Digrin price data: {str(e)}")
                        st.info("This could be due to web scraping limitations or connectivity issues.")
                
                with tab3:
                    st.subheader("Financial Data")
                    try:
                        # Try to get financial data
                        income_statement = ticker.yahoo_api_income_statement(frequency='quarterly')
                        st.write("Income Statement (Last 5 rows):")
                        st.dataframe(income_statement.tail(), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error fetching financial data: {str(e)}")
                        st.info("This could be due to API limitations or connectivity issues.")
                
                # Display any errors or warnings
                st.subheader("Debug Information")
                st.info("This section shows any errors or warnings encountered during data fetching.")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please try a different ticker symbol or check your internet connection.")
    else:
        st.info("Enter a ticker symbol and click 'Fetch Data' to begin.")
        st.write("Example tickers: SPY, QQQ, DIA, AAPL, MSFT")

if __name__ == "__main__":
    main()