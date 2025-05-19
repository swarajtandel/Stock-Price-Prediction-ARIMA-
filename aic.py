import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

# Function to download stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to select the best ARIMA parameters using grid search
def find_best_arima_params(data, p_range, d_range, q_range):
    best_aic, best_params = float("inf"), None
    for p, d, q in product(p_range, d_range, q_range):
        try:
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            if aic < best_aic:
                best_aic, best_params = aic, (p, d, q)
        except:
            continue
    return best_params, best_aic

# Main function
def main():
    # Download stock data
    ticker = 'AAPL'  # Change the ticker symbol as per your requirement
    start_date = '2017-01-01'
    end_date = '2020-01-01'
    data = download_stock_data(ticker, start_date, end_date)['Close']
    
    # Initialize variables to store best AIC and its corresponding parameters
    best_aic = float("inf")
    best_params = None
    
    # Loop through possible ARIMA orders
    p_range = range(3)
    d_range = range(3)
    q_range = range(3)
    for p in p_range:
        for d in d_range:
            for q in q_range:
                # Skip orders that violate Box-Jenkins method
                if p + q <= 2:
                    params = (p, d, q)
                    try:
                        _, aic = find_best_arima_params(data, [p], [d], [q])
                        # Update best AIC and parameters if found
                        if aic < best_aic:
                            best_aic = aic
                            best_params = params
                    except:
                        continue
    
    # Print best AIC and its corresponding parameters
    print("Best AIC:", best_aic)
    print("Best ARIMA parameters:", best_params)

if __name__ == "__main__":
    main()
