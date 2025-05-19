import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to download stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']

# Function to train ARIMA model and make predictions
def arima_prediction(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train, test = data[:train_size], data[train_size:]
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(0, 1, 0))  # Setting p=0, d=1, q=0
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    r_squared = r2_score(test, predictions)
    return predictions, mse, rmse, r_squared

# Function to plot rolling mean and exponential mean
def plot_means(data):
    rolling_mean = data.rolling(window=12).mean()
    exponential_mean = data.ewm(span=12, adjust=False).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual', color='blue')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(exponential_mean, label='Exponential Mean', color='green')
    plt.title('Rolling Mean and Exponential Mean')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function to plot autocorrelation and partial autocorrelation
def plot_acf_pacf(data):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data, ax=ax[0], lags=30)
    plot_pacf(data, ax=ax[1], lags=30)
    plt.show()

# Main function
def main():
    # Download stock data
    ticker = 'AAPL'  # Change the ticker symbol as per your requirement
    start_date = '2017-01-01'
    end_date = '2020-01-01'
    data = download_stock_data(ticker, start_date, end_date)
    
    # Plot rolling mean and exponential mean
    plot_means(data)
    
    # Plot autocorrelation and partial autocorrelation
    plot_acf_pacf(data)
    
    # Train ARIMA model and make predictions
    train_ratio = 0.8
    predictions, mse, rmse, r_squared = arima_prediction(data, train_ratio)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared Error:", r_squared)

    # Split data for plotting
    train_size = int(len(data) * train_ratio)
    train_data, test_data = data[:train_size], data[train_size:]
    test_dates = data.index[train_size:]

    # Plot expected vs actual values for the last 10 days
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data, label='Actual', color='blue')
    plt.plot(test_dates, predictions, label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print the last 10 actual and predicted values
    print("\nActual vs Predicted (Last 10 Days):")
    print(pd.DataFrame({'Actual': test_data[-10:], 'Predicted': predictions[-10:]}))

if __name__ == "__main__":
    main()
