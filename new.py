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
    
    if data.empty:
        raise ValueError(f"No data found for {ticker} in the given date range.")
    
    data = data[['Close']].dropna()  # Keep only the 'Close' column and drop NaN values
    return data

# Function to train ARIMA model and make predictions
def arima_prediction(data, train_ratio):
    train_size = int(len(data) * train_ratio)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    history = train['Close'].tolist()  # Convert Series to list
    predictions = []

    for t in range(len(test)):
        model = ARIMA(history, order=(0, 1, 0))  # p=0, d=1, q=0
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t]['Close'])  # Append next test value

    mse = mean_squared_error(test['Close'], predictions)
    rmse = np.sqrt(mse)
    r_squared = r2_score(test['Close'], predictions)
    
    return predictions, mse, rmse, r_squared, test.index  # Return test dates for plotting

# Function to plot rolling mean and exponential mean
def plot_means(data):
    rolling_mean = data['Close'].rolling(window=12).mean()
    exponential_mean = data['Close'].ewm(span=12, adjust=False).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual', color='blue')
    plt.plot(data.index, rolling_mean, label='Rolling Mean', color='red')
    plt.plot(data.index, exponential_mean, label='Exponential Mean', color='green')
    plt.title('Rolling Mean and Exponential Mean')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function to plot autocorrelation and partial autocorrelation
def plot_acf_pacf(data):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(data['Close'], ax=ax[0], lags=30)
    plot_pacf(data['Close'], ax=ax[1], lags=30)
    plt.show()

# Main function
def main():
    ticker = 'AAPL'  # Change ticker as needed
    start_date = '2017-01-01'
    end_date = '2020-01-01'
    
    try:
        data = download_stock_data(ticker, start_date, end_date)
    except ValueError as e:
        print(e)
        return

    # Plot rolling mean and exponential mean
    plot_means(data)
    
    # Plot autocorrelation and partial autocorrelation
    plot_acf_pacf(data)
    
    # Train ARIMA model and make predictions
    train_ratio = 0.8
    predictions, mse, rmse, r_squared, test_dates = arima_prediction(data, train_ratio)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Error: {r_squared:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Actual', color='blue')
    plt.plot(test_dates, predictions, label='Predicted', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print last 10 actual vs predicted values
    print("\nActual vs Predicted (Last 10 Days):")
    print(pd.DataFrame({'Actual': data.iloc[-10:]['Close'].values, 'Predicted': predictions[-10:]}))

if __name__ == "__main__":
    main()
