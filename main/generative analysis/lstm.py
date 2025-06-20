#note(disclaimer) : ai was used to compose this code 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

def get_user_input():
    """Get user input for stock ticker and date range"""
    print("Stock Price Prediction System")
    print("----------------------------")
    
    ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOG): ").upper()
    
    while True:
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        
        try:
            # Validate dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start >= end:
                print("Error: Start date must be before end date. Please try again.")
                continue
                
            if end > datetime.now():
                print("Warning: End date is in the future. Using current date instead.")
                end_date = datetime.now().strftime("%Y-%m-%d")
                
            return ticker, start_date, end_date
            
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")

def get_stock_data(ticker, start, end):
    """Load stock data from Yahoo Finance"""
    print(f"\nFetching data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker} in the specified date range.")
    
    return df[['Close', 'Volume']]

def engineer_features(data):
    """Add technical indicators to the data"""
    data['Close_MA50'] = data['Close'].rolling(window=50).mean()
    data['Close_MA200'] = data['Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    return data

def create_dataset(data, target, time_step=50):
    """Create time-series dataset for LSTM"""
    X, Y = [], []
    for i in range(len(data) - time_step - 10):
        X.append(data[i:(i + time_step)])
        Y.append(target[i + time_step:i + time_step + 10].flatten())
    return np.array(X), np.array(Y)

def build_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(10)  # Predict next 10 days
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    return model

def main():
    try:
        # Get user input
        ticker, start_date, end_date = get_user_input()
        
        # Fetch and prepare data
        data = get_stock_data(ticker, start_date, end_date)
        data = engineer_features(data)
        
        # Check if we have enough data
        if len(data) < 260:  # About 1 year of trading days
            print(f"\nWarning: Only {len(data)} days of data available. For better results, use at least 1 year of data.")
            proceed = input("Continue anyway? (y/n): ").lower()
            if proceed != 'y':
                print("Exiting...")
                return

        # Scale data
        close_scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaled_close = close_scaler.fit_transform(data[['Close']])
        data_scaled = data_scaler.fit_transform(data)

        # Prepare dataset
        time_step = 50
        dataX, dataY = create_dataset(data_scaled, scaled_close, time_step)
        
        if len(dataX) == 0:
            raise ValueError("Not enough data to create training samples. Try a longer date range.")
            
        # Split data
        split_idx = int(len(dataX) * 0.8)
        X_train, y_train = dataX[:split_idx], dataY[:split_idx]
        X_test, y_test = dataX[split_idx:], dataY[split_idx:]

        # Reshape data
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # Build and train model
        print("\nBuilding and training the LSTM model...")
        model = build_model((time_step, X_train.shape[2]))
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                          validation_data=(X_test, y_test), verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions
        y_pred = close_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
        y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

        # Calculate metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, y_pred)

        # Display results
        print("\nModel Evaluation Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        # Show next 10 day prediction
        last_prediction = y_pred[-1]
        last_date = data.index[-1]
        prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10)
        
        print("\nNext 10 Trading Day Predictions:")
        for i, (date, price) in enumerate(zip(prediction_dates, last_prediction), 1):
            print(f"Day {i} ({date.strftime('%Y-%m-%d')}): ${price:.2f}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual[:, 0], label='Actual Prices', color='blue')
        plt.plot(y_pred[:, 0], label='Predicted Prices', color='red')
        plt.title(f'{ticker} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    main()
