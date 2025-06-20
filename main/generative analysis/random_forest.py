#note(disclaimer) ai was used for the construction of this code 

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                           r2_score, explained_variance_score)
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
from scipy.stats import randint
import time
from datetime import datetime, timedelta

def get_user_input():
    """Get user input for stock prediction parameters"""
    print("\n📈 Stock Price Prediction System")
    print("--------------------------------")
    
    # Get ticker symbol
    ticker = input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOG): ").upper()
    
    # Get date range with validation
    while True:
        try:
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD, or press Enter for today): ") or datetime.now().strftime('%Y-%m-%d')
            
            # Validate dates
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start >= end:
                print("Error: Start date must be before end date. Please try again.")
                continue
                
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")
    
    # Get forecast days
    while True:
        try:
            forecast_days = int(input("Enter number of days to forecast (1-30): "))
            if 1 <= forecast_days <= 30:
                break
            print("Please enter a number between 1 and 30.")
        except ValueError:
            print("Please enter a valid number.")
    
    return ticker, start_date, end_date, forecast_days

def get_stock_data(ticker, start_date, end_date):
    """Download and preprocess stock data"""
    print(f"\n📊 Fetching {ticker} data from {start_date} to {end_date}")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError("No data returned - check ticker symbol and date range")
    return data

def create_features(data, window=5):
    """Create technical features for prediction"""
    df = data[['Close']].copy()

    # Lagged features
    for i in range(1, window+1):
        df[f'Lag_{i}'] = df['Close'].shift(i)

    # Technical indicators
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['EMA_5'] = df['Close'].ewm(span=5).mean()
    df['Momentum'] = df['Close'].pct_change(3)

    df['Target'] = df['Close'].shift(-1)  # Next day's price
    df.dropna(inplace=True)
    return df

def train_optimized_rf(X_train, y_train):
    """Hyperparameter optimization with feature importance"""
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', 0.5, 0.8],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    print("\n🔍 Optimizing Random Forest hyperparameters...")
    start = time.time()
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=5,
        n_jobs=-1, random_state=42,
        scoring='neg_mean_squared_error'
    ).fit(X_train, y_train)
    print(f"⚡ Optimization completed in {time.time()-start:.2f}s")

    best_model = search.best_estimator_
    print("\n🏆 Best Parameters Found:")
    print(tabulate([[k,v] for k,v in search.best_params_.items()],
                  headers=['Parameter', 'Value'], tablefmt='grid'))

    return best_model

def generate_predictions(model, X, days=5):
    """Generate multi-step predictions with uncertainty estimation"""
    predictions = []
    current_features = X[-1].copy()

    for _ in range(days):
        # Get predictions from all trees for uncertainty
        preds = [tree.predict([current_features])[0]
                for tree in model.estimators_]
        mean_pred = np.mean(preds)
        std_pred = np.std(preds)

        predictions.append({
            'prediction': mean_pred,
            'lower_bound': mean_pred - 1.96*std_pred,
            'upper_bound': mean_pred + 1.96*std_pred
        })

        # Update features
        current_features = np.roll(current_features, -1)
        current_features[-1] = mean_pred

    return predictions

def run_prediction_pipeline(ticker, start_date, end_date, forecast_days):
    """Complete prediction pipeline with error handling"""
    try:
        # 1. Data Preparation
        data = get_stock_data(ticker, start_date, end_date)
        df = create_features(data)

        # Check if we have enough data
        if len(df) < 30:
            print(f"\n⚠️ Warning: Only {len(df)} days of data available. For better results, use at least 3 months of data.")
            proceed = input("Continue anyway? (y/n): ").lower()
            if proceed != 'y':
                print("Exiting...")
                return None, None, None

        # 2. Train-Test Split
        X = df.drop(['Close', 'Target'], axis=1)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        # 3. Model Training
        model = train_optimized_rf(X_train.values, y_train.values)

        # 4. Test Evaluation
        y_pred = model.predict(X_test.values)

        # 5. Future Forecasting
        last_window = X.values[-1]
        future_preds = generate_predictions(model, X.values, forecast_days)

        # 6. Results Compilation
        test_results = pd.DataFrame({
            'Date': X_test.index,
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred
        })

        future_dates = [df.index[-1] + timedelta(days=i)
                       for i in range(1, forecast_days+1)]
        forecast_results = pd.DataFrame({
            'Date': future_dates,
            'Predicted': [p['prediction'] for p in future_preds],
            'Lower_CI': [p['lower_bound'] for p in future_preds],
            'Upper_CI': [p['upper_bound'] for p in future_preds]
        })

        # 7. Enhanced Metrics
        directional_accuracy = np.mean(
            np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))) * 100

        metrics = [
            ["R² Score", r2_score(y_test, y_pred)],
            ["Explained Variance", explained_variance_score(y_test, y_pred)],
            ["Mean Absolute Error", mean_absolute_error(y_test, y_pred)],
            ["Root Mean Squared Error", np.sqrt(mean_squared_error(y_test, y_pred))],
            ["Directional Accuracy (%)", directional_accuracy],
            ["Mean Absolute Percentage Error",
             np.mean(np.abs((y_test - y_pred)/y_test)) * 100]
        ]

        # 8. Enhanced Visualization
        plt.figure(figsize=(16, 8))

        # Historical data
        plt.plot(df.index, df['Close'], label='Historical Prices',
                color='navy', alpha=0.8, linewidth=2)

        # Test predictions
        plt.plot(test_results['Date'], test_results['Predicted'],
                label='Test Predictions', color='green', linestyle='--', linewidth=2)

        # Future forecast with confidence interval
        plt.plot(forecast_results['Date'], forecast_results['Predicted'],
                label=f'{forecast_days}-Day Forecast', color='red', marker='o', markersize=8)
        plt.fill_between(forecast_results['Date'],
                        forecast_results['Lower_CI'],
                        forecast_results['Upper_CI'],
                        color='pink', alpha=0.3, label='95% Confidence Interval')

        # Formatting
        plt.title(f'{ticker} Stock Price Prediction\n{start_date} to {end_date} | Forecast: {forecast_days} Days',
                 fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price ($)', fontsize=14)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)

        # 9. Enhanced Output
        print("\n📈 MODEL PERFORMANCE METRICS:")
        print(tabulate(metrics, headers=["Metric", "Value"],
                      tablefmt="grid", floatfmt=".4f"))

        print("\n🔮 FUTURE PRICE PREDICTIONS:")
        forecast_display = forecast_results.copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
        print(tabulate(forecast_display, headers='keys',
                      tablefmt='grid', showindex=False, floatfmt=".2f"))

        plt.tight_layout()
        plt.savefig(f'{ticker}_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()

        return model, test_results, forecast_results

    except Exception as e:
        print(f"\n❌ Error encountered: {str(e)}")
        print("Please check your inputs and try again.")
        return None, None, None

def main():
    """Main function to run the prediction system"""
    print("\n💹 Advanced Stock Price Prediction System")
    print("----------------------------------------")
    
    # Get user inputs
    ticker, start_date, end_date, forecast_days = get_user_input()
    
    # Run prediction pipeline
    model, test_results, forecast = run_prediction_pipeline(
        ticker, start_date, end_date, forecast_days
    )
    
    if model is not None:
        print("\n✅ Prediction completed successfully!")
        print(f"Results saved to {ticker}_forecast.png")

if __name__ == "__main__":
    main()
