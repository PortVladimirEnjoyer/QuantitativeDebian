#note(disclaimer) : ai was used for this code 


import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

def main_garch():
    print("\n=== GARCH Volatility Model ===")
    
    # Step 1: Ask for data source
    print("\nChoose data source:")
    print("1. Simulated returns (random)")
    print("2. Real-world asset (Yahoo Finance)")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        # Simulate random returns
        n = int(input("Number of data points (e.g., 1000): "))
        mu = float(input("Mean return (e.g., 0.0): "))
        sigma = float(input("Initial volatility (e.g., 0.01): "))
        returns = np.random.normal(mu, sigma, n)
        data_name = "Simulated Data"
    elif choice == "2":
        # Fetch real asset data
        ticker = input("Enter ticker (e.g., AAPL, BTC-USD): ").upper()
        data = yf.download(ticker, period="1y")["Close"]
        returns = 100 * data.pct_change().dropna()  # Convert to % returns
        data_name = f"{ticker} Returns"
    else:
        print("Invalid choice. Exiting.")
        return

    # Step 2: Ask for GARCH parameters
    print("\nSelect GARCH model type:")
    print("1. GARCH")
    print("2. EGARCH (asymmetric effects)")
    print("3. GJR-GARCH (leverage effects)")
    model_type = input("Enter 1, 2, or 3: ").lower()

    p = int(input("GARCH order (p, e.g., 1): "))  # Lag for volatility
    q = int(input("ARCH order (q, e.g., 1): "))   # Lag for shocks

    # Step 3: Fit the model
    if model_type == "1":
        model = arch_model(returns, vol="GARCH", p=p, q=q)
    elif model_type == "2":
        model = arch_model(returns, vol="EGARCH", p=p, q=q)
    elif model_type == "3":
        model = arch_model(returns, vol="GJR", p=p, q=q)
    else:
        print("Invalid model type. Using GARCH(1,1) by default.")
        model = arch_model(returns, vol="GARCH", p=1, q=1)

    result = model.fit(update_freq=5, disp="off")
    print("\n=== Model Results ===")
    print(result.summary())

    # Step 4: Forecast volatility
    forecast_horizon = int(input("\nForecast horizon (days, e.g., 5): "))
    forecasts = result.forecast(horizon=forecast_horizon)
    print(f"\nForecasted Volatility (next {forecast_horizon} days):")
    print(forecasts.variance.iloc[-1].apply(np.sqrt))  # Convert variance to std dev

    # Step 5: Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Returns")
    plt.plot(result.conditional_volatility, label="GARCH Volatility", color="red")
    plt.title(f"{data_name} - {model.volatility.__class__.__name__}({p},{q})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main_garch()
