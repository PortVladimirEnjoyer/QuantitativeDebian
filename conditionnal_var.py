#note(disclaimer) ai was used for this code 

import numpy as np
import yfinance as yf
from scipy.stats import norm

def compute_cvar(losses, alpha):
    """Compute VaR and CVaR for a given loss distribution."""
    var = np.quantile(losses, alpha)
    cvar = losses[losses >= var].mean()
    return var, cvar

def main():
    print("\n=== Conditional Value at Risk (CVaR) Calculator ===")
    
    # Step 1: Ask for confidence level (e.g., 95%)
    while True:
        try:
            confidence = float(input("\nEnter confidence level (e.g., 95 for 95%): ")) / 100
            if 0 < confidence < 1:
                break
            else:
                print("Error: Please enter a value between 1 and 99.")
        except ValueError:
            print("Error: Please enter a valid number.")

    # Step 2: Ask for data source
    print("\nChoose loss data source:")
    print("1. Randomly generated (Normal distribution)")
    print("2. Historical stock/crypto losses (Yahoo Finance)")
    choice = input("Enter 1 or 2: ")

    # Step 3: Generate losses based on choice
    if choice == "1":
        # Random losses (Normal distribution)
        mu = float(input("Enter mean loss (e.g., 0.02): "))
        sigma = float(input("Enter loss volatility (e.g., 0.05): "))
        losses = -np.random.normal(mu, sigma, 10_000)
        data_source = f"Random Normal Data (μ={mu}, σ={sigma})"
    elif choice == "2":
        # Historical losses (user-provided ticker)
        ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, BTC-USD): ").upper()
        try:
            data = yf.download(ticker, period="1y")["Close"]
            losses = -data.pct_change().dropna()
            data_source = f"Historical {ticker} Losses"
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    # Step 4: Compute VaR and CVaR
    var, cvar = compute_cvar(losses, confidence)
    
    # Step 5: Display results
    print("\n=== Results ===")
    print(f"Data Source: {data_source}")
    print(f"Confidence Level: {confidence*100:.1f}%")
    print(f"VaR (Threshold Loss): {var:.6f}")
    print(f"CVaR (Avg Loss Beyond VaR): {cvar:.6f}")

if __name__ == "__main__":
    main()
