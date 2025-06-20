import numpy as np
import matplotlib.pyplot as plt

def monte_carlo():
    print("Monte Carlo Simulation for Stock Market & Crypto")

    # User inputs
    asset_name = input("Enter the asset name (e.g., BTC, AAPL): ")
    S0 = float(input(f"Enter the initial price of {asset_name}: "))
    mu = float(input(f"Enter the expected annual return of {asset_name} (e.g., 0.05 for 5%): "))
    sigma = float(input(f"Enter the annual volatility of {asset_name} (e.g., 0.2 for 20%): "))
    T = float(input("Enter the time horizon in years: "))
    dt = 1/252  # Daily steps (assuming 252 trading days per year)
    num_simulations = int(input("Enter the number of simulations: "))

    # Derived parameters
    num_steps = int(T / dt)  # Number of time steps

    # Monte Carlo simulation
    simulations = np.zeros((num_steps, num_simulations))
    simulations[0] = S0

    for t in range(1, num_steps):
        Z = np.random.standard_normal(num_simulations)  # Random normal values
        simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Plot simulated price paths
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # First subplot for price paths
    plt.plot(simulations, lw=1)
    plt.title(f"Monte Carlo Simulation of {asset_name} Prices")
    plt.xlabel("Trading Days")
    plt.ylabel(f"{asset_name} Price")

    # Plot distribution of final prices
    final_prices = simulations[-1]  # Final prices from all simulations
    plt.subplot(1, 2, 2)  # Second subplot for distribution
    plt.hist(final_prices, bins=50, color='blue', alpha=0.7)
    plt.title(f"Distribution of {asset_name} Final Prices")
    plt.xlabel(f"Final {asset_name} Price")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# This block ensures the code only runs when executed directly
if __name__ == "__main__":
    monte_carlo()
