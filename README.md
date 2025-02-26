# Arun-Dahal-Khatri
Black-Scholes Model using Python for  Calculating the Call and Put Option Prices

//Required Libraries:

import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

# Import necessary libraries
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt

Black-Scholes Model and Greeks Calculation:
# Black-Scholes Option Pricing Function
def black_scholes(S, X, T, r, sigma, option_type='call'):
    """
    Calculate European option price using the Black-Scholes model.
    
    Parameters:
    S : float : Current stock price
    X : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    option_type : str : 'call' for Call option, 'put' for Put option
    
    Returns:
    float : Option price
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * si.norm.cdf(d1, 0.0, 1.0) - X * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif option_type == 'put':
        price = X * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
    return price

# Greeks Calculation
def greeks(S, X, T, r, sigma, option_type='call'):
    """
    Calculate Greeks for European options using Black-Scholes model.
    
    Returns:
    dict : Greeks (Delta, Gamma, Vega, Theta, Rho)
    """
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = si.norm.cdf(d1, 0.0, 1.0) if option_type == 'call' else -si.norm.cdf(-d1, 0.0, 1.0)
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    theta = (- (S * si.norm.pdf(d1, 0.0, 1.0) * sigma) / (2 * np.sqrt(T))
             - r * X * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    rho = X * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) if option_type == 'call' else -X * T * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0)
    
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

Sensitivity Analysis and Visualization:
# Sensitivity Analysis
def sensitivity_analysis(S, X, T, r, sigma, option_type='call'):
    stock_prices = np.linspace(50, 150, 100)
    option_prices = [black_scholes(S, X, T, r, sigma, option_type) for S in stock_prices]

    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, option_prices, label='Option Price')
    plt.title(f'Sensitivity Analysis of {option_type.capitalize()} Option')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
S = 100  # Current stock price
X = 110  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free rate (5%)
sigma = 0.2  # Volatility (20%)

call_price = black_scholes(S, X, T, r, sigma, option_type='call')
put_price = black_scholes(S, X, T, r, sigma, option_type='put')

print("Call Option Price:", call_price)
print("Put Option Price:", put_price)

call_greeks = greeks(S, X, T, r, sigma, option_type='call')
put_greeks = greeks(S, X, T, r, sigma, option_type='put')

print("\nCall Option Greeks:", call_greeks)
print("Put Option Greeks:", put_greeks)

# Sensitivity Analysis
sensitivity_analysis(S, X, T, r, sigma, option_type='call')
sensitivity_analysis(S, X, T, r, sigma, option_type='put')
