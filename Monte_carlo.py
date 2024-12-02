import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend for non-interactive environments
matplotlib.use('Agg')

# Application Title
st.title("Beta Project: Monte-Carlo simulation of leveraged equity portfolio 🎲")
st.subheader("Made by Konsta Sutinen, Aaro Tuominen, Elias Vanninen and Kalle Juven")

# User Input for Simulation Parameters
# User Input for Initial Investment with a minimum constraint
initial_investment = st.number_input("Initial Investment ($)", value=1000, step=100, min_value=100)

# Check and warn if the user tries to set an invalid value
if initial_investment < 100:
    st.error("Initial investment must be at least $100.")

percentage_of_leverage_in_portfolio = st.slider(
    "Leverage as % of Portfolio", min_value=0, max_value=95, value=50
)
time_horizon_years = st.slider("Time Horizon (Years)", min_value=1, max_value=10, value=5)
start_rate = st.number_input(
    "Starting Interest Rate (%)", 
    value=4.58,  # Default assumption
    step=0.01, 
    format="%.2f"
) / 100  # Convert percentage to decimal

# User Input for Interest Rate Margin as a percentage input
interest_rate_margin = st.number_input(
    "Interest Rate Margin (%)", 
    value=1.0,  # Default value
    step=0.1,   # Increment step
    format="%.2f",  # Format to 2 decimal places
    min_value=0.0,  # Minimum value
    max_value=10.0  # Maximum value
) / 100  # Convert to decimal

num_simulations = st.selectbox("Number of Simulations", options=[1, 100, 1000, 10000], index=2)

# Leverage Calculation
leverage = initial_investment / (1 - (percentage_of_leverage_in_portfolio / 100)) - initial_investment
portfolio_start_value = initial_investment + leverage

# Donut Chart to Visualize Portfolio Allocation
st.write("### Portfolio Allocation")

# Data for the chart
data = [initial_investment, leverage]
labels = ["Equity ", "Leverage "]

# Create the figure
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    data, labels=labels, autopct='%1.1f%%', startangle=90, textprops=dict(color="w")
)
ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

# Annotate the chart with dollar values
total_portfolio = initial_investment + leverage
for i, value in enumerate(data):
    percentage = value / total_portfolio * 100
    ax.text(
        0,
        -1.2 - (i * 0.1),  # Adjust position for each value
        f"{labels[i]}: ${value:,.2f} ({percentage:.1f}%)",
        horizontalalignment='center',
        fontsize=10,
        color='black',
    )

# Display the chart
st.pyplot(fig)

# Button to Trigger Simulation
simulate_button = st.button("Simulate")
if simulate_button:
    st.markdown("## 🎯 Simulation Results")


    # Vasicek Model Simulation Function
    def vasicek_simulation(start_rate, mean_rate, kappa, sigma, num_simulations, forecast_period):
        dt = 1 / 252
        random_shocks = np.random.normal(0, 1, (forecast_period, num_simulations))
        rate_paths = np.zeros((forecast_period + 1, num_simulations))
        rate_paths[0] = start_rate
        for t in range(1, forecast_period + 1):
            dr = kappa * (mean_rate - rate_paths[t - 1]) * dt + sigma * random_shocks[t - 1] * np.sqrt(dt)
            rate_paths[t] = rate_paths[t - 1] + dr
        return rate_paths


    # Simulation Parameters
    avg_daily_return = 0.0360 / 100
    daily_volatility = 0.010806037
    trading_days_per_year = 252
    total_days = time_horizon_years * trading_days_per_year

    # Vasicek Model Parameters
    mean_rate = 0.0461
    kappa = -0.003794768
    sigma = 0.003141746

    # Simulate Interest Rates
    interest_rate_paths = vasicek_simulation(
        start_rate=start_rate,
        mean_rate=mean_rate,
        kappa=kappa,
        sigma=sigma,
        num_simulations=num_simulations,
        forecast_period=total_days,
    )

    # Daily Returns
    daily_returns = np.random.normal(avg_daily_return, daily_volatility, (total_days, num_simulations))

    # Initialize Portfolios
    portfolio_values = np.zeros((total_days + 1, num_simulations))
    portfolio_values2 = np.zeros((total_days + 1, num_simulations))
    portfolio_values[0] = portfolio_start_value
    portfolio_values2[0] = initial_investment

    # Initialize equity array for plotting
    equity_values_all = np.zeros((total_days + 1, num_simulations))

    # Simulate Portfolio with Updated Logic
    for t in range(1, total_days + 1):
        # Update portfolio with daily returns and deduct daily interest
        portfolio_values[t] = portfolio_values[t - 1] * (1 + daily_returns[t - 1]) - (
                leverage * ((interest_rate_paths[t] + interest_rate_margin) / trading_days_per_year)
        )
        portfolio_values2[t] = portfolio_values2[t - 1] * (1 + daily_returns[t - 1])

        # Calculate equity (portfolio value - leverage) and ensure it does not go negative
        equity_values = np.maximum(portfolio_values[t] - leverage, 0)

        # Apply margin call: If equity <= 0, set equity and portfolio to 0
        margin_call_occurred = equity_values <= 0
        portfolio_values[t] = np.where(margin_call_occurred, 0, portfolio_values[t])

        # Store equity values for plotting
        equity_values_all[t] = equity_values

    # Calculate final values
    final_equity_values = np.maximum(portfolio_values[-1] - leverage, 0)
    final_portfolio_values2 = portfolio_values2[-1]

    # Calculate percentiles
    percentiles = np.percentile(final_equity_values, [10, 25, 50, 75, 90])
    percentiles2 = np.percentile(final_portfolio_values2, [10, 25, 50, 75, 90])

    # Margin Call Percentage
    margin_call_percentage = (portfolio_values[-1] == 0).sum() / num_simulations * 100

    st.metric(label="Margin Call Rate", value=f"{margin_call_percentage:.2f}%", )
    st.metric(label="Median Levered Portfolio", value=f"${np.median(final_equity_values):,.2f}")
    st.metric(label="Median Unlevered Portfolio", value=f"${np.median(final_portfolio_values2):,.2f}")

    # Percentile Table
    combined_percentile_table = pd.DataFrame(
        {
            "Levered Portfolio": percentiles,
            "Unlevered Portfolio": percentiles2,
        },
        index=["10th", "25th", "50th (Median)", "75th", "90th"],
    )

    # Format the table to display as currency with 2 decimal places
    formatted_table = combined_percentile_table.style.format("${:,.2f}")

    # Display the formatted table in Streamlit
    st.write("### Percentile Table")
    st.dataframe(formatted_table)

    # Calculate Average Annualized Return (AAR) for both portfolios
    avg_annual_return_levered = ((np.mean(final_equity_values) / initial_investment) ** (1 / time_horizon_years)) - 1
    avg_annual_return_unlevered = ((np.mean(final_portfolio_values2) / initial_investment) ** (
            1 / time_horizon_years)) - 1

    # Display the results
    st.write(f"Average Annualized Return (Levered Portfolio): {avg_annual_return_levered:.2%}")
    st.write(f"Average Annualized Return (Unlevered Portfolio): {avg_annual_return_unlevered:.2%}")

    # Plot Simulated Equity Paths for Leveraged Portfolio
    st.write("### Simulated Equity Paths (Leveraged Portfolio)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(50, num_simulations)):
        ax.plot(range(total_days + 1), equity_values_all[:, i], alpha=0.5, linewidth=0.7)
    ax.axhline(0, color="red", linestyle="--", label="Margin Call Trigger (Equity = $0)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Plot Simulated Equity Paths for Unlevered Portfolio
    st.write("### Simulated Equity Paths (Unlevered Portfolio)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(50, num_simulations)):
        ax.plot(range(total_days + 1), portfolio_values2[:, i], alpha=0.5, linewidth=0.7)
    ax.grid(True)
    st.pyplot(fig)

    # Plot Simulated Interest Rate Paths
    st.write("### Simulated Interest Rate Paths")
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(50, num_simulations)):
        ax.plot(range(total_days + 1), interest_rate_paths[:, i], alpha=0.5, linewidth=0.7)
    ax.grid(True)
    st.pyplot(fig)

    # Ensure final equity values are calculated
    final_equity_values = np.maximum(portfolio_values[-1] - leverage, 0)
    final_portfolio_values2 = portfolio_values2[-1]

    # Calculate VaR as percentage loss
    var_5_levered = (np.percentile(final_equity_values, 5) / initial_investment - 1) * 100
    var_1_levered = (np.percentile(final_equity_values, 1) / initial_investment - 1) * 100
    var_5_unlevered = (np.percentile(final_portfolio_values2, 5) / initial_investment - 1) * 100
    var_1_unlevered = (np.percentile(final_portfolio_values2, 1) / initial_investment - 1) * 100

    # Create a DataFrame for VaR results
    var_table = pd.DataFrame(
    {
        "Levered Portfolio": [f"{var_5_levered:.2f}%", f"{var_1_levered:.2f}%"],
        "Unlevered Portfolio": [f"{var_5_unlevered:.2f}%", f"{var_1_unlevered:.2f}%"],
    },
    index=["VaR at 5%", "VaR at 1%"]
    )

    # Display the VaR table
    st.write("### Value at Risk (VaR) as Percentage Loss")
    st.dataframe(var_table)

    # Calculate probability of achieving 5X the initial investment
    threshold = 5 * initial_investment
    prob_5x_levered = (final_equity_values >= threshold).sum() / num_simulations * 100
    prob_5x_unlevered = (final_portfolio_values2 >= threshold).sum() / num_simulations * 100

    # Display the probabilities
    st.write(f"Probability of Levered Portfolio ≥ 5X Initial Investment: {prob_5x_levered:.2f}%")
    st.write(f"Probability of Unlevered Portfolio ≥ 5X Initial Investment: {prob_5x_unlevered:.2f}%")
