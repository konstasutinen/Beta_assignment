import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set backend to Agg for matplotlib in Streamlit
import matplotlib

matplotlib.use("Agg")

# Title and Team Information
st.title("Beta Project: Leveraged Portfolio Monte-Carlo Simulation (Daily Returns)")
st.write(
    "**Portfolio Management 2024 Beta Assignment.** Made by Konsta Sutinen, Aaro Tuominen, Elias Vanninen, and Kalle Juven")

# Simulation Parameters
st.write("### Simulation Parameters")
initial_investment = st.number_input("Initial Investment ($)", value=10000)

# Updated leverage ratio slider (1 means no leverage)
leverage_ratio = st.slider("Leverage Ratio", min_value=1, max_value=5, value=1)
investment_horizon_years = st.slider("Investment Horizon (Years)", min_value=1, max_value=10, value=10)
investment_horizon_days = int(investment_horizon_years * 252)  # Convert years to trading days

# Interest rate slider
annual_interest_rate = st.slider("Annual Interest Rate (%)", min_value=0.0, max_value=6.0, value=2.0) / 100
daily_interest_rate = annual_interest_rate / 252  # Calculate daily interest rate based on the chosen annual rate

daily_mean_return = 0.000358  # Daily mean return for S&P 500
daily_volatility = 0.010807  # Daily volatility for S&P 500
num_simulations = 10000  # Fixed number of simulations

# Calculate loaned capital and total investment based on leverage ratio
loaned_capital = initial_investment * (leverage_ratio - 1)
total_investment = initial_investment + loaned_capital
margin_threshold = loaned_capital  # Liquidate if portfolio value <= loaned capital

# Display the Donut Chart for Investment Composition
st.write("#### Portfolio Composition")
composition_labels = [f"Equity (${initial_investment:,.2f})", f"Loaned Capital (${loaned_capital:,.2f})"]
composition_values = [initial_investment, loaned_capital]

# Donut chart setup
fig, ax = plt.subplots()
ax.pie(composition_values, labels=composition_labels, startangle=90, wedgeprops=dict(width=0.3),
       colors=["#1f77b4", "#ff7f0e"])
ax.text(0, 0, f"Total\n${total_investment:,.2f}", ha='center', va='center', fontsize=12, weight='bold')
ax.set_title("Portfolio Value Composition")
st.pyplot(fig)

# Run simulation when button is clicked
if st.button("Run Simulation"):
    final_portfolio_values = np.zeros(num_simulations)
    margin_call_count = 0
    portfolio_paths = np.zeros((num_simulations, investment_horizon_days + 1))

    for i in range(num_simulations):
        portfolio_value = total_investment
        equity = initial_investment  # Start with the initial equity

        for day in range(investment_horizon_days):
            # Step 1: Simulate daily return
            daily_return = np.random.normal(daily_mean_return, daily_volatility)

            # Step 2: Update portfolio value based on daily return
            portfolio_value *= (1 + daily_return)

            # Step 3: Deduct daily interest if leverage is used
            if leverage_ratio > 1:
                portfolio_value -= loaned_capital * daily_interest_rate

            # Step 4: Calculate equity (portfolio value - loaned capital)
            equity = portfolio_value - loaned_capital

            # Step 5: Check for margin call (liquidate if portfolio value <= loaned capital)
            if portfolio_value <= loaned_capital:
                equity = 0
                margin_call_count += 1
                break  # Stop further simulation for this path

            # Store portfolio equity for each day
            portfolio_paths[i, day + 1] = equity

        final_portfolio_values[i] = equity

    # Expected Portfolio Value and Margin Call Percentage
    mean_value = np.mean(final_portfolio_values)
    margin_call_percentage = (margin_call_count / num_simulations) * 100

    # Display Explanation with Actual Parameters
    st.write("""
    ### Explanation of the Simulation
    This Monte Carlo simulation models a leveraged portfolio with **daily compounding** based on S&P 500's daily return statistics. The investor maintains an initial leverage ratio of **{leverage_ratio}x** and does **not rebalance** daily. Additionally, the investor pays an annual interest rate of **{annual_interest_rate:.2%}** on the loaned amount, which is deducted daily at a rate of **{daily_interest_rate:.5f}**.

    #### Key Parameters:
    - **Initial Investment**: ${initial_investment:,.2f}
    - **Leverage Ratio**: {leverage_ratio}x
    - **Loaned Capital**: ${loaned_capital:,.2f}
    - **Total Investment**: ${total_investment:,.2f}
    - **Investment Horizon**: {investment_horizon_years} years (compounded daily over 252 trading days per year)
    - **Daily Mean Return**: 0.0358%
    - **Daily Volatility**: 1.0807%
    - **Daily Interest Cost**: {daily_interest_rate:.5f} of the loaned amount, deducted daily (if leverage > 1)
    - **Margin Call**: The portfolio is liquidated if the portfolio value falls below or equals the loaned capital.
    - **Number of Simulations**: {num_simulations}
    """.format(
        leverage_ratio=leverage_ratio,
        initial_investment=initial_investment,
        loaned_capital=loaned_capital,
        total_investment=total_investment,
        investment_horizon_years=investment_horizon_years,
        annual_interest_rate=annual_interest_rate,
        daily_interest_rate=daily_interest_rate,
        num_simulations=num_simulations,
    ))

    # Display Results Summary
    st.write(f"### Simulation Results")
    st.write(f"**Expected Equity Value after {investment_horizon_years} years**: ${mean_value:,.2f}")
    st.write(f"**Percentage of Portfolios that lost all due to Margin Call**: {margin_call_percentage:.2f}%")

    # Percentiles of final equity values
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(final_portfolio_values, percentiles)

    # Display percentile values in a table
    percentile_df = pd.DataFrame({
        "Percentile": [f"{p}%" for p in percentiles],
        "Equity Value": [f"${v:,.2f}" for v in percentile_values]
    })
    st.write("#### Percentile Values of Final Equity")
    st.table(percentile_df)

    # Plot 1: Percentile Line Plot
    percentile_paths = np.percentile(portfolio_paths, percentiles, axis=0)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    days = np.arange(investment_horizon_days + 1) / 252  # Convert days back to years for x-axis
    for i, percentile in enumerate(percentiles):
        ax1.plot(days, percentile_paths[i], label=f"{percentile}th Percentile")
    ax1.set_title("Equity Value Percentiles Over Time")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Equity Value")
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2: Histogram of Final Equity Values
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(final_portfolio_values, bins=100, edgecolor='black')
    ax2.set_xlabel('Final Equity Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Final Equity Values')
    st.pyplot(fig2)

    # Plot 3: Sample Equity Paths (Showing a subset for visual clarity)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    for path in portfolio_paths[:100]:  # Show 100 sample paths
        ax3.plot(days, path, alpha=0.1)
    ax3.set_title("Sample Equity Paths Over Time")
    ax3.set_xlabel("Years")
    ax3.set_ylabel("Equity Value")
    st.pyplot(fig3)
