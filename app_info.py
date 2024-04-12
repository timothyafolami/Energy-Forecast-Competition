import streamlit as st
# Page configuration
st.set_page_config(
    page_title="Price Forecast Analysis",
    page_icon="📈",
    layout="wide",
)

# Sidebar
st.sidebar.success("Select a page above.")

st.markdown(
    """
## US Electricity Prices and Sales by State, Sector, and Year

This comprehensive dataset offers a detailed look at the United States electricity market, providing valuable insights into prices, sales, and revenue across various states, sectors, and years. With data spanning from 2001 onwards to 2024, this dataset is a powerful tool for analyzing the complex dynamics of the US electricity market and understanding how it has evolved over time.

The dataset includes eight key variables:

        year: The year of the observation
        month: The month of the observation
        stateDescription: The name of the state
        sectorName: The sector of the electricity market (residential, commercial, industrial, other, or all sectors)
        customers: The number of customers (missing for some observations)
        price: The average price of electricity per kilowatt-hour (kWh) in cents
        revenue: The total revenue generated from electricity sales in millions of dollars
        sales: The total electricity sales in millions of kilowatt-hours (kWh)
        By providing such granular data, this dataset enables users to conduct in-depth analyses of electricity market trends, comparing prices and consumption patterns across different states and sectors, and examining the impact of seasonality on demand and prices.

One of the primary applications of this dataset is in forecasting future electricity prices and sales based on historical trends. By leveraging the extensive time series data available, researchers and analysts can develop sophisticated models to predict how prices and demand may change in the coming years, taking into account factors such as economic growth, population shifts, and policy changes. This predictive power is invaluable for policymakers, energy companies, and investors looking to make informed decisions in the rapidly evolving electricity market.

Another key use case for this dataset is in investigating the complex relationships between electricity prices, sales volumes, and revenue. By combining the price, sales, and revenue data, users can explore how changes in prices impact consumer behavior and utility company bottom lines. This analysis can shed light on important questions such as the price elasticity of electricity demand, the effectiveness of energy efficiency programs, and the potential impact of new technologies like renewable energy and energy storage on the market.

Beyond its immediate applications in the energy sector, this dataset also has broader implications for understanding the US economy and society as a whole. Electricity is a critical input for businesses and households across the country, and changes in electricity prices and consumption can have far-reaching effects on economic growth, competitiveness, and quality of life. By providing such a rich and detailed portrait of the US electricity market, this dataset opens up new avenues for research and insights that can inform public policy, business strategy, and academic inquiry.
"""
)


