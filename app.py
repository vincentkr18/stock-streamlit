import streamlit as st
from pages import stocks  # Import your stocks page

# Import other pages as needed
# from pages import dashboard, portfolio, etc.

# Set page config - this should be the first Streamlit command
st.set_page_config(
    page_title="Financial Analysis App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application with navigation"""

    # Create navigation in sidebar
    with st.sidebar:
        st.title("📊 Financial Dashboard")
        page = st.selectbox(
            "Choose a page:",
            ["Home", "Stock Analysis", "Portfolio", "Market Overview", "Settings"],
        )

    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Stock Analysis":
        stocks.main()  # Call the main function from stocks.py
    elif page == "Portfolio":
        show_portfolio_page()
    elif page == "Market Overview":
        show_market_overview_page()
    elif page == "Settings":
        show_settings_page()


def show_home_page():
    """Display the home page"""
    st.title("🏠 Welcome to Financial Analysis Dashboard")
    st.markdown(
        """
    Welcome to your comprehensive financial analysis tool!
    
    ## Available Features:
    
    ### 📈 Stock Analysis
    - Technical indicator analysis (MACD, RSI, Bollinger Bands)
    - Interactive charts with multiple timeframes
    - Multiple stock comparison
    - Real-time data from Yahoo Finance
    
    ### 📊 Portfolio Management
    - Track your investments
    - Performance analytics
    - Risk assessment
    
    ### 🌍 Market Overview
    - Market trends and analysis
    - Sector performance
    - Economic indicators
    
    Use the sidebar to navigate between different sections.
    """
    )


def show_portfolio_page():
    """Placeholder for portfolio page"""
    st.title("📊 Portfolio Management")
    st.info("Portfolio management features coming soon!")


def show_market_overview_page():
    """Placeholder for market overview page"""
    st.title("🌍 Market Overview")
    st.info("Market overview features coming soon!")


def show_settings_page():
    """Settings page"""
    st.title("⚙️ Settings")
    st.markdown("### Application Settings")

    # Theme settings
    theme = st.selectbox("Choose theme:", ["Light", "Dark", "Auto"])

    # Data refresh settings
    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (minutes)", 1, 60, 15)

    # API settings
    st.markdown("### Data Settings")
    data_source = st.selectbox(
        "Data source:", ["Yahoo Finance", "Alpha Vantage", "IEX Cloud"]
    )

    if st.button("Save Settings"):
        st.success("Settings saved successfully!")


if __name__ == "__main__":
    main()
