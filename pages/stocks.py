import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time


class StockAnalyzer:
    def __init__(self):
        self.data = {}

    def fetch_data(self, ticker, period="2y", progress_bar=None):
        """Fetch stock data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if not data.empty:
                self.calculate_indicators(ticker, data)
                return True
            return False
        except Exception as e:
            st.error(f"Error downloading {ticker}: {e}")
            return False

    def calculate_indicators(self, ticker, df):
        """Calculate MACD and RSI indicators"""
        # Calculate MACD
        exp1 = df["Close"].ewm(span=12).mean()
        exp2 = df["Close"].ewm(span=26).mean()
        df["MACD"] = exp1 - exp2
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Calculate Simple Moving Averages
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()

        # Calculate Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        self.data[ticker] = df

    def create_stock_price_chart(self, ticker, start_date, end_date):
        """Create stock price chart with moving averages and Bollinger Bands"""
        if ticker not in self.data:
            return None

        df = self.data[ticker]

        # Handle timezone issues - convert dates to match dataframe timezone
        if df.index.tz is not None:
            if start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            if end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)

        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            return None

        fig = go.Figure()

        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["BB_Upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="lightgray", width=1),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["BB_Lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="lightgray", width=1),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
                showlegend=False,
            )
        )

        # Add Moving Averages
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["SMA_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color="orange", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["SMA_50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="purple", width=2),
            )
        )

        # Add Stock Price
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="blue", width=3),
            )
        )

        fig.update_layout(
            title=f"{ticker} - Stock Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            xaxis=dict(rangeslider=dict(visible=True)),
        )

        return fig

    def create_macd_chart(self, ticker, start_date, end_date):
        """Create MACD chart"""
        if ticker not in self.data:
            return None

        df = self.data[ticker]

        # Handle timezone issues - convert dates to match dataframe timezone
        if df.index.tz is not None:
            if start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            if end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)

        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            return None

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("MACD Line and Signal", "MACD Histogram"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        # MACD Line and Signal with price in hover
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="blue", width=2),
                customdata=filtered_df["Close"],
                hovertemplate="<b>MACD</b><br>Date: %{x}<br>MACD: %{y:.4f}<br>Stock Price: $%{customdata:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["MACD_Signal"],
                mode="lines",
                name="Signal",
                line=dict(color="red", width=2),
                customdata=filtered_df["Close"],
                hovertemplate="<b>MACD Signal</b><br>Date: %{x}<br>Signal: %{y:.4f}<br>Stock Price: $%{customdata:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # MACD Histogram with price in hover
        colors = [
            "green" if val >= 0 else "red" for val in filtered_df["MACD_Histogram"]
        ]
        fig.add_trace(
            go.Bar(
                x=filtered_df.index,
                y=filtered_df["MACD_Histogram"],
                name="Histogram",
                marker_color=colors,
                customdata=filtered_df["Close"],
                hovertemplate="<b>MACD Histogram</b><br>Date: %{x}<br>Histogram: %{y:.4f}<br>Stock Price: $%{customdata:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add zero line
        fig.add_hline(
            y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1
        )

        fig.update_layout(
            title=f"{ticker} - MACD Indicator",
            hovermode="closest",
            template="plotly_white",
            height=500,
        )

        fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

        return fig

    def create_rsi_chart(self, ticker, start_date, end_date):
        """Create RSI chart"""
        if ticker not in self.data:
            return None

        df = self.data[ticker]

        # Handle timezone issues - convert dates to match dataframe timezone
        if df.index.tz is not None:
            if start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            if end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)

        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask]

        if filtered_df.empty:
            return None

        fig = go.Figure()

        # Add RSI line with stock price in hover
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple", width=3),
                customdata=filtered_df["Close"],
                hovertemplate="<b>RSI</b><br>Date: %{x}<br>RSI: %{y:.1f}<br>Stock Price: $%{customdata:.2f}<extra></extra>",
            )
        )

        # Add overbought and oversold lines
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            annotation_text="Overbought (70)",
            annotation_position="right",
        )
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Oversold (30)",
            annotation_position="right",
        )
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)

        # Color fill for overbought/oversold regions
        fig.add_hrect(
            y0=70,
            y1=100,
            fillcolor="red",
            opacity=0.1,
            annotation_text="Overbought",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=0,
            y1=30,
            fillcolor="green",
            opacity=0.1,
            annotation_text="Oversold",
            annotation_position="bottom left",
        )

        fig.update_layout(
            title=f"{ticker} - RSI Indicator",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            hovermode="closest",
            template="plotly_white",
            height=400,
            xaxis=dict(rangeslider=dict(visible=True)),
        )

        return fig

    def generate_trading_signals(self, ticker):
        """Generate trading signals based on technical indicators"""
        if ticker not in self.data:
            return pd.DataFrame()

        df = self.data[ticker].copy()
        signals = []

        for i in range(1, len(df)):
            signal = None
            reason = ""

            # Get current and previous values
            curr_macd = df["MACD"].iloc[i]
            prev_macd = df["MACD"].iloc[i - 1]
            curr_signal = df["MACD_Signal"].iloc[i]
            prev_signal = df["MACD_Signal"].iloc[i - 1]
            curr_rsi = df["RSI"].iloc[i]
            curr_price = df["Close"].iloc[i]
            curr_sma20 = df["SMA_20"].iloc[i]
            curr_sma50 = df["SMA_50"].iloc[i]

            # Buy Signal Rules
            if (
                prev_macd <= prev_signal
                and curr_macd > curr_signal
                and curr_rsi < 35
                and curr_price > curr_sma20
            ):
                signal = "BUY"
                reason = "MACD bullish cross + RSI oversold + Price above SMA20"

            elif (
                prev_macd <= prev_signal
                and curr_macd > curr_signal
                and 30 <= curr_rsi <= 70
                and curr_sma20 > curr_sma50
            ):
                signal = "BUY"
                reason = "MACD bullish cross + RSI neutral + SMA20 > SMA50"

            # Sell Signal Rules
            elif prev_macd >= prev_signal and curr_macd < curr_signal and curr_rsi > 65:
                signal = "SELL"
                reason = "MACD bearish cross + RSI overbought"

            elif curr_rsi > 80 or (curr_rsi > 70 and curr_price < curr_sma20):
                signal = "SELL"
                reason = "RSI extremely overbought or Price below SMA20"

            if signal:
                signals.append(
                    {
                        "Date": df.index[i],
                        "Price": curr_price,
                        "Signal": signal,
                        "Reason": reason,
                        "RSI": curr_rsi,
                        "MACD": curr_macd,
                    }
                )

        return pd.DataFrame(signals)


def initialize_session_state():
    """Initialize session state variables"""
    if "stocks_tickers" not in st.session_state:
        st.session_state.stocks_tickers = [
            "NVDA",
            "META",
            "BRK-B",
            "NFLX",
            "PG",
            "AVGO",
            "ORCL",
            "TSLA",
            "ANET",
            "ORLY",
            "V",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMD",
            "TSM",
            "PLTR",
            "SMC",
        ]
    if "stocks_analyzer" not in st.session_state:
        st.session_state.stocks_analyzer = StockAnalyzer()
    if "stocks_data_loaded" not in st.session_state:
        st.session_state.stocks_data_loaded = set()


def show_stocks_page():
    """Main function to display the stocks page"""
    initialize_session_state()

    st.title("📈 Interactive Stock Analysis Dashboard")
    st.markdown("Analyze multiple stocks with technical indicators and trading signals")

    # Sidebar for ticker management
    with st.sidebar:
        st.header("🎯 Ticker Management")

        # Add new ticker
        new_ticker = st.text_input("Add New Ticker:", placeholder="e.g., TSLA").upper()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ Add", disabled=not new_ticker):
                if new_ticker and new_ticker not in st.session_state.stocks_tickers:
                    st.session_state.stocks_tickers.append(new_ticker)
                    st.success(f"Added {new_ticker}")
                    st.rerun()

        # Current tickers
        st.subheader("Current Tickers:")
        tickers_to_remove = []

        for ticker in st.session_state.stocks_tickers:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(ticker)
            with col2:
                if st.button("🗑️", key=f"remove_{ticker}"):
                    tickers_to_remove.append(ticker)

        # Remove tickers
        for ticker in tickers_to_remove:
            st.session_state.stocks_tickers.remove(ticker)
            if ticker in st.session_state.stocks_data_loaded:
                st.session_state.stocks_data_loaded.remove(ticker)
            if ticker in st.session_state.stocks_analyzer.data:
                del st.session_state.stocks_analyzer.data[ticker]
            st.rerun()

        # Data loading settings
        st.subheader("📊 Data Settings")
        period = st.selectbox(
            "Time Period:", options=["1y", "2y", "3y", "5y", "10y"], index=1
        )

        # Load data button
        if st.button("🔄 Load/Refresh Data"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(st.session_state.stocks_tickers):
                status_text.text(f"Loading {ticker}...")
                if st.session_state.stocks_analyzer.fetch_data(ticker, period):
                    st.session_state.stocks_data_loaded.add(ticker)
                progress_bar.progress((i + 1) / len(st.session_state.stocks_tickers))

            status_text.text("✅ Data loading complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

    # Create tabs for each ticker
    available_tickers = [
        t
        for t in st.session_state.stocks_tickers
        if t in st.session_state.stocks_data_loaded
    ]

    if not available_tickers:
        st.warning("No data available. Please load data first.")
        return

    # Get the date range from all loaded data
    all_dates = []
    for ticker in available_tickers:
        if ticker in st.session_state.stocks_analyzer.data:
            # Convert timezone-aware dates to naive dates for date picker
            dates = st.session_state.stocks_analyzer.data[ticker].index
            if dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)
            all_dates.extend(dates.tolist())

    if not all_dates:
        st.error("No date data available")
        return

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    # Date range selector and time period buttons
    st.subheader("📅 Select Date Range")

    # Time period buttons
    st.write("Quick Select:")
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    # Get current date for calculations
    current_date = max_date

    with col1:
        if st.button("1W"):
            st.session_state.stocks_start_date = current_date - timedelta(weeks=1)
            st.session_state.stocks_end_date = current_date
    with col2:
        if st.button("1M"):
            st.session_state.stocks_start_date = current_date - timedelta(days=30)
            st.session_state.stocks_end_date = current_date
    with col3:
        if st.button("3M"):
            st.session_state.stocks_start_date = current_date - timedelta(days=90)
            st.session_state.stocks_end_date = current_date
    with col4:
        if st.button("6M"):
            st.session_state.stocks_start_date = current_date - timedelta(days=180)
            st.session_state.stocks_end_date = current_date
    with col5:
        if st.button("1Y"):
            st.session_state.stocks_start_date = current_date - timedelta(days=365)
            st.session_state.stocks_end_date = current_date
    with col6:
        if st.button("2Y"):
            st.session_state.stocks_start_date = current_date - timedelta(days=730)
            st.session_state.stocks_end_date = current_date
    with col7:
        if st.button("3Y"):
            st.session_state.stocks_start_date = current_date - timedelta(days=1095)
            st.session_state.stocks_end_date = current_date
    with col8:
        if st.button("4Y"):
            st.session_state.stocks_start_date = current_date - timedelta(days=1460)
            st.session_state.stocks_end_date = current_date
    with col9:
        if st.button("5Y"):
            st.session_state.stocks_start_date = current_date - timedelta(days=1825)
            st.session_state.stocks_end_date = current_date

    col1, col2 = st.columns(2)

    # Initialize session state for dates if not exists
    if "stocks_start_date" not in st.session_state:
        st.session_state.stocks_start_date = min_date
    if "stocks_end_date" not in st.session_state:
        st.session_state.stocks_end_date = max_date

    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=st.session_state.stocks_start_date,
            min_value=min_date,
            max_value=max_date,
            key="stocks_start_date_input",
        )
        st.session_state.stocks_start_date = start_date
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=st.session_state.stocks_end_date,
            min_value=min_date,
            max_value=max_date,
            key="stocks_end_date_input",
        )
        st.session_state.stocks_end_date = end_date

    # Convert to timezone-aware timestamps to match yfinance data
    if available_tickers:
        # Get timezone from first ticker's data
        sample_ticker = available_tickers[0]
        sample_data = st.session_state.stocks_analyzer.data[sample_ticker]
        if sample_data.index.tz is not None:
            start_date = pd.Timestamp(start_date).tz_localize(sample_data.index.tz)
            end_date = pd.Timestamp(end_date).tz_localize(sample_data.index.tz)
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

    # Create tabs for each ticker
    ticker_tabs = st.tabs([f"📊 {ticker}" for ticker in available_tickers])

    for i, ticker in enumerate(available_tickers):
        with ticker_tabs[i]:
            if ticker not in st.session_state.stocks_analyzer.data:
                st.warning(f"No data available for {ticker}")
                continue

            # Get current price info
            current_data = st.session_state.stocks_analyzer.data[ticker].iloc[-1]
            prev_data = st.session_state.stocks_analyzer.data[ticker].iloc[-2]
            price_change = current_data["Close"] - prev_data["Close"]
            price_change_pct = (price_change / prev_data["Close"]) * 100

            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "Current Price",
                    f"${current_data['Close']:.2f}",
                    f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                )
            with col2:
                st.metric("Volume", f"{current_data['Volume']:,}")
            with col3:
                st.metric("RSI", f"{current_data['RSI']:.1f}")
            with col4:
                st.metric("MACD", f"{current_data['MACD']:.3f}")
            with col5:
                st.metric(
                    "52W Range",
                    f"${st.session_state.stocks_analyzer.data[ticker]['Low'].min():.2f} - ${st.session_state.stocks_analyzer.data[ticker]['High'].max():.2f}",
                )

            # Create charts stacked vertically - no tabs, just one below another
            st.subheader("📈 Stock Price & Technical Indicators")

            # Chart 1: Price & Indicators
            fig1 = st.session_state.stocks_analyzer.create_stock_price_chart(
                ticker, start_date, end_date
            )
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)

            # Chart 2: MACD
            fig2 = st.session_state.stocks_analyzer.create_macd_chart(
                ticker, start_date, end_date
            )
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)

            # Chart 3: RSI
            fig3 = st.session_state.stocks_analyzer.create_rsi_chart(
                ticker, start_date, end_date
            )
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)


# This is the main function that will be called from main.py
def main():
    """Entry point for the stocks page"""
    show_stocks_page()


# Allow the page to run standalone for testing
if __name__ == "__main__":
    # Set page config only when running standalone
    st.set_page_config(
        page_title="Stock Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
