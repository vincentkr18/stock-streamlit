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

    def generate_trading_signals(self, ticker, start_date=None, end_date=None):
        """Generate trading signals based on technical indicators"""
        if ticker not in self.data:
            return pd.DataFrame()

        df = self.data[ticker].copy()

        # Filter by date range if provided
        if start_date and end_date:
            if df.index.tz is not None:
                if start_date.tz is None:
                    start_date = start_date.tz_localize(df.index.tz)
                if end_date.tz is None:
                    end_date = end_date.tz_localize(df.index.tz)
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]

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
                        "Date": (
                            df.index[i].strftime("%Y-%m-%d")
                            if hasattr(df.index[i], "strftime")
                            else str(df.index[i])
                        ),
                        "Price": round(curr_price, 2),
                        "Signal": signal,
                        "Reason": reason,
                        "RSI": round(curr_rsi, 1),
                        "MACD": round(curr_macd, 4),
                    }
                )

        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            # Sort by date (most recent first)
            signals_df = signals_df.sort_values("Date", ascending=False)

        return signals_df

    def create_performance_chart(self, ticker, start_date, end_date):
        """Create monthly/yearly performance bar chart"""
        if ticker not in self.data:
            return None

        df = self.data[ticker].copy()

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

        # Calculate the time span to determine if we show monthly or yearly data
        time_span = (end_date - start_date).days

        if time_span <= 730:  # Less than 2 years - show monthly performance
            # Resample to monthly data
            monthly_data = (
                filtered_df["Close"]
                .resample("M")
                .agg({"start": "first", "end": "last"})
            )
            monthly_data = monthly_data.dropna()

            # Calculate monthly percentage change
            monthly_performance = (
                (monthly_data["end"] - monthly_data["start"])
                / monthly_data["start"]
                * 100
            )

            periods = [f"{date.strftime('%Y-%m')}" for date in monthly_data.index]
            values = monthly_performance.values
            title_suffix = "Monthly Performance"

        else:  # More than 2 years - show yearly performance
            # Resample to yearly data
            yearly_data = (
                filtered_df["Close"]
                .resample("Y")
                .agg({"start": "first", "end": "last"})
            )
            yearly_data = yearly_data.dropna()

            # Calculate yearly percentage change
            yearly_performance = (
                (yearly_data["end"] - yearly_data["start"]) / yearly_data["start"] * 100
            )

            periods = [f"{date.year}" for date in yearly_data.index]
            values = yearly_performance.values
            title_suffix = "Yearly Performance"

        if len(values) == 0:
            return None

        # Create bar chart
        colors = ["green" if val >= 0 else "red" for val in values]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=periods,
                y=values,
                marker_color=colors,
                text=[f"{val:.1f}%" for val in values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Performance: %{y:.2f}%<extra></extra>",
            )
        )

        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        fig.update_layout(
            title=f"{ticker} - {title_suffix}",
            xaxis_title="Period",
            yaxis_title="Performance (%)",
            template="plotly_white",
            height=400,
            showlegend=False,
        )

        return fig

    def create_signal_performance_chart(self, ticker, start_date, end_date):
        """Create bar chart showing performance based on buy/sell signals"""
        if ticker not in self.data:
            return None

        # Get signals for the time period
        signals_df = self.generate_trading_signals(ticker, start_date, end_date)

        if signals_df.empty:
            return None

        # Convert date strings back to datetime for processing
        signals_df["Date"] = pd.to_datetime(signals_df["Date"])
        signals_df = signals_df.sort_values("Date")

        # Calculate performance for each signal pair (BUY followed by SELL)
        performance_data = []
        buy_signals = signals_df[signals_df["Signal"] == "BUY"].copy()
        sell_signals = signals_df[signals_df["Signal"] == "SELL"].copy()

        for _, buy_signal in buy_signals.iterrows():
            # Find the next sell signal after this buy signal
            next_sell = sell_signals[sell_signals["Date"] > buy_signal["Date"]]

            if not next_sell.empty:
                sell_signal = next_sell.iloc[0]

                # Calculate percentage change
                buy_price = buy_signal["Price"]
                sell_price = sell_signal["Price"]
                performance = ((sell_price - buy_price) / buy_price) * 100

                performance_data.append(
                    {
                        "Signal_Pair": f"{buy_signal['Date'].strftime('%Y-%m-%d')} → {sell_signal['Date'].strftime('%Y-%m-%d')}",
                        "Buy_Date": buy_signal["Date"],
                        "Sell_Date": sell_signal["Date"],
                        "Buy_Price": buy_price,
                        "Sell_Price": sell_price,
                        "Performance": performance,
                    }
                )

        if not performance_data:
            return None

        perf_df = pd.DataFrame(performance_data)

        # Create bar chart
        colors = ["green" if val >= 0 else "red" for val in perf_df["Performance"]]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=list(range(len(perf_df))),
                y=perf_df["Performance"],
                marker_color=colors,
                text=[f"{val:.1f}%" for val in perf_df["Performance"]],
                textposition="outside",
                customdata=perf_df[["Signal_Pair", "Buy_Price", "Sell_Price"]],
                hovertemplate="<b>Trade %{x}</b><br>"
                + "Period: %{customdata[0]}<br>"
                + "Buy Price: $%{customdata[1]:.2f}<br>"
                + "Sell Price: $%{customdata[2]:.2f}<br>"
                + "Performance: %{y:.2f}%<extra></extra>",
            )
        )

        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

        # Calculate summary statistics
        total_trades = len(perf_df)
        winning_trades = len(perf_df[perf_df["Performance"] > 0])
        losing_trades = len(perf_df[perf_df["Performance"] < 0])
        avg_performance = perf_df["Performance"].mean()

        fig.update_layout(
            title=f"{ticker} - Signal-Based Trading Performance<br>"
            + f"<span style='font-size:12px'>Total: {total_trades} trades | "
            + f"Wins: {winning_trades} | Losses: {losing_trades} | "
            + f"Avg: {avg_performance:.1f}%</span>",
            xaxis_title="Trade Number",
            yaxis_title="Performance (%)",
            template="plotly_white",
            height=400,
            showlegend=False,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )

        return fig


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

            # NEW: Trading Signals Table
            st.subheader("📋 Trading Signals")
            signals_df = st.session_state.stocks_analyzer.generate_trading_signals(
                ticker, start_date, end_date
            )

            if not signals_df.empty:
                # Display summary metrics
                buy_signals = len(signals_df[signals_df["Signal"] == "BUY"])
                sell_signals = len(signals_df[signals_df["Signal"] == "SELL"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Signals", len(signals_df))
                with col2:
                    st.metric("Buy Signals", buy_signals)
                with col3:
                    st.metric("Sell Signals", sell_signals)

                # Display the signals dataframe with color coding
                def color_signal(val):
                    if val == "BUY":
                        return "background-color: #90EE90; color: #006400; font-weight: bold"  # Light green background, dark green text
                    elif val == "SELL":
                        return "background-color: #FFB6C1; color: #8B0000; font-weight: bold"  # Light red background, dark red text
                    return ""

                # Apply styling to the dataframe
                styled_df = signals_df.style.applymap(color_signal, subset=["Signal"])

                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    column_config={
                        "Date": st.column_config.DateColumn("Date"),
                        "Price": st.column_config.NumberColumn(
                            "Price ($)", format="%.2f"
                        ),
                        "Signal": st.column_config.TextColumn("Signal"),
                        "Reason": st.column_config.TextColumn("Reason"),
                        "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                        "MACD": st.column_config.NumberColumn("MACD", format="%.4f"),
                    },
                )

                # Bar chart from signals dataframe
                fig_signals = go.Figure()
                colors = [
                    "green" if signal == "BUY" else "red"
                    for signal in signals_df["Signal"]
                ]

                fig_signals.add_trace(
                    go.Bar(
                        x=signals_df["Date"],
                        y=signals_df["Price"],
                        marker_color=colors,
                        text=signals_df["Signal"],
                        textposition="outside",
                        hovertemplate="<b>%{text}</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                    )
                )

                fig_signals.update_layout(
                    title=f"{ticker} - Buy/Sell Signals by Price",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_white",
                    height=400,
                    showlegend=False,
                )

                st.plotly_chart(fig_signals, use_container_width=True)

                # Trading Performance Table
                st.subheader("💰 Trading Performance Analysis")

                # Create buy-sell pairs
                buy_signals = signals_df[signals_df["Signal"] == "BUY"].copy()
                sell_signals = signals_df[signals_df["Signal"] == "SELL"].copy()

                if not buy_signals.empty and not sell_signals.empty:
                    # Convert dates to datetime for comparison
                    buy_signals["Date"] = pd.to_datetime(buy_signals["Date"])
                    sell_signals["Date"] = pd.to_datetime(sell_signals["Date"])

                    # Sort by date
                    buy_signals = buy_signals.sort_values("Date")
                    sell_signals = sell_signals.sort_values("Date")

                    trading_pairs = []

                    for _, buy_signal in buy_signals.iterrows():
                        # Find the first sell signal after this buy signal
                        next_sell = sell_signals[
                            sell_signals["Date"] > buy_signal["Date"]
                        ]

                        if not next_sell.empty:
                            sell_signal = next_sell.iloc[0]

                            # Calculate percentage change
                            buy_price = buy_signal["Price"]
                            sell_price = sell_signal["Price"]
                            pct_change = ((sell_price - buy_price) / buy_price) * 100

                            trading_pairs.append(
                                {
                                    "Buy Date": buy_signal["Date"].strftime("%Y-%m-%d"),
                                    "Buy Price": buy_price,
                                    "Sell Date": sell_signal["Date"].strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "Sell Price": sell_price,
                                    "% Change": pct_change,
                                }
                            )

                    if trading_pairs:
                        trading_df = pd.DataFrame(trading_pairs)

                        # Display summary metrics
                        total_trades = len(trading_df)
                        profitable_trades = len(trading_df[trading_df["% Change"] > 0])
                        avg_return = trading_df["% Change"].mean()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Trades", total_trades)
                        with col2:
                            st.metric(
                                "Profitable Trades",
                                f"{profitable_trades}/{total_trades}",
                            )
                        with col3:
                            st.metric("Average Return", f"{avg_return:.2f}%")

                        # Color code the percentage column
                        def color_pct_change(val):
                            if val > 0:
                                return "background-color: #90EE90; color: #006400; font-weight: bold"
                            elif val < 0:
                                return "background-color: #FFB6C1; color: #8B0000; font-weight: bold"
                            return ""

                        styled_trading_df = trading_df.style.applymap(
                            color_pct_change, subset=["% Change"]
                        )

                        st.dataframe(
                            styled_trading_df,
                            use_container_width=True,
                            column_config={
                                "Buy Date": st.column_config.DateColumn("Buy Date"),
                                "Buy Price": st.column_config.NumberColumn(
                                    "Buy Price ($)", format="%.2f"
                                ),
                                "Sell Date": st.column_config.DateColumn("Sell Date"),
                                "Sell Price": st.column_config.NumberColumn(
                                    "Sell Price ($)", format="%.2f"
                                ),
                                "% Change": st.column_config.NumberColumn(
                                    "% Change", format="%.2f%%"
                                ),
                            },
                        )
                    else:
                        st.info("No complete buy-sell pairs found.")
                else:
                    st.info("Insufficient buy or sell signals for trading analysis.")

                # Yearly Trading Performance Analysis

                st.subheader("📅 Yearly Trading Performance (Last 6 Years)")

                # Get the last 6 years of data
                current_year = datetime.now().year
                yearly_performance = []

                for year in range(current_year - 5, current_year + 1):
                    year_start = pd.Timestamp(f"{year}-01-01")
                    year_end = pd.Timestamp(f"{year}-12-31")

                    # Get signals for this year
                    year_signals_df = (
                        st.session_state.stocks_analyzer.generate_trading_signals(
                            ticker, year_start, year_end
                        )
                    )

                    if not year_signals_df.empty:
                        # Convert dates to datetime
                        year_signals_df["Date"] = pd.to_datetime(
                            year_signals_df["Date"]
                        )

                        # Get buy and sell signals for this year
                        year_buys = year_signals_df[
                            year_signals_df["Signal"] == "BUY"
                        ].copy()
                        year_sells = year_signals_df[
                            year_signals_df["Signal"] == "SELL"
                        ].copy()

                        if not year_buys.empty:
                            year_buys = year_buys.sort_values("Date")
                            year_sells = year_sells.sort_values("Date")

                            year_trades = []

                            for _, buy_signal in year_buys.iterrows():
                                # Find the first sell signal after this buy signal within the year
                                next_sell = year_sells[
                                    year_sells["Date"] > buy_signal["Date"]
                                ]

                                if not next_sell.empty:
                                    sell_signal = next_sell.iloc[0]
                                    buy_price = buy_signal["Price"]
                                    sell_price = sell_signal["Price"]
                                    pct_change = (
                                        (sell_price - buy_price) / buy_price
                                    ) * 100

                                    year_trades.append(
                                        {
                                            "Buy Date": buy_signal["Date"].strftime(
                                                "%Y-%m-%d"
                                            ),
                                            "Buy Price": buy_price,
                                            "Sell Date": sell_signal["Date"].strftime(
                                                "%Y-%m-%d"
                                            ),
                                            "Sell Price": sell_price,
                                            "% Change": pct_change,
                                        }
                                    )
                                else:
                                    # Buy signal without corresponding sell (still open position)
                                    year_trades.append(
                                        {
                                            "Buy Date": buy_signal["Date"].strftime(
                                                "%Y-%m-%d"
                                            ),
                                            "Buy Price": buy_signal["Price"],
                                            "Sell Date": "",
                                            "Sell Price": "",
                                            "% Change": "",
                                        }
                                    )

                            if year_trades:
                                yearly_performance.append(
                                    {"Year": year, "Trades": year_trades}
                                )

                            # Display yearly performance in tabs
                            if yearly_performance:
                                year_tabs = st.tabs(
                                    [f"{data['Year']}" for data in yearly_performance]
                                )

                                for i, year_data in enumerate(yearly_performance):
                                    with year_tabs[i]:
                                        trades_df = pd.DataFrame(year_data["Trades"])

                                        # Calculate metrics (excluding empty rows)
                                        completed_trades = trades_df[
                                            trades_df["% Change"] != ""
                                        ]
                                        total_trades = len(trades_df)
                                        completed_count = len(completed_trades)
                                        open_positions = total_trades - completed_count

                                        if completed_count > 0:
                                            profitable_trades = len(
                                                completed_trades[
                                                    completed_trades["% Change"] > 0
                                                ]
                                            )
                                            avg_return = completed_trades[
                                                "% Change"
                                            ].mean()
                                        else:
                                            profitable_trades = 0
                                            avg_return = 0

                                        # Display metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Total Signals", total_trades)
                                        with col2:
                                            st.metric(
                                                "Completed Trades", completed_count
                                            )
                                        with col3:
                                            st.metric("Open Positions", open_positions)
                                        with col4:
                                            if completed_count > 0:
                                                st.metric(
                                                    "Avg Return", f"{avg_return:.2f}%"
                                                )
                                            else:
                                                st.metric("Avg Return", "N/A")

                                        # Color coding function for the dataframe
                                        def color_yearly_pct_change(val):
                                            if val == "":
                                                return ""
                                            if float(val) > 0:
                                                return "background-color: #90EE90; color: #006400; font-weight: bold"
                                            elif float(val) < 0:
                                                return "background-color: #FFB6C1; color: #8B0000; font-weight: bold"
                                            return ""

                                        # Apply styling
                                        if completed_count > 0:
                                            styled_yearly_df = trades_df.style.applymap(
                                                color_yearly_pct_change,
                                                subset=["% Change"],
                                            )
                                        else:
                                            styled_yearly_df = trades_df.style

                                        st.dataframe(
                                            styled_yearly_df,
                                            use_container_width=True,
                                            column_config={
                                                "Buy Date": st.column_config.DateColumn(
                                                    "Buy Date"
                                                ),
                                                "Buy Price": st.column_config.NumberColumn(
                                                    "Buy Price ($)", format="%.2f"
                                                ),
                                                "Sell Date": st.column_config.TextColumn(
                                                    "Sell Date"
                                                ),
                                                "Sell Price": st.column_config.TextColumn(
                                                    "Sell Price ($)"
                                                ),
                                                "% Change": st.column_config.TextColumn(
                                                    "% Change"
                                                ),
                                            },
                                        )
            else:
                st.info("No trading signals found in the last 6 years.")


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
