import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests


class CryptoAnalyzer:
    def __init__(self):
        self.data = {}
        self.crypto_symbols = {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD",
            "XRP": "XRP-USD",
            "Solana": "SOL-USD",
            "Dogecoin": "DOGE-USD",
            "Shiba Inu": "SHIB-USD",
            "Cardano": "ADA-USD",
            "Polygon": "MATIC-USD",
            "Chainlink": "LINK-USD",
            "Litecoin": "LTC-USD",
            "Avalanche": "AVAX-USD",
            "Polkadot": "DOT-USD",
        }

    def fetch_data(self, crypto_name, period="1y", progress_bar=None):
        """Fetch cryptocurrency data for a given crypto"""
        try:
            symbol = self.crypto_symbols.get(crypto_name)
            if not symbol:
                st.error(f"Unknown cryptocurrency: {crypto_name}")
                return False

            crypto = yf.Ticker(symbol)
            data = crypto.history(period=period)
            if not data.empty:
                self.calculate_indicators(crypto_name, data)
                return True
            return False
        except Exception as e:
            st.error(f"Error downloading {crypto_name}: {e}")
            return False

    def calculate_indicators(self, crypto_name, df):
        """Calculate MACD, RSI, and other technical indicators"""
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
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Calculate Exponential Moving Averages
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()

        # Calculate Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        # Calculate volatility (20-day rolling standard deviation)
        df["Volatility"] = (
            df["Close"].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        )

        # Calculate support and resistance levels
        df["Support"] = df["Low"].rolling(window=20).min()
        df["Resistance"] = df["High"].rolling(window=20).max()

        self.data[crypto_name] = df

    def create_crypto_price_chart(self, crypto_name, start_date, end_date):
        """Create cryptocurrency price chart with moving averages and Bollinger Bands"""
        if crypto_name not in self.data:
            return None

        df = self.data[crypto_name]

        # Handle timezone issues
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

        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["SMA_200"],
                mode="lines",
                name="SMA 200",
                line=dict(color="red", width=2),
            )
        )

        # Add Crypto Price
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="blue", width=3),
            )
        )

        # Add volume bars as secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["Volume"],
                mode="lines",
                name="Volume",
                line=dict(color="gray", width=1),
                opacity=0.3,
                yaxis="y2",
            )
        )

        fig.update_layout(
            title=f"{crypto_name} - Price Analysis with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            hovermode="x unified",
            template="plotly_dark",
            height=600,
            xaxis=dict(rangeslider=dict(visible=True)),
            plot_bgcolor="rgba(0,0,0,0.1)",
        )

        return fig

    def create_macd_chart(self, crypto_name, start_date, end_date):
        """Create MACD chart for cryptocurrency"""
        if crypto_name not in self.data:
            return None

        df = self.data[crypto_name]

        # Handle timezone issues
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

        # MACD Line and Signal
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="cyan", width=2),
                customdata=filtered_df["Close"],
                hovertemplate="<b>MACD</b><br>Date: %{x}<br>MACD: %{y:.6f}<br>Price: $%{customdata:.4f}<extra></extra>",
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
                line=dict(color="yellow", width=2),
                customdata=filtered_df["Close"],
                hovertemplate="<b>MACD Signal</b><br>Date: %{x}<br>Signal: %{y:.6f}<br>Price: $%{customdata:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # MACD Histogram
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
                hovertemplate="<b>MACD Histogram</b><br>Date: %{x}<br>Histogram: %{y:.6f}<br>Price: $%{customdata:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Add zero lines
        fig.add_hline(
            y=0, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1
        )
        fig.add_hline(
            y=0, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1
        )

        fig.update_layout(
            title=f"{crypto_name} - MACD Indicator",
            hovermode="closest",
            template="plotly_dark",
            height=500,
            plot_bgcolor="rgba(0,0,0,0.1)",
        )

        fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

        return fig

    def create_rsi_chart(self, crypto_name, start_date, end_date):
        """Create RSI chart for cryptocurrency"""
        if crypto_name not in self.data:
            return None

        df = self.data[crypto_name]

        # Handle timezone issues
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

        # Add RSI line
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="purple", width=3),
                customdata=filtered_df["Close"],
                hovertemplate="<b>RSI</b><br>Date: %{x}<br>RSI: %{y:.1f}<br>Price: $%{customdata:.4f}<extra></extra>",
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
            opacity=0.2,
            annotation_text="Overbought Zone",
            annotation_position="top left",
        )
        fig.add_hrect(
            y0=0,
            y1=30,
            fillcolor="green",
            opacity=0.2,
            annotation_text="Oversold Zone",
            annotation_position="bottom left",
        )

        fig.update_layout(
            title=f"{crypto_name} - RSI Indicator",
            xaxis_title="Date",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            hovermode="closest",
            template="plotly_dark",
            height=400,
            xaxis=dict(rangeslider=dict(visible=True)),
            plot_bgcolor="rgba(0,0,0,0.1)",
        )

        return fig

    def create_volatility_chart(self, crypto_name, start_date, end_date):
        """Create volatility chart for cryptocurrency"""
        if crypto_name not in self.data:
            return None

        df = self.data[crypto_name]

        # Handle timezone issues
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

        # Add volatility line
        fig.add_trace(
            go.Scatter(
                x=filtered_df.index,
                y=filtered_df["Volatility"],
                mode="lines",
                name="Volatility",
                line=dict(color="orange", width=2),
                fill="tonexty",
                fillcolor="rgba(255,165,0,0.1)",
            )
        )

        # Add average volatility line
        avg_volatility = filtered_df["Volatility"].mean()
        fig.add_hline(
            y=avg_volatility,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {avg_volatility:.1f}%",
            annotation_position="right",
        )

        fig.update_layout(
            title=f"{crypto_name} - Price Volatility (20-day)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode="closest",
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(0,0,0,0.1)",
        )

        return fig

    def generate_crypto_signals(self, crypto_name):
        """Generate trading signals based on technical indicators for crypto"""
        if crypto_name not in self.data:
            return pd.DataFrame()

        df = self.data[crypto_name].copy()
        signals = []

        for i in range(1, len(df)):
            signal = None
            reason = ""
            strength = "Medium"

            # Get current and previous values
            curr_macd = df["MACD"].iloc[i]
            prev_macd = df["MACD"].iloc[i - 1]
            curr_signal = df["MACD_Signal"].iloc[i]
            prev_signal = df["MACD_Signal"].iloc[i - 1]
            curr_rsi = df["RSI"].iloc[i]
            curr_price = df["Close"].iloc[i]
            curr_sma20 = df["SMA_20"].iloc[i]
            curr_sma50 = df["SMA_50"].iloc[i]
            curr_sma200 = df["SMA_200"].iloc[i]
            curr_vol = (
                df["Volatility"].iloc[i] if not pd.isna(df["Volatility"].iloc[i]) else 0
            )

            # Strong Buy Signals
            if (
                prev_macd <= prev_signal
                and curr_macd > curr_signal
                and curr_rsi < 30
                and curr_price > curr_sma20
                and curr_sma20 > curr_sma50
            ):
                signal = "STRONG BUY"
                reason = "MACD bullish cross + RSI oversold + Strong uptrend"
                strength = "Strong"

            # Buy Signals
            elif (
                prev_macd <= prev_signal
                and curr_macd > curr_signal
                and 30 <= curr_rsi <= 50
                and curr_price > curr_sma50
            ):
                signal = "BUY"
                reason = "MACD bullish cross + Price above SMA50"

            elif curr_rsi < 25 and curr_price > curr_sma200:
                signal = "BUY"
                reason = "Extremely oversold + Long-term uptrend"

            # Strong Sell Signals
            elif (
                prev_macd >= prev_signal
                and curr_macd < curr_signal
                and curr_rsi > 70
                and curr_price < curr_sma20
            ):
                signal = "STRONG SELL"
                reason = "MACD bearish cross + RSI overbought + Price below SMA20"
                strength = "Strong"

            # Sell Signals
            elif curr_rsi > 80:
                signal = "SELL"
                reason = "RSI extremely overbought"

            elif curr_price < curr_sma50 and curr_sma50 < curr_sma200 and curr_rsi > 60:
                signal = "SELL"
                reason = "Bearish trend + RSI overbought"

            # Hold/Neutral signals for high volatility
            elif curr_vol > 100:  # High volatility threshold
                if not signal:
                    signal = "HOLD"
                    reason = f"High volatility ({curr_vol:.1f}%) - Wait for stability"

            if signal:
                signals.append(
                    {
                        "Date": df.index[i],
                        "Price": curr_price,
                        "Signal": signal,
                        "Strength": strength,
                        "Reason": reason,
                        "RSI": curr_rsi,
                        "MACD": curr_macd,
                        "Volatility": curr_vol,
                    }
                )

        return pd.DataFrame(signals)

    def get_crypto_info(self, crypto_name):
        """Get additional crypto information"""
        if crypto_name not in self.data:
            return {}

        df = self.data[crypto_name]
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2]

        # Calculate 24h change
        price_change_24h = current_price - prev_price
        price_change_pct_24h = (price_change_24h / prev_price) * 100

        # Calculate 7-day change
        if len(df) >= 7:
            price_7d_ago = df["Close"].iloc[-7]
            price_change_7d = current_price - price_7d_ago
            price_change_pct_7d = (price_change_7d / price_7d_ago) * 100
        else:
            price_change_7d = price_change_pct_7d = 0

        # Calculate all-time high and low in the data
        ath = df["High"].max()
        atl = df["Low"].min()

        return {
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "price_change_pct_24h": price_change_pct_24h,
            "price_change_7d": price_change_7d,
            "price_change_pct_7d": price_change_pct_7d,
            "ath": ath,
            "atl": atl,
            "volume_24h": df["Volume"].iloc[-1],
            "market_cap_estimate": current_price
            * df["Volume"].iloc[-1]
            / 1000000,  # Rough estimate
        }


def initialize_session_state():
    """Initialize session state variables for crypto"""
    if "crypto_coins" not in st.session_state:
        st.session_state.crypto_coins = [
            "Bitcoin",
            "Ethereum",
            "XRP",
            "Solana",
            "Dogecoin",
            "Shiba Inu",
            "Cardano",
            "Polygon",
        ]
    if "crypto_analyzer" not in st.session_state:
        st.session_state.crypto_analyzer = CryptoAnalyzer()
    if "crypto_data_loaded" not in st.session_state:
        st.session_state.crypto_data_loaded = set()


def show_crypto_page():
    """Main function to display the crypto page"""
    initialize_session_state()

    st.title("₿ Cryptocurrency Analysis Dashboard")
    st.markdown("🚀 **Advanced Technical Analysis for Major Cryptocurrencies**")

    # Sidebar for crypto management
    with st.sidebar:
        st.header("💰 Cryptocurrency Selection")

        # Add new crypto
        available_cryptos = list(st.session_state.crypto_analyzer.crypto_symbols.keys())
        new_crypto = st.selectbox(
            "Add New Cryptocurrency:",
            [""]
            + [c for c in available_cryptos if c not in st.session_state.crypto_coins],
            key="new_crypto_select",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add", disabled=not new_crypto):
                if new_crypto and new_crypto not in st.session_state.crypto_coins:
                    st.session_state.crypto_coins.append(new_crypto)
                    st.success(f"Added {new_crypto}")
                    st.rerun()

        # Current cryptocurrencies
        st.subheader("📋 Selected Cryptocurrencies:")
        cryptos_to_remove = []

        for crypto in st.session_state.crypto_coins:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Add emoji based on crypto name
                emoji_map = {
                    "Bitcoin": "₿",
                    "Ethereum": "Ξ",
                    "XRP": "🪙",
                    "Solana": "◎",
                    "Dogecoin": "🐕",
                    "Shiba Inu": "🐕",
                    "Cardano": "🔷",
                    "Polygon": "🔺",
                }
                emoji = emoji_map.get(crypto, "🪙")
                st.text(f"{emoji} {crypto}")
            with col2:
                if st.button("🗑️", key=f"remove_{crypto}"):
                    cryptos_to_remove.append(crypto)

        # Remove cryptos
        for crypto in cryptos_to_remove:
            st.session_state.crypto_coins.remove(crypto)
            if crypto in st.session_state.crypto_data_loaded:
                st.session_state.crypto_data_loaded.remove(crypto)
            if crypto in st.session_state.crypto_analyzer.data:
                del st.session_state.crypto_analyzer.data[crypto]
            st.rerun()

        # Data loading settings
        st.subheader("📊 Data Settings")
        period = st.selectbox(
            "Time Period:", options=["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3
        )

        # Load data button
        if st.button("🔄 Load/Refresh Data"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, crypto in enumerate(st.session_state.crypto_coins):
                status_text.text(f"Loading {crypto}...")
                if st.session_state.crypto_analyzer.fetch_data(crypto, period):
                    st.session_state.crypto_data_loaded.add(crypto)
                progress_bar.progress((i + 1) / len(st.session_state.crypto_coins))

            status_text.text("✅ Data loading complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

        # Market overview section
        st.subheader("📈 Market Overview")
        if st.session_state.crypto_data_loaded:
            total_portfolio_value = 0
            for crypto in st.session_state.crypto_data_loaded:
                info = st.session_state.crypto_analyzer.get_crypto_info(crypto)
                if info:
                    total_portfolio_value += info.get("current_price", 0)

            st.metric("Portfolio Value", f"${total_portfolio_value:,.2f}")

    # Create main content area
    available_cryptos = [
        c
        for c in st.session_state.crypto_coins
        if c in st.session_state.crypto_data_loaded
    ]

    if not available_cryptos:
        st.warning(
            "⚠️ No cryptocurrency data available. Please load data first using the sidebar."
        )
        return

    # Get date range from all loaded data
    all_dates = []
    for crypto in available_cryptos:
        if crypto in st.session_state.crypto_analyzer.data:
            dates = st.session_state.crypto_analyzer.data[crypto].index
            if dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)
            all_dates.extend(dates.tolist())

    if not all_dates:
        st.error("❌ No date data available")
        return

    min_date = min(all_dates).date()
    max_date = max(all_dates).date()

    # Date range selector with quick buttons
    st.subheader("📅 Select Analysis Period")

    # Quick select buttons
    st.write("⚡ Quick Select:")
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    current_date = max_date

    quick_periods = [
        ("1D", 1),
        ("1W", 7),
        ("1M", 30),
        ("3M", 90),
        ("6M", 180),
        ("1Y", 365),
        ("2Y", 730),
        ("3Y", 1095),
        ("All", None),
    ]

    for i, (label, days) in enumerate(quick_periods):
        with [col1, col2, col3, col4, col5, col6, col7, col8, col9][i]:
            if st.button(label):
                if days:
                    # Ensure the calculated start date doesn't go below min_date
                    calculated_start = current_date - timedelta(days=days)
                    st.session_state.crypto_start_date = max(calculated_start, min_date)
                else:
                    st.session_state.crypto_start_date = min_date
                st.session_state.crypto_end_date = current_date

    # Date inputs
    col1, col2 = st.columns(2)

    # Initialize session state for dates
    if "crypto_start_date" not in st.session_state:
        # Set default to 3 months ago or min_date, whichever is later
        default_start = max(current_date - timedelta(days=90), min_date)
        st.session_state.crypto_start_date = default_start
    if "crypto_end_date" not in st.session_state:
        st.session_state.crypto_end_date = max_date

    # Ensure session state dates are within bounds
    if st.session_state.crypto_start_date < min_date:
        st.session_state.crypto_start_date = min_date
    if st.session_state.crypto_start_date > max_date:
        st.session_state.crypto_start_date = max_date
    if st.session_state.crypto_end_date < min_date:
        st.session_state.crypto_end_date = min_date
    if st.session_state.crypto_end_date > max_date:
        st.session_state.crypto_end_date = max_date

    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=st.session_state.crypto_start_date,
            min_value=min_date,
            max_value=max_date,
            key="crypto_start_date_input",
        )
        st.session_state.crypto_start_date = start_date
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=st.session_state.crypto_end_date,
            min_value=min_date,
            max_value=max_date,
            key="crypto_end_date_input",
        )
        st.session_state.crypto_end_date = end_date

    # Convert to timezone-aware timestamps
    if available_cryptos:
        sample_crypto = available_cryptos[0]
        sample_data = st.session_state.crypto_analyzer.data[sample_crypto]
        if sample_data.index.tz is not None:
            start_date = pd.Timestamp(start_date).tz_localize(sample_data.index.tz)
            end_date = pd.Timestamp(end_date).tz_localize(sample_data.index.tz)
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

    # Create tabs for each cryptocurrency
    crypto_tabs = st.tabs([f"🪙 {crypto}" for crypto in available_cryptos])

    for i, crypto in enumerate(available_cryptos):
        with crypto_tabs[i]:
            if crypto not in st.session_state.crypto_analyzer.data:
                st.warning(f"⚠️ No data available for {crypto}")
                continue

            # Get crypto info
            crypto_info = st.session_state.crypto_analyzer.get_crypto_info(crypto)

            # Display key metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "💰 Current Price",
                    f"${crypto_info['current_price']:,.4f}",
                    f"{crypto_info['price_change_pct_24h']:+.2f}%",
                )
            with col2:
                st.metric("📊 24h Volume", f"{crypto_info['volume_24h']:,.0f}")
            with col3:
                current_data = st.session_state.crypto_analyzer.data[crypto].iloc[-1]
                st.metric("📈 RSI", f"{current_data['RSI']:.1f}")
            with col4:
                st.metric("⚡ MACD", f"{current_data['MACD']:.6f}")
            with col5:
                st.metric("📈 7d Change", f"{crypto_info['price_change_pct_7d']:+.2f}%")
            with col6:
                volatility = current_data.get("Volatility", 0)
                st.metric("🌊 Volatility", f"{volatility:.1f}%")

            # Trading signals section
            signals_df = st.session_state.crypto_analyzer.generate_crypto_signals(
                crypto
            )
            if not signals_df.empty:
                latest_signal = signals_df.iloc[-1]
                signal_color = {
                    "STRONG BUY": "🟢",
                    "BUY": "🟢",
                    "HOLD": "🟡",
                    "SELL": "🔴",
                    "STRONG SELL": "🔴",
                }.get(latest_signal["Signal"], "⚪")

                st.info(
                    f"🎯 **Latest Signal**: {signal_color} {latest_signal['Signal']} - {latest_signal['Reason']}"
                )

            # Price Chart
            st.subheader("📈 Price Analysis & Technical Indicators")
            fig1 = st.session_state.crypto_analyzer.create_crypto_price_chart(
                crypto, start_date, end_date
            )
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)

            # Create two columns for MACD and RSI
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 MACD Analysis")
                fig2 = st.session_state.crypto_analyzer.create_macd_chart(
                    crypto, start_date, end_date
                )
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

            with col2:
                st.subheader("⚡ RSI Analysis")
                fig3 = st.session_state.crypto_analyzer.create_rsi_chart(
                    crypto, start_date, end_date
                )
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)

            # Volatility Chart
            st.subheader("🌊 Price Volatility Analysis")
            fig4 = st.session_state.crypto_analyzer.create_volatility_chart(
                crypto, start_date, end_date
            )
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)

            # Trading Signals Table
            if not signals_df.empty:
                st.subheader("🎯 Recent Trading Signals")

                # Display last 10 signals
                recent_signals = signals_df.tail(10).copy()
                recent_signals["Date"] = recent_signals["Date"].dt.strftime("%Y-%m-%d")
                recent_signals["Price"] = recent_signals["Price"].apply(
                    lambda x: f"${x:.4f}"
                )
                recent_signals["RSI"] = recent_signals["RSI"].apply(
                    lambda x: f"{x:.1f}"
                )
                recent_signals["MACD"] = recent_signals["MACD"].apply(
                    lambda x: f"{x:.6f}"
                )
                recent_signals["Volatility"] = recent_signals["Volatility"].apply(
                    lambda x: f"{x:.1f}%"
                )

                # Color code signals
                def color_signals(val):
                    if val in ["STRONG BUY", "BUY"]:
                        return "background-color: rgba(0, 255, 0, 0.3)"
                    elif val in ["STRONG SELL", "SELL"]:
                        return "background-color: rgba(255, 0, 0, 0.3)"
                    elif val == "HOLD":
                        return "background-color: rgba(255, 255, 0, 0.3)"
                    return ""

                styled_df = recent_signals.style.applymap(
                    color_signals, subset=["Signal"]
                ).format(
                    {
                        "Date": lambda x: x,
                        "Price": lambda x: x,
                        "Signal": lambda x: x,
                        "Strength": lambda x: x,
                        "RSI": lambda x: x,
                        "MACD": lambda x: x,
                        "Volatility": lambda x: x,
                    }
                )

                st.dataframe(styled_df, use_container_width=True)

            # Market Analysis Summary
            st.subheader("📋 Market Analysis Summary")

            # Create analysis summary
            current_rsi = current_data["RSI"]
            current_macd = current_data["MACD"]
            current_signal = current_data["MACD_Signal"]
            current_price = current_data["Close"]
            sma_20 = current_data["SMA_20"]
            sma_50 = current_data["SMA_50"]
            sma_200 = current_data["SMA_200"]

            # Market sentiment analysis
            sentiment_score = 0
            sentiment_factors = []

            if current_rsi < 30:
                sentiment_score += 2
                sentiment_factors.append(
                    "🟢 RSI indicates oversold conditions (bullish)"
                )
            elif current_rsi > 70:
                sentiment_score -= 2
                sentiment_factors.append(
                    "🔴 RSI indicates overbought conditions (bearish)"
                )
            else:
                sentiment_factors.append("🟡 RSI in neutral zone")

            if current_macd > current_signal:
                sentiment_score += 1
                sentiment_factors.append("🟢 MACD above signal line (bullish)")
            else:
                sentiment_score -= 1
                sentiment_factors.append("🔴 MACD below signal line (bearish)")

            if current_price > sma_20 > sma_50:
                sentiment_score += 2
                sentiment_factors.append(
                    "🟢 Price above short-term moving averages (bullish)"
                )
            elif current_price < sma_20 < sma_50:
                sentiment_score -= 2
                sentiment_factors.append(
                    "🔴 Price below short-term moving averages (bearish)"
                )

            if current_price > sma_200:
                sentiment_score += 1
                sentiment_factors.append(
                    "🟢 Price above 200-day SMA (long-term bullish)"
                )
            else:
                sentiment_score -= 1
                sentiment_factors.append(
                    "🔴 Price below 200-day SMA (long-term bearish)"
                )

            # Determine overall sentiment
            if sentiment_score >= 3:
                overall_sentiment = "🚀 **VERY BULLISH**"
                sentiment_color = "success"
            elif sentiment_score >= 1:
                overall_sentiment = "📈 **BULLISH**"
                sentiment_color = "success"
            elif sentiment_score <= -3:
                overall_sentiment = "📉 **VERY BEARISH**"
                sentiment_color = "error"
            elif sentiment_score <= -1:
                overall_sentiment = "📉 **BEARISH**"
                sentiment_color = "error"
            else:
                overall_sentiment = "➡️ **NEUTRAL**"
                sentiment_color = "info"

            st.markdown(f"### Overall Market Sentiment: {overall_sentiment}")

            # Display factors
            st.markdown("**Key Factors:**")
            for factor in sentiment_factors:
                st.markdown(f"• {factor}")

            # Price targets and support/resistance
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📊 Key Levels:**")
                resistance = current_data.get("Resistance", current_price * 1.05)
                support = current_data.get("Support", current_price * 0.95)
                st.markdown(f"• **Resistance**: ${resistance:.4f}")
                st.markdown(f"• **Support**: ${support:.4f}")
                st.markdown(f"• **All-Time High**: ${crypto_info['ath']:.4f}")
                st.markdown(f"• **All-Time Low**: ${crypto_info['atl']:.4f}")

            with col2:
                st.markdown("**🎯 Moving Averages:**")
                st.markdown(f"• **SMA 20**: ${sma_20:.4f}")
                st.markdown(f"• **SMA 50**: ${sma_50:.4f}")
                st.markdown(f"• **SMA 200**: ${sma_200:.4f}")

                # Distance from MA
                distance_sma20 = ((current_price - sma_20) / sma_20) * 100
                st.markdown(f"• **Distance from SMA20**: {distance_sma20:+.2f}%")

    # Portfolio overview section
    if len(available_cryptos) > 1:
        st.header("🏦 Portfolio Overview")

        # Create portfolio comparison chart
        portfolio_data = []
        for crypto in available_cryptos:
            if crypto in st.session_state.crypto_analyzer.data:
                info = st.session_state.crypto_analyzer.get_crypto_info(crypto)
                portfolio_data.append(
                    {
                        "Cryptocurrency": crypto,
                        "Current Price": info["current_price"],
                        "24h Change %": info["price_change_pct_24h"],
                        "7d Change %": info["price_change_pct_7d"],
                        "Volume 24h": info["volume_24h"],
                        "ATH": info["ath"],
                        "ATL": info["atl"],
                    }
                )

        portfolio_df = pd.DataFrame(portfolio_data)

        # Display portfolio table
        st.subheader("📊 Portfolio Performance")

        def color_change(val):
            if val > 0:
                return "color: green"
            elif val < 0:
                return "color: red"
            return ""

        styled_portfolio = portfolio_df.style.applymap(
            color_change, subset=["24h Change %", "7d Change %"]
        ).format(
            {
                "Current Price": "${:.4f}",
                "24h Change %": "{:+.2f}%",
                "7d Change %": "{:+.2f}%",
                "Volume 24h": "{:,.0f}",
                "ATH": "${:.4f}",
                "ATL": "${:.4f}",
            }
        )

        st.dataframe(styled_portfolio, use_container_width=True)

        # Create portfolio performance comparison chart
        fig_portfolio = go.Figure()

        for crypto in available_cryptos:
            if crypto in st.session_state.crypto_analyzer.data:
                df = st.session_state.crypto_analyzer.data[crypto]
                mask = (df.index >= start_date) & (df.index <= end_date)
                filtered_df = df.loc[mask]

                if not filtered_df.empty:
                    # Normalize prices to show percentage change
                    normalized_prices = (
                        filtered_df["Close"] / filtered_df["Close"].iloc[0] - 1
                    ) * 100

                    fig_portfolio.add_trace(
                        go.Scatter(
                            x=filtered_df.index,
                            y=normalized_prices,
                            mode="lines",
                            name=crypto,
                            line=dict(width=3),
                        )
                    )

        fig_portfolio.update_layout(
            title="🏆 Portfolio Performance Comparison (Normalized %)",
            xaxis_title="Date",
            yaxis_title="Price Change (%)",
            hovermode="x unified",
            template="plotly_dark",
            height=500,
            plot_bgcolor="rgba(0,0,0,0.1)",
        )

        st.plotly_chart(fig_portfolio, use_container_width=True)


# Main function for the crypto page
def main():
    """Entry point for the crypto page"""
    show_crypto_page()


# Allow standalone running
if __name__ == "__main__":
    st.set_page_config(
        page_title="Crypto Analysis Dashboard",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
