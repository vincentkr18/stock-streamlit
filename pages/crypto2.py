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
            "Tether": "USDT-USD",
            "BNB": "BNB-USD",
            "Solana": "SOL-USD",
            "USD Coin": "USDC-USD",
            "Solana": "SOL-USD",
            "Dogecoin": "DOGE-USD",
            "Shiba Inu": "SHIB-USD",
            "Cardano": "ADA-USD",
            "Polygon": "MATIC-USD",
            "Chainlink": "LINK-USD",
            "Litecoin": "LTC-USD",
            "Avalanche": "AVAX-USD",
            "Polkadot": "DOT-USD",
            "TRON": "TRX-USD",
            "Bitcoin Cash": "BCH-USD",
            "Hedera": "HBAR-USD",
            "UNUS SED LEO": "LEO-USD",
            "Litecoin": "LTC-USD",
            "Cronos": "CRO-USD",
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
            if hasattr(start_date, "tz") and start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            elif not hasattr(start_date, "tz"):
                start_date = pd.Timestamp(start_date).tz_localize(df.index.tz)
            if hasattr(end_date, "tz") and end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)
            elif not hasattr(end_date, "tz"):
                end_date = pd.Timestamp(end_date).tz_localize(df.index.tz)

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
            if hasattr(start_date, "tz") and start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            elif not hasattr(start_date, "tz"):
                start_date = pd.Timestamp(start_date).tz_localize(df.index.tz)
            if hasattr(end_date, "tz") and end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)
            elif not hasattr(end_date, "tz"):
                end_date = pd.Timestamp(end_date).tz_localize(df.index.tz)

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
            if hasattr(start_date, "tz") and start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            elif not hasattr(start_date, "tz"):
                start_date = pd.Timestamp(start_date).tz_localize(df.index.tz)
            if hasattr(end_date, "tz") and end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)
            elif not hasattr(end_date, "tz"):
                end_date = pd.Timestamp(end_date).tz_localize(df.index.tz)

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
            if hasattr(start_date, "tz") and start_date.tz is None:
                start_date = start_date.tz_localize(df.index.tz)
            elif not hasattr(start_date, "tz"):
                start_date = pd.Timestamp(start_date).tz_localize(df.index.tz)
            if hasattr(end_date, "tz") and end_date.tz is None:
                end_date = end_date.tz_localize(df.index.tz)
            elif not hasattr(end_date, "tz"):
                end_date = pd.Timestamp(end_date).tz_localize(df.index.tz)

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

    def calculate_buy_sell_returns(self, signals_df):
        """Calculate returns from buy to sell signals"""
        if signals_df.empty:
            return pd.DataFrame()

        returns_data = []
        buy_price = None
        buy_date = None

        for _, row in signals_df.iterrows():
            if row["Signal"] in ["BUY", "STRONG BUY"] and buy_price is None:
                buy_price = row["Price"]
                buy_date = row["Date"]
                returns_data.append(
                    {
                        "Date": row["Date"],
                        "Signal": row["Signal"],
                        "Price": row["Price"],
                        "Return_%": 0.0,
                        "Action": "BUY",
                    }
                )

            elif row["Signal"] in ["SELL", "STRONG SELL"] and buy_price is not None:
                sell_price = row["Price"]
                return_pct = ((sell_price - buy_price) / buy_price) * 100
                returns_data.append(
                    {
                        "Date": row["Date"],
                        "Signal": row["Signal"],
                        "Price": row["Price"],
                        "Return_%": return_pct,
                        "Action": "SELL",
                        "Buy_Price": buy_price,
                        "Buy_Date": buy_date,
                    }
                )
                buy_price = None
                buy_date = None

        return pd.DataFrame(returns_data)

    def create_monthly_returns_chart(self, signals_df, crypto_name):
        """Create monthly returns bar chart based on buy/sell signals"""
        if signals_df.empty:
            return None

        returns_df = self.calculate_buy_sell_returns(signals_df)
        sell_signals = returns_df[returns_df["Action"] == "SELL"].copy()

        if sell_signals.empty:
            return None

        # Group by month
        sell_signals["Month"] = sell_signals["Date"].dt.to_period("M")
        monthly_returns = sell_signals.groupby("Month")["Return_%"].sum().reset_index()
        monthly_returns["Month"] = monthly_returns["Month"].astype(str)

        fig = go.Figure()

        colors = ["green" if x >= 0 else "red" for x in monthly_returns["Return_%"]]

        fig.add_trace(
            go.Bar(
                x=monthly_returns["Month"],
                y=monthly_returns["Return_%"],
                marker_color=colors,
                name="Monthly Returns",
                text=[f"{x:.1f}%" for x in monthly_returns["Return_%"]],
                textposition="outside",
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

        fig.update_layout(
            title=f"{crypto_name} - Monthly Trading Returns (%)",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400,
            plot_bgcolor="rgba(0,0,0,0.1)",
            showlegend=False,
        )

        return fig

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

    # Quick select buttons (continuing from where it was cut off)
    quick_periods = [
        ("1D", 1),
        ("1W", 7),
        ("1M", 30),
        ("3M", 90),
        ("6M", 180),
        ("1Y", 365),
        ("2Y", 730),
        ("5Y", 1825),
        ("All", None),
    ]

    selected_period = None

    with col1:
        if st.button("1D"):
            selected_period = 1
    with col2:
        if st.button("1W"):
            selected_period = 7
    with col3:
        if st.button("1M"):
            selected_period = 30
    with col4:
        if st.button("3M"):
            selected_period = 90
    with col5:
        if st.button("6M"):
            selected_period = 180
    with col6:
        if st.button("1Y"):
            selected_period = 365
    with col7:
        if st.button("2Y"):
            selected_period = 730
    with col8:
        if st.button("5Y"):
            selected_period = 1825
    with col9:
        if st.button("All"):
            selected_period = None

    # Set date range based on quick selection or manual selection
    if selected_period is not None:
        end_date = current_date
        start_date = (
            datetime.combine(current_date, datetime.min.time())
            - timedelta(days=selected_period)
        ).date()
        start_date = max(start_date, min_date)  # Don't go before available data
    else:
        start_date = (
            min_date
            if selected_period is None and "selected_period" in locals()
            else min_date
        )
        end_date = current_date

    # Manual date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input(
            "Start Date", value=start_date, min_value=min_date, max_value=max_date
        )
    with col2:
        end_date_input = st.date_input(
            "End Date", value=end_date, min_value=min_date, max_value=max_date
        )

    # Use manual input if provided
    if start_date_input and end_date_input:
        start_date = start_date_input
        end_date = end_date_input

    # Convert to datetime for analysis
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())

    # Crypto selection for detailed analysis
    st.subheader("🎯 Select Cryptocurrency for Analysis")
    selected_crypto = st.selectbox(
        "Choose cryptocurrency:", available_cryptos, key="crypto_analysis_select"
    )

    if selected_crypto:
        # Display crypto information
        info = st.session_state.crypto_analyzer.get_crypto_info(selected_crypto)

        # Create metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            price_color = (
                "normal" if info.get("price_change_pct_24h", 0) >= 0 else "inverse"
            )
            st.metric(
                "💰 Current Price",
                f"${info.get('current_price', 0):,.4f}",
                f"{info.get('price_change_pct_24h', 0):+.2f}%",
                delta_color=price_color,
            )

        with col2:
            price_7d_color = (
                "normal" if info.get("price_change_pct_7d", 0) >= 0 else "inverse"
            )
            st.metric(
                "📊 7d Change",
                f"${info.get('price_change_7d', 0):+,.4f}",
                f"{info.get('price_change_pct_7d', 0):+.2f}%",
                delta_color=price_7d_color,
            )

        with col3:
            st.metric("📈 ATH", f"${info.get('ath', 0):,.4f}")

        with col4:
            st.metric("📉 ATL", f"${info.get('atl', 0):,.4f}")

        with col5:
            st.metric("💎 Volume (24h)", f"${info.get('volume_24h', 0):,.0f}")

        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📈 Price Chart",
                "🔄 MACD",
                "⚡ RSI",
                "🌊 Volatility",
                "🚦 Signals",
                "💹 Returns",
            ]
        )

        with tab1:
            st.subheader(f"📈 {selected_crypto} Price Analysis")
            price_chart = st.session_state.crypto_analyzer.create_crypto_price_chart(
                selected_crypto, start_datetime, end_datetime
            )
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
            else:
                st.warning("⚠️ No data available for the selected date range")

        with tab2:
            st.subheader(f"🔄 {selected_crypto} MACD Analysis")
            macd_chart = st.session_state.crypto_analyzer.create_macd_chart(
                selected_crypto, start_datetime, end_datetime
            )
            if macd_chart:
                st.plotly_chart(macd_chart, use_container_width=True)

                # MACD interpretation
                st.info(
                    """
                **📖 MACD Interpretation:**
                - **Bullish Signal**: MACD line crosses above Signal line
                - **Bearish Signal**: MACD line crosses below Signal line  
                - **Histogram**: Shows momentum strength
                - **Above Zero**: Generally bullish momentum
                - **Below Zero**: Generally bearish momentum
                """
                )
            else:
                st.warning("⚠️ No MACD data available for the selected date range")

        with tab3:
            st.subheader(f"⚡ {selected_crypto} RSI Analysis")
            rsi_chart = st.session_state.crypto_analyzer.create_rsi_chart(
                selected_crypto, start_datetime, end_datetime
            )
            if rsi_chart:
                st.plotly_chart(rsi_chart, use_container_width=True)

                # RSI interpretation
                st.info(
                    """
                **📖 RSI Interpretation:**
                - **Above 70**: Potentially overbought (consider selling)
                - **Below 30**: Potentially oversold (consider buying)
                - **50**: Neutral momentum
                - **Trending Up**: Bullish momentum building
                - **Trending Down**: Bearish momentum building
                """
                )
            else:
                st.warning("⚠️ No RSI data available for the selected date range")

        with tab4:
            st.subheader(f"🌊 {selected_crypto} Volatility Analysis")
            volatility_chart = st.session_state.crypto_analyzer.create_volatility_chart(
                selected_crypto, start_datetime, end_datetime
            )
            if volatility_chart:
                st.plotly_chart(volatility_chart, use_container_width=True)

                # Volatility interpretation
                st.info(
                    """
                **📖 Volatility Interpretation:**
                - **High Volatility**: Greater price swings, higher risk/reward
                - **Low Volatility**: More stable prices, lower risk/reward
                - **Above Average**: Market uncertainty or strong momentum
                - **Below Average**: Market consolidation or stability
                """
                )
            else:
                st.warning("⚠️ No volatility data available for the selected date range")

        with tab5:
            st.subheader(f"🚦 {selected_crypto} Trading Signals")

            # Generate signals
            signals_df = st.session_state.crypto_analyzer.generate_crypto_signals(
                selected_crypto
            )

            if not signals_df.empty:
                # Filter signals by date range - handle timezone compatibility
                start_datetime_compare, end_datetime_compare = (
                    handle_timezone_comparison(
                        signals_df["Date"], start_datetime, end_datetime
                    )
                )

                mask = (signals_df["Date"] >= start_datetime_compare) & (
                    signals_df["Date"] <= end_datetime_compare
                )
                filtered_signals = signals_df.loc[mask]

                if not filtered_signals.empty:
                    # Display recent signals
                    st.write("**🔥 Recent Trading Signals:**")

                    # Color coding for signals
                    def color_signals(val):
                        if val in ["STRONG BUY", "BUY"]:
                            return "color: green; font-weight: bold"
                        elif val in ["STRONG SELL", "SELL"]:
                            return "color: red; font-weight: bold"
                        else:
                            return "color: orange; font-weight: bold"

                    # Style the dataframe
                    styled_signals = (
                        filtered_signals.tail(20)
                        .style.applymap(color_signals, subset=["Signal"])
                        .format(
                            {
                                "Price": "${:.4f}",
                                "RSI": "{:.1f}",
                                "MACD": "{:.6f}",
                                "Volatility": "{:.1f}%",
                            }
                        )
                    )

                    st.dataframe(styled_signals, use_container_width=True)

                    # Signal summary
                    signal_counts = filtered_signals["Signal"].value_counts()

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "🟢 Buy Signals",
                            signal_counts.get("BUY", 0)
                            + signal_counts.get("STRONG BUY", 0),
                        )
                    with col2:
                        st.metric(
                            "🔴 Sell Signals",
                            signal_counts.get("SELL", 0)
                            + signal_counts.get("STRONG SELL", 0),
                        )
                    with col3:
                        st.metric("🟡 Hold Signals", signal_counts.get("HOLD", 0))
                    with col4:
                        if len(filtered_signals) > 0:
                            latest_signal = filtered_signals.iloc[-1]["Signal"]
                            st.metric("🎯 Latest Signal", latest_signal)

                else:
                    st.info("📋 No trading signals in the selected date range")
            else:
                st.warning(
                    "⚠️ Unable to generate trading signals for this cryptocurrency"
                )

        with tab6:
            st.subheader(f"💹 {selected_crypto} Trading Returns Analysis")

            if not signals_df.empty:
                # Calculate and display returns
                returns_chart = (
                    st.session_state.crypto_analyzer.create_monthly_returns_chart(
                        signals_df, selected_crypto
                    )
                )

                if returns_chart:
                    st.plotly_chart(returns_chart, use_container_width=True)

                    # Calculate overall performance metrics
                    returns_df = (
                        st.session_state.crypto_analyzer.calculate_buy_sell_returns(
                            signals_df
                        )
                    )

                    if not returns_df.empty:
                        sell_trades = returns_df[returns_df["Action"] == "SELL"]

                        if not sell_trades.empty:
                            total_return = sell_trades["Return_%"].sum()
                            avg_return = sell_trades["Return_%"].mean()
                            win_rate = (
                                (sell_trades["Return_%"] > 0).sum()
                                / len(sell_trades)
                                * 100
                            )
                            num_trades = len(sell_trades)

                            # Display performance metrics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                return_color = (
                                    "normal" if total_return >= 0 else "inverse"
                                )
                                st.metric(
                                    "📊 Total Return",
                                    f"{total_return:.2f}%",
                                    delta_color=return_color,
                                )

                            with col2:
                                avg_color = "normal" if avg_return >= 0 else "inverse"
                                st.metric(
                                    "📈 Avg Return/Trade",
                                    f"{avg_return:.2f}%",
                                    delta_color=avg_color,
                                )

                            with col3:
                                win_color = "normal" if win_rate >= 50 else "inverse"
                                st.metric(
                                    "🎯 Win Rate",
                                    f"{win_rate:.1f}%",
                                    delta_color=win_color,
                                )

                            with col4:
                                st.metric("🔄 Total Trades", f"{num_trades}")

                            # Performance interpretation
                            if total_return > 0:
                                st.success(
                                    f"🎉 Strategy shows positive returns of {total_return:.2f}% over {num_trades} trades!"
                                )
                            else:
                                st.error(
                                    f"⚠️ Strategy shows negative returns of {total_return:.2f}% over {num_trades} trades."
                                )

                        else:
                            st.info(
                                "📋 No completed trades (buy-sell pairs) found in the data period"
                            )
                    else:
                        st.info("📋 No return data available")
                else:
                    st.info("📋 No returns data available for charting")
            else:
                st.warning("⚠️ No signals available to calculate returns")

    # Multi-crypto comparison section
    st.subheader("⚖️ Multi-Crypto Comparison")

    if len(available_cryptos) > 1:
        comparison_cryptos = st.multiselect(
            "Select cryptocurrencies to compare:",
            available_cryptos,
            default=(
                available_cryptos[:3]
                if len(available_cryptos) >= 3
                else available_cryptos
            ),
        )

        if comparison_cryptos and len(comparison_cryptos) > 1:
            # Create comparison chart
            fig = go.Figure()

            for crypto in comparison_cryptos:
                if crypto in st.session_state.crypto_analyzer.data:
                    df = st.session_state.crypto_analyzer.data[crypto]

                    # Handle timezone for comparison using helper function
                    start_datetime_tz, end_datetime_tz = handle_timezone_comparison(
                        df.index, start_datetime, end_datetime
                    )

                    mask = (df.index >= start_datetime_tz) & (
                        df.index <= end_datetime_tz
                    )
                    filtered_df = df.loc[mask]

                    if not filtered_df.empty:
                        # Normalize prices to percentage change from first value
                        normalized_prices = (
                            filtered_df["Close"] / filtered_df["Close"].iloc[0] - 1
                        ) * 100

                        fig.add_trace(
                            go.Scatter(
                                x=filtered_df.index,
                                y=normalized_prices,
                                mode="lines",
                                name=crypto,
                                line=dict(width=3),
                            )
                        )

            fig.update_layout(
                title="📊 Cryptocurrency Performance Comparison (Normalized %)",
                xaxis_title="Date",
                yaxis_title="Return (%)",
                hovermode="x unified",
                template="plotly_dark",
                height=500,
                plot_bgcolor="rgba(0,0,0,0.1)",
            )

            fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

            st.plotly_chart(fig, use_container_width=True)

            # Comparison table
            comparison_data = []
            for crypto in comparison_cryptos:
                info = st.session_state.crypto_analyzer.get_crypto_info(crypto)
                if info:
                    comparison_data.append(
                        {
                            "Crypto": crypto,
                            "Current Price": f"${info.get('current_price', 0):,.4f}",
                            "24h Change": f"{info.get('price_change_pct_24h', 0):+.2f}%",
                            "7d Change": f"{info.get('price_change_pct_7d', 0):+.2f}%",
                            "Volume (24h)": f"${info.get('volume_24h', 0):,.0f}",
                            "ATH": f"${info.get('ath', 0):,.4f}",
                            "ATL": f"${info.get('atl', 0):,.4f}",
                        }
                    )

            if comparison_data:
                st.write("**📋 Comparison Table:**")
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("💡 Load more cryptocurrencies to enable comparison features")

        # NEW SECTION: Trading Performance Analysis Table
    st.subheader(f"📋 {selected_crypto} Trading Performance Analysis")

    if not signals_df.empty:
        # Filter signals by date range first
        start_datetime_compare, end_datetime_compare = handle_timezone_comparison(
            signals_df["Date"], start_datetime, end_datetime
        )

        mask = (signals_df["Date"] >= start_datetime_compare) & (
            signals_df["Date"] <= end_datetime_compare
        )
        filtered_signals = signals_df.loc[mask]

        if not filtered_signals.empty:
            # Create trading pairs table
            trading_pairs = []
            current_buy = None

            for _, row in filtered_signals.iterrows():
                signal = row["Signal"]

                # Handle buy signals - only take the latest buy, ignore consecutive buys
                if signal in ["BUY", "STRONG BUY"]:
                    current_buy = {
                        "buy_date": row["Date"],
                        "buy_price": row["Price"],
                        "buy_signal": signal,
                        "buy_rsi": row["RSI"],
                        "buy_reason": row["Reason"],
                    }

                # Handle sell signals - match with latest buy
                elif signal in ["SELL", "STRONG SELL"] and current_buy is not None:
                    price_change = row["Price"] - current_buy["buy_price"]
                    price_change_pct = (price_change / current_buy["buy_price"]) * 100

                    trading_pairs.append(
                        {
                            "Buy Date": current_buy["buy_date"].strftime("%Y-%m-%d"),
                            "Buy Price": current_buy["buy_price"],
                            "Buy Signal": current_buy["buy_signal"],
                            "Buy RSI": current_buy["buy_rsi"],
                            "Sell Date": row["Date"].strftime("%Y-%m-%d"),
                            "Sell Price": row["Price"],
                            "Sell Signal": signal,
                            "Sell RSI": row["RSI"],
                            "Price Change ($)": price_change,
                            "Price Change (%)": price_change_pct,
                            "Days Held": (row["Date"] - current_buy["buy_date"]).days,
                            "Buy Reason": current_buy["buy_reason"],
                            "Sell Reason": row["Reason"],
                        }
                    )

                    # Reset current buy after successful sell
                    current_buy = None

            if trading_pairs:
                st.write("**🔄 Buy-Sell Trading Pairs:**")

                trading_df = pd.DataFrame(trading_pairs)

                # Style the dataframe with conditional formatting
                def style_returns(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "background-color: rgba(0, 255, 0, 0.2); color: green; font-weight: bold"
                        elif val < 0:
                            return "background-color: rgba(255, 0, 0, 0.2); color: red; font-weight: bold"
                    return ""

                def style_signals(val):
                    if "STRONG BUY" in str(val) or "BUY" in str(val):
                        return "color: green; font-weight: bold"
                    elif "STRONG SELL" in str(val) or "SELL" in str(val):
                        return "color: red; font-weight: bold"
                    return ""

                styled_trading_df = (
                    trading_df.style.applymap(
                        style_returns, subset=["Price Change ($)", "Price Change (%)"]
                    )
                    .applymap(style_signals, subset=["Buy Signal", "Sell Signal"])
                    .format(
                        {
                            "Buy Price": "${:.4f}",
                            "Sell Price": "${:.4f}",
                            "Price Change ($)": "${:+.4f}",
                            "Price Change (%)": "{:+.2f}%",
                            "Buy RSI": "{:.1f}",
                            "Sell RSI": "{:.1f}",
                        }
                    )
                )

                st.dataframe(styled_trading_df, use_container_width=True)

                # Summary statistics for trading pairs
                col1, col2, col3, col4, col5 = st.columns(5)

                total_trades = len(trading_pairs)
                winning_trades = sum(
                    1 for trade in trading_pairs if trade["Price Change (%)"] > 0
                )
                losing_trades = total_trades - winning_trades
                avg_return = (
                    sum(trade["Price Change (%)"] for trade in trading_pairs)
                    / total_trades
                    if total_trades > 0
                    else 0
                )
                total_return = sum(trade["Price Change (%)"] for trade in trading_pairs)
                avg_hold_days = (
                    sum(trade["Days Held"] for trade in trading_pairs) / total_trades
                    if total_trades > 0
                    else 0
                )

                with col1:
                    st.metric("🎯 Total Pairs", total_trades)

                with col2:
                    win_rate = (
                        (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    )
                    st.metric(
                        "📈 Win Rate",
                        f"{win_rate:.1f}%",
                        f"{winning_trades}W / {losing_trades}L",
                    )

                with col3:
                    return_color = "normal" if avg_return >= 0 else "inverse"
                    st.metric(
                        "📊 Avg Return", f"{avg_return:.2f}%", delta_color=return_color
                    )

                with col4:
                    total_color = "normal" if total_return >= 0 else "inverse"
                    st.metric(
                        "💰 Total Return",
                        f"{total_return:.2f}%",
                        delta_color=total_color,
                    )

                with col5:
                    st.metric("📅 Avg Hold", f"{avg_hold_days:.1f} days")

                # Show best and worst trades
                if trading_pairs:
                    best_trade = max(trading_pairs, key=lambda x: x["Price Change (%)"])
                    worst_trade = min(
                        trading_pairs, key=lambda x: x["Price Change (%)"]
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(
                            f"""
                        **🏆 Best Trade:**
                        - Date: {best_trade['Buy Date']} → {best_trade['Sell Date']}
                        - Return: {best_trade['Price Change (%)']:+.2f}%
                        - Price: ${best_trade['Buy Price']:.4f} → ${best_trade['Sell Price']:.4f}
                        """
                        )

                    with col2:
                        st.error(
                            f"""
                        **📉 Worst Trade:**
                        - Date: {worst_trade['Buy Date']} → {worst_trade['Sell Date']}
                        - Return: {worst_trade['Price Change (%)']:+.2f}%
                        - Price: ${worst_trade['Buy Price']:.4f} → ${worst_trade['Sell Price']:.4f}
                        """
                        )

            else:
                st.info(
                    "📋 No completed buy-sell pairs found in the selected date range"
                )

            # Show pending buy signal (if any)
            if current_buy is not None:
                st.warning(
                    f"""
                **⏳ Pending Buy Signal:**
                - Date: {current_buy['buy_date'].strftime('%Y-%m-%d')}
                - Price: ${current_buy['buy_price']:.4f}
                - Signal: {current_buy['buy_signal']}
                - Waiting for sell signal...
                """
                )

        else:
            st.info("📋 No signals available in the selected date range")

        # Yearly Trading Performance Analysis (Last 6 Years)
        st.subheader(f"📅 {selected_crypto} Yearly Trading Performance (Last 6 Years)")

        if not signals_df.empty:
            # Get the last 6 years of data
            current_year = datetime.now().year
            yearly_crypto_performance = []

            for year in range(current_year - 5, current_year + 1):
                year_start = datetime(year, 1, 1)
                year_end = datetime(year, 12, 31, 23, 59, 59)

                # Handle timezone compatibility for yearly analysis
                start_datetime_compare, end_datetime_compare = (
                    handle_timezone_comparison(signals_df["Date"], year_start, year_end)
                )

                # Filter signals for this year
                mask = (signals_df["Date"] >= start_datetime_compare) & (
                    signals_df["Date"] <= end_datetime_compare
                )
                year_signals = signals_df.loc[mask]

                if not year_signals.empty:
                    # Create trading pairs for this year
                    year_trades = []
                    current_buy = None

                    for _, row in year_signals.iterrows():
                        signal = row["Signal"]

                        # Handle buy signals
                        if signal in ["BUY", "STRONG BUY"]:
                            current_buy = {
                                "buy_date": row["Date"],
                                "buy_price": row["Price"],
                                "buy_signal": signal,
                                "buy_rsi": row["RSI"],
                                "buy_reason": row["Reason"],
                            }

                        # Handle sell signals
                        elif (
                            signal in ["SELL", "STRONG SELL"]
                            and current_buy is not None
                        ):
                            price_change = row["Price"] - current_buy["buy_price"]
                            price_change_pct = (
                                price_change / current_buy["buy_price"]
                            ) * 100

                            year_trades.append(
                                {
                                    "Buy Date": current_buy["buy_date"].strftime(
                                        "%Y-%m-%d"
                                    ),
                                    "Buy Price": current_buy["buy_price"],
                                    "Buy Signal": current_buy["buy_signal"],
                                    "Sell Date": row["Date"].strftime("%Y-%m-%d"),
                                    "Sell Price": row["Price"],
                                    "Sell Signal": signal,
                                    "Price Change (%)": price_change_pct,
                                    "Days Held": (
                                        row["Date"] - current_buy["buy_date"]
                                    ).days,
                                }
                            )

                            current_buy = None

                    # Add any remaining open buy position
                    if current_buy is not None:
                        year_trades.append(
                            {
                                "Buy Date": current_buy["buy_date"].strftime(
                                    "%Y-%m-%d"
                                ),
                                "Buy Price": current_buy["buy_price"],
                                "Buy Signal": current_buy["buy_signal"],
                                "Sell Date": "",
                                "Sell Price": "",
                                "Sell Signal": "",
                                "Price Change (%)": "",
                                "Days Held": "",
                            }
                        )

                    if year_trades:
                        yearly_crypto_performance.append(
                            {"Year": year, "Trades": year_trades}
                        )

            # Display yearly performance in tabs
            if yearly_crypto_performance:
                crypto_year_tabs = st.tabs(
                    [f"{data['Year']}" for data in yearly_crypto_performance]
                )

                for i, year_data in enumerate(yearly_crypto_performance):
                    with crypto_year_tabs[i]:
                        trades_df = pd.DataFrame(year_data["Trades"])

                        # Calculate metrics for this year
                        completed_trades = trades_df[
                            trades_df["Price Change (%)"] != ""
                        ]
                        total_signals = len(trades_df)
                        completed_count = len(completed_trades)
                        open_positions = total_signals - completed_count

                        if completed_count > 0:
                            profitable_trades = len(
                                completed_trades[
                                    completed_trades["Price Change (%)"] > 0
                                ]
                            )
                            avg_return = completed_trades["Price Change (%)"].mean()
                            total_return = completed_trades["Price Change (%)"].sum()
                            win_rate = (profitable_trades / completed_count) * 100
                        else:
                            profitable_trades = 0
                            avg_return = 0
                            total_return = 0
                            win_rate = 0

                        # Display yearly metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Signals", total_signals)
                        with col2:
                            st.metric("Completed Trades", completed_count)
                        with col3:
                            st.metric("Open Positions", open_positions)
                        with col4:
                            if completed_count > 0:
                                return_color = (
                                    "normal" if avg_return >= 0 else "inverse"
                                )
                                st.metric(
                                    "Avg Return",
                                    f"{avg_return:.2f}%",
                                    delta_color=return_color,
                                )
                            else:
                                st.metric("Avg Return", "N/A")
                        with col5:
                            if completed_count > 0:
                                total_color = (
                                    "normal" if total_return >= 0 else "inverse"
                                )
                                st.metric(
                                    "Total Return",
                                    f"{total_return:.2f}%",
                                    delta_color=total_color,
                                )
                            else:
                                st.metric("Total Return", "N/A")

                        # Color coding for crypto yearly analysis
                        def color_crypto_returns(val):
                            if val == "":
                                return ""
                            if float(val) > 0:
                                return "background-color: rgba(0, 255, 0, 0.2); color: green; font-weight: bold"
                            elif float(val) < 0:
                                return "background-color: rgba(255, 0, 0, 0.2); color: red; font-weight: bold"
                            return ""

                        def color_crypto_signals(val):
                            if "STRONG BUY" in str(val) or "BUY" in str(val):
                                return "color: green; font-weight: bold"
                            elif "STRONG SELL" in str(val) or "SELL" in str(val):
                                return "color: red; font-weight: bold"
                            return ""

                        # Apply styling to yearly crypto dataframe
                        if completed_count > 0 or open_positions > 0:
                            styled_yearly_crypto_df = (
                                trades_df.style.applymap(
                                    color_crypto_returns, subset=["Price Change (%)"]
                                )
                                .applymap(
                                    color_crypto_signals,
                                    subset=["Buy Signal", "Sell Signal"],
                                )
                                .format(
                                    {
                                        "Buy Price": "${:.4f}",
                                        "Sell Price": lambda x: (
                                            "${:.4f}".format(x) if x != "" else ""
                                        ),
                                        "Price Change (%)": lambda x: (
                                            "{:+.2f}%".format(x) if x != "" else ""
                                        ),
                                    }
                                )
                            )
                        else:
                            styled_yearly_crypto_df = trades_df.style

                        st.dataframe(
                            styled_yearly_crypto_df,
                            use_container_width=True,
                            column_config={
                                "Buy Date": st.column_config.DateColumn("Buy Date"),
                                "Buy Price": st.column_config.NumberColumn(
                                    "Buy Price ($)", format="%.4f"
                                ),
                                "Buy Signal": st.column_config.TextColumn("Buy Signal"),
                                "Sell Date": st.column_config.TextColumn("Sell Date"),
                                "Sell Price": st.column_config.TextColumn(
                                    "Sell Price ($)"
                                ),
                                "Sell Signal": st.column_config.TextColumn(
                                    "Sell Signal"
                                ),
                                "Price Change (%)": st.column_config.TextColumn(
                                    "Return (%)"
                                ),
                                "Days Held": st.column_config.TextColumn("Days Held"),
                            },
                        )

                        # Show year summary if there are completed trades
                        if completed_count > 0:
                            if win_rate >= 50:
                                st.success(
                                    f"🎉 {year_data['Year']}: {win_rate:.1f}% win rate with {total_return:+.2f}% total return"
                                )
                            else:
                                st.error(
                                    f"⚠️ {year_data['Year']}: {win_rate:.1f}% win rate with {total_return:+.2f}% total return"
                                )
                        elif open_positions > 0:
                            st.info(
                                f"📋 {year_data['Year']}: {open_positions} open position(s), no completed trades"
                            )
                        else:
                            st.info(f"📋 {year_data['Year']}: No trading activity")
            else:
                st.info(
                    "📋 No trading signals found in the last 6 years for this cryptocurrency"
                )
        else:
            st.warning("⚠️ No signals available for yearly analysis")

    else:
        st.warning("⚠️ No signals available for trading analysis")

    # Footer with disclaimer
    st.markdown("---")
    st.markdown(
        """
    **⚠️ Disclaimer:** This analysis is for educational purposes only and should not be considered as financial advice. 
    Cryptocurrency trading involves substantial risk of loss. Always do your own research and consider consulting 
    with a qualified financial advisor before making investment decisions.
    
    **📊 Technical Indicators Used:**
    - **MACD**: Moving Average Convergence Divergence
    - **RSI**: Relative Strength Index (14-period)
    - **SMA**: Simple Moving Averages (20, 50, 200)
    - **Bollinger Bands**: 20-period with 2 standard deviations
    - **Volatility**: 20-day rolling standard deviation (annualized)
    """
    )

    # Performance stats in footer
    if st.session_state.crypto_data_loaded:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(
                f"**📈 Cryptos Loaded:** {len(st.session_state.crypto_data_loaded)}"
            )

        with col2:
            if available_cryptos:
                total_datapoints = sum(
                    [
                        len(st.session_state.crypto_analyzer.data[crypto])
                        for crypto in available_cryptos
                    ]
                )
                st.info(f"**📊 Total Data Points:** {total_datapoints:,}")

        with col3:
            st.info(f"**📅 Analysis Period:** {start_date} to {end_date}")


# Additional helper functions that might be missing


def handle_timezone_comparison(df_datetime_index, start_dt, end_dt):
    """
    Handle timezone compatibility between DataFrame datetime index and comparison datetimes
    Returns timezone-compatible start and end datetimes
    """
    # Convert pandas datetime index to simple datetime objects if needed
    if hasattr(df_datetime_index, "to_pydatetime"):
        sample_dt = df_datetime_index[0] if len(df_datetime_index) > 0 else None
    else:
        sample_dt = df_datetime_index.iloc[0] if len(df_datetime_index) > 0 else None

    if sample_dt is None:
        return start_dt, end_dt

    # Check timezone compatibility
    if hasattr(sample_dt, "tzinfo") and sample_dt.tzinfo is not None:
        # DataFrame has timezone-aware datetimes
        if hasattr(start_dt, "tzinfo") and start_dt.tzinfo is None:
            # Convert naive datetimes to UTC (most common case)
            import pytz

            try:
                start_dt = pytz.UTC.localize(start_dt)
                end_dt = pytz.UTC.localize(end_dt)
            except:
                # If timezone localization fails, convert DataFrame to naive
                pass
    else:
        # DataFrame has timezone-naive datetimes
        if hasattr(start_dt, "tzinfo") and start_dt.tzinfo is not None:
            # Convert timezone-aware to naive
            start_dt = start_dt.replace(tzinfo=None)
            end_dt = end_dt.replace(tzinfo=None)

    return start_dt, end_dt


def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num/1_000:.2f}K"
    else:
        return f"${num:.2f}"


def get_signal_emoji(signal):
    """Get emoji for trading signals"""
    signal_emojis = {
        "STRONG BUY": "🟢🚀",
        "BUY": "🟢",
        "HOLD": "🟡",
        "SELL": "🔴",
        "STRONG SELL": "🔴💥",
    }
    return signal_emojis.get(signal, "⚪")


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
