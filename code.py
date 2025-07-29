import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RangeSlider
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


class StockChartsWithScroll:
    def __init__(self, tickers=["MSFT", "AAPL", "AMZN"], period="6y"):
        self.tickers = tickers
        self.colors = ["blue", "green", "red", "orange", "purple"]
        self.data = {}
        self.fig = None
        self.axes = []
        self.slider = None

        # Download data
        self.fetch_data(period)
        self.calculate_indicators()
        self.create_charts()

    def fetch_data(self, period):
        """Fetch stock data for all tickers"""
        print("Fetching stock data...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                self.data[ticker] = stock.history(period=period)
                print(f"Downloaded {ticker} data: {len(self.data[ticker])} days")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")

    def calculate_indicators(self):
        """Calculate MACD and RSI for all tickers"""
        for ticker in self.tickers:
            if ticker in self.data:
                df = self.data[ticker]

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

                self.data[ticker] = df

    def create_charts(self):
        """Create the main chart interface"""
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 12))
        self.fig.suptitle("Stock Analysis Dashboard", fontsize=16, fontweight="bold")

        # Adjust layout to make room for slider
        plt.subplots_adjust(bottom=0.15, hspace=0.3)

        # Get date range for slider
        all_dates = []
        for ticker in self.tickers:
            if ticker in self.data:
                all_dates.extend(self.data[ticker].index.tolist())

        if not all_dates:
            print("No data available!")
            return

        self.min_date = min(all_dates)
        self.max_date = max(all_dates)

        # Create slider
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.03])
        self.slider = RangeSlider(
            slider_ax,
            "Time Range",
            mdates.date2num(self.min_date),
            mdates.date2num(self.max_date),
            valinit=(mdates.date2num(self.min_date), mdates.date2num(self.max_date)),
            valfmt="%s",
        )

        # Connect slider to update function
        self.slider.on_changed(self.update_charts)

        # Initial plot
        self.plot_charts(self.min_date, self.max_date)

        plt.show()

    def plot_charts(self, start_date, end_date):
        """Plot all three charts for the given date range"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Chart 1: Stock Prices
        ax1 = self.axes[0]
        ax1.set_title("Stock Prices", fontweight="bold", pad=20)
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, alpha=0.3)

        # Chart 2: MACD
        ax2 = self.axes[1]
        ax2.set_title("MACD Indicator", fontweight="bold", pad=20)
        ax2.set_ylabel("MACD")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Chart 3: RSI
        ax3 = self.axes[2]
        ax3.set_title("RSI Indicator", fontweight="bold", pad=20)
        ax3.set_ylabel("RSI")
        ax3.set_xlabel("Date")
        ax3.grid(True, alpha=0.3)
        ax3.axhline(
            y=70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)"
        )
        ax3.axhline(
            y=30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)"
        )
        ax3.set_ylim(0, 100)

        # Plot data for each ticker
        for i, ticker in enumerate(self.tickers):
            if ticker not in self.data:
                continue

            df = self.data[ticker]
            color = self.colors[i % len(self.colors)]

            # Filter data by date range
            mask = (df.index >= start_date) & (df.index <= end_date)
            filtered_df = df.loc[mask]

            if filtered_df.empty:
                continue

            # Plot stock price
            ax1.plot(
                filtered_df.index,
                filtered_df["Close"],
                color=color,
                linewidth=2,
                label=ticker,
            )

            # Plot MACD
            ax2.plot(
                filtered_df.index,
                filtered_df["MACD"],
                color=color,
                linewidth=2,
                label=f"{ticker} MACD",
            )
            ax2.plot(
                filtered_df.index,
                filtered_df["MACD_Signal"],
                color=color,
                linewidth=1,
                linestyle="--",
                alpha=0.7,
            )

            # Plot RSI
            ax3.plot(
                filtered_df.index,
                filtered_df["RSI"],
                color=color,
                linewidth=2,
                label=f"{ticker} RSI",
            )

        # Format x-axis
        for ax in self.axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            ax.legend(loc="upper left")

        # Adjust layout
        plt.tight_layout()
        self.fig.canvas.draw()

    def update_charts(self, val):
        """Update charts based on slider values"""
        start_num, end_num = self.slider.val
        start_date = mdates.num2date(start_num).date()
        end_date = mdates.num2date(end_num).date()

        self.plot_charts(pd.Timestamp(start_date), pd.Timestamp(end_date))

    def generate_trading_signals(self, ticker):
        """Generate buy/sell signals based on MACD and RSI"""
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

            # Buy Signal Rules
            # 1. MACD crosses above signal line AND RSI is oversold (< 30)
            if prev_macd <= prev_signal and curr_macd > curr_signal and curr_rsi < 35:
                signal = "BUY"
                reason = "MACD bullish cross + RSI oversold"

            # 2. MACD crosses above signal line AND RSI is neutral (30-70)
            elif (
                prev_macd <= prev_signal
                and curr_macd > curr_signal
                and 30 <= curr_rsi <= 70
            ):
                signal = "BUY"
                reason = "MACD bullish cross + RSI neutral"

            # Sell Signal Rules
            # 1. MACD crosses below signal line AND RSI is overbought (> 70)
            elif prev_macd >= prev_signal and curr_macd < curr_signal and curr_rsi > 65:
                signal = "SELL"
                reason = "MACD bearish cross + RSI overbought"

            # 2. MACD crosses below signal line AND RSI is neutral
            elif (
                prev_macd >= prev_signal
                and curr_macd < curr_signal
                and 30 <= curr_rsi <= 70
            ):
                signal = "SELL"
                reason = "MACD bearish cross + RSI neutral"

            # 3. RSI extremely overbought (> 80) regardless of MACD
            elif curr_rsi > 80:
                signal = "SELL"
                reason = "RSI extremely overbought"

            signals.append(
                {
                    "Date": df.index[i],
                    "Price": df["Close"].iloc[i],
                    "Signal": signal,
                    "Reason": reason,
                    "MACD": curr_macd,
                    "MACD_Signal": curr_signal,
                    "RSI": curr_rsi,
                }
            )

        return pd.DataFrame(signals)

    def backtest_strategy(self, ticker, initial_capital=10000):
        """Backtest the trading strategy for a specific ticker"""
        signals_df = self.generate_trading_signals(ticker)

        if signals_df.empty:
            return None

        # Filter only actual buy/sell signals
        trades = signals_df[signals_df["Signal"].notna()].copy()

        if trades.empty:
            return None

        # Initialize tracking variables
        capital = initial_capital
        shares = 0
        trades_executed = []
        portfolio_value = []

        for _, trade in trades.iterrows():
            if trade["Signal"] == "BUY" and shares == 0:  # Buy only if not holding
                shares = capital / trade["Price"]
                capital = 0
                trades_executed.append(
                    {
                        "Date": trade["Date"],
                        "Action": "BUY",
                        "Price": trade["Price"],
                        "Shares": shares,
                        "Reason": trade["Reason"],
                        "Portfolio_Value": shares * trade["Price"],
                    }
                )

            elif trade["Signal"] == "SELL" and shares > 0:  # Sell only if holding
                capital = shares * trade["Price"]
                portfolio_value.append(capital)
                trades_executed.append(
                    {
                        "Date": trade["Date"],
                        "Action": "SELL",
                        "Price": trade["Price"],
                        "Shares": shares,
                        "Reason": trade["Reason"],
                        "Portfolio_Value": capital,
                    }
                )
                shares = 0

        # If still holding shares at the end, calculate final value
        if shares > 0:
            final_price = self.data[ticker]["Close"].iloc[-1]
            final_value = shares * final_price
        else:
            final_value = capital

        # Calculate performance metrics
        total_return = final_value - initial_capital
        return_percentage = (total_return / initial_capital) * 100

        # Calculate buy and hold performance for comparison
        start_price = self.data[ticker]["Close"].iloc[0]
        end_price = self.data[ticker]["Close"].iloc[-1]
        buy_hold_return = ((end_price - start_price) / start_price) * 100

        return {
            "ticker": ticker,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "return_percentage": return_percentage,
            "buy_hold_return": buy_hold_return,
            "num_trades": len(trades_executed),
            "trades": trades_executed,
        }

    def run_trading_analysis(self, initial_capital=10000):
        """Run trading analysis for all tickers"""
        print("\n" + "=" * 80)
        print(f"TRADING STRATEGY ANALYSIS (Initial Capital: ${initial_capital:,.2f})")
        print("=" * 80)
        print("\nStrategy Rules:")
        print("BUY Signals:")
        print("  1. MACD crosses above signal line + RSI < 35 (oversold)")
        print("  2. MACD crosses above signal line + RSI 30-70 (neutral)")
        print("\nSELL Signals:")
        print("  1. MACD crosses below signal line + RSI > 65 (overbought)")
        print("  2. MACD crosses below signal line + RSI 30-70 (neutral)")
        print("  3. RSI > 80 (extremely overbought)")
        print("\n" + "-" * 80)

        all_results = {}

        for ticker in self.tickers:
            if ticker not in self.data:
                continue

            result = self.backtest_strategy(ticker, initial_capital)
            if result is None:
                print(f"\n{ticker}: No trading signals generated")
                continue

            all_results[ticker] = result

            print(f"\n{ticker} PERFORMANCE:")
            print(f"  Initial Capital:     ${result['initial_capital']:,.2f}")
            print(f"  Final Value:         ${result['final_value']:,.2f}")
            print(f"  Total Return:        ${result['total_return']:,.2f}")
            print(f"  Return Percentage:   {result['return_percentage']:.2f}%")
            print(f"  Buy & Hold Return:   {result['buy_hold_return']:.2f}%")
            print(
                f"  Strategy vs B&H:     {result['return_percentage'] - result['buy_hold_return']:+.2f}%"
            )
            print(f"  Number of Trades:    {result['num_trades']}")

            # Show recent trades
            if result["trades"]:
                print(f"\n  Recent Trades:")
                recent_trades = result["trades"][-3:]  # Last 3 trades
                for trade in recent_trades:
                    print(
                        f"    {trade['Date'].strftime('%Y-%m-%d')}: {trade['Action']} at ${trade['Price']:.2f}"
                    )
                    print(f"      Reason: {trade['Reason']}")

        # Summary
        if all_results:
            print(f"\n" + "=" * 80)
            print("PORTFOLIO SUMMARY")
            print("=" * 80)

            total_strategy_value = sum(r["final_value"] for r in all_results.values())
            total_initial = len(all_results) * initial_capital
            total_return = total_strategy_value - total_initial
            total_return_pct = (total_return / total_initial) * 100

            print(f"Total Initial Capital:   ${total_initial:,.2f}")
            print(f"Total Final Value:       ${total_strategy_value:,.2f}")
            print(f"Total Return:            ${total_return:,.2f}")
            print(f"Total Return Percentage: {total_return_pct:.2f}%")

            # Best and worst performers
            best_performer = max(
                all_results.items(), key=lambda x: x[1]["return_percentage"]
            )
            worst_performer = min(
                all_results.items(), key=lambda x: x[1]["return_percentage"]
            )

            print(
                f"\nBest Performer:  {best_performer[0]} ({best_performer[1]['return_percentage']:.2f}%)"
            )
            print(
                f"Worst Performer: {worst_performer[0]} ({worst_performer[1]['return_percentage']:.2f}%)"
            )

        return all_results


def main():
    """Main function to run the application"""
    print("Starting Stock Charts Application...")
    print("This will download stock data and create interactive charts.")
    print("Use the slider at the bottom to adjust the time range.")

    # Create the application
    try:
        app = StockChartsWithScroll(tickers=["AVGO"], period="3y")

        # Run trading analysis
        print("\nRunning trading strategy analysis...")
        trading_results = app.run_trading_analysis(initial_capital=10000)

        # You can also run individual ticker analysis
        print("\n" + "=" * 80)
        print("INDIVIDUAL TICKER DETAILED ANALYSIS")
        print("=" * 80)

        # Example: Get detailed signals for AAPL
        if "AAPL" in app.data:
            signals = app.generate_trading_signals("AAPL")
            recent_signals = signals[signals["Signal"].notna()].tail(5)

            if not recent_signals.empty:
                print("\nRecent AAPL Trading Signals:")
                for _, signal in recent_signals.iterrows():
                    print(
                        f"  {signal['Date'].strftime('%Y-%m-%d')}: {signal['Signal']} at ${signal['Price']:.2f}"
                    )
                    print(
                        f"    Reason: {signal['Reason']} (RSI: {signal['RSI']:.1f}, MACD: {signal['MACD']:.3f})"
                    )

    except Exception as e:
        print(f"Error creating application: {e}")
        print("Make sure you have internet connection and required packages installed:")
        print("pip install matplotlib pandas numpy yfinance")


if __name__ == "__main__":
    main()
