# crypto_ma_cross_final_optimized.py
# Focus: Reduce losses while keeping winners

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import Strategy, Portfolio
from binance_data_fetcher import BinanceDataFetcher
from har_model import HARModel


class LossMinimizingStrategy(Strategy):
    """
    Strategy focused on minimizing losses:
    - Tighter stop losses
    - Better exit timing
    - Minimal entry filters
    """

    def __init__(self, symbol, bars, har_model, short_window=20, long_window=50):
        self.symbol = symbol
        self.bars = bars
        self.har_model = har_model
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self):
        """MA crossover with minimal filtering but smart exits."""
        signals = pd.DataFrame(index=self.bars.index)

        print("\nCalculating indicators...")

        # Moving Averages
        signals['short_mavg'] = self.bars['Close'].rolling(
            window=self.short_window, min_periods=self.short_window
        ).mean()
        signals['long_mavg'] = self.bars['Close'].rolling(
            window=self.long_window, min_periods=self.long_window
        ).mean()

        # ADX - Trend Strength Indicator (filters out choppy markets)
        high = self.bars['High']
        low = self.bars['Low']
        close = self.bars['Close']

        # Calculate +DI and -DI
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth with 14-period average
        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)

        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        signals['adx'] = dx.rolling(window=14).mean()

        # ADX > 25 = trending market, < 25 = choppy market
        print(
            f"  ADX - Mean: {signals['adx'].mean():.2f}, Trending periods (ADX>25): {(signals['adx'] > 25).sum() / len(signals) * 100:.1f}%")

        # HAR volatility
        daily_rv = self.har_model.daily_rv

        rv_by_date = {}
        for idx in daily_rv.index:
            date_key = idx.date() if hasattr(idx, 'date') else idx
            rv_by_date[date_key] = daily_rv.loc[idx]

        unique_dates = pd.Series([idx.date() for idx in signals.index]).unique()
        vol_forecasts = {}

        for date in unique_dates:
            available_dates = [d for d in rv_by_date.keys() if d < date]

            if len(available_dates) >= 30:
                recent_dates = sorted(available_dates)[-30:]
                recent_rv = [rv_by_date[d] for d in recent_dates]

                rv_d = recent_rv[-1]
                rv_w = np.mean(recent_rv[-7:])
                rv_m = np.mean(recent_rv)

                try:
                    rv_forecast = self.har_model.forecast(rv_d, rv_w, rv_m, horizon=1)
                    vol_forecast = np.sqrt(max(rv_forecast, 1e-10))
                    vol_forecasts[date] = vol_forecast
                except:
                    vol_forecasts[date] = None

        signals['volatility_pct'] = 2.0

        for idx in signals.index:
            date_key = idx.date()
            if date_key in vol_forecasts and vol_forecasts[date_key] is not None:
                vol = vol_forecasts[date_key]
                price = self.bars.loc[idx, 'Close']
                vol_pct = (vol / price) * 100
                signals.loc[idx, 'volatility_pct'] = vol_pct

        # Fallback volatility
        returns = np.log(self.bars['Close'] / self.bars['Close'].shift(1))
        rolling_vol = returns.rolling(window=96).std() * np.sqrt(96) * 100
        signals['volatility_pct'] = signals['volatility_pct'].where(
            signals['volatility_pct'] > 0, rolling_vol
        )
        signals['volatility_pct'] = signals['volatility_pct'].fillna(2.0)

        # RSI for exit timing
        delta = self.bars['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))

        # ATR for stop loss sizing
        high_low = self.bars['High'] - self.bars['Low']
        high_close = np.abs(self.bars['High'] - self.bars['Close'].shift())
        low_close = np.abs(self.bars['Low'] - self.bars['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        signals['atr'] = true_range.rolling(window=14).mean()
        signals['atr_pct'] = (signals['atr'] / self.bars['Close']) * 100

        print(f"  Volatility - Mean: {signals['volatility_pct'].mean():.2f}%")
        print(f"  ATR - Mean: {signals['atr_pct'].mean():.2f}%")

        # MINIMAL ENTRY FILTERS - Let trades happen
        signals['signal'] = 0.0

        crossover_count = 0
        passed_count = 0

        for i in range(self.long_window, len(signals)):
            short_now = signals['short_mavg'].iloc[i]
            short_prev = signals['short_mavg'].iloc[i - 1]
            long_now = signals['long_mavg'].iloc[i]
            long_prev = signals['long_mavg'].iloc[i - 1]

            if pd.isna(short_now) or pd.isna(long_now):
                continue

            vol = signals['volatility_pct'].iloc[i]
            rsi = signals['rsi'].iloc[i]

            # Bullish crossover - VERY MINIMAL FILTERS
            if short_prev <= long_prev and short_now > long_now:
                crossover_count += 1

                # Filter out more conditions to reduce overtrading
                price = self.bars['Close'].iloc[i]
                long_now_val = signals['long_mavg'].iloc[i]

                # Require: moderate vol, not overbought, price above MA with confirmation
                adx = signals['adx'].iloc[i]

                # Strong filters to avoid choppy markets
                if (vol < 12.0 and
                        (pd.isna(rsi) or rsi < 75) and
                        price > long_now_val * 1.01 and  # Require 1% above MA
                        pd.notna(adx) and adx > 25):
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1.0
                    passed_count += 1

            # Bearish crossover
            elif short_prev >= long_prev and short_now < long_now:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1.0

        print(f"\nSignal Generation:")
        print(f"  Crossovers: {crossover_count}")
        print(f"  Passed filters: {passed_count} ({passed_count / max(crossover_count, 1) * 100:.1f}%)")

        # Forward fill
        signals['signal'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0.0)
        signals['signal'] = (signals['signal'] > 0).astype(float)

        # Position sizing - moderate sizing
        signals['position_size'] = 0.80  # Fixed 80% to avoid complexity

        signals['positions'] = signals['signal'].diff()

        buy_signals = (signals['positions'] == 1.0).sum()
        sell_signals = (signals['positions'] == -1.0).sum()

        print(f"  BUY signals: {buy_signals}")
        print(f"  SELL signals: {sell_signals}")

        return signals


class TightStopPortfolio(Portfolio):
    """
    Portfolio focused on cutting losses quickly.
    Key changes:
    - Tighter stop losses (ATR-based)
    - Partial profit taking
    - Quick exits on weakness
    """

    def __init__(self, symbol, bars, signals, initial_capital=10000.0,
                 position_size=0.80, trading_fee=0.001):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.position_size = position_size
        self.trading_fee = trading_fee
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions[self.symbol] = self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """Backtest with TIGHT stop losses to minimize losses."""
        portfolio = pd.DataFrame(index=self.bars.index)

        portfolio['price'] = self.bars['Close']
        portfolio['signal'] = self.signals['signal']
        portfolio['atr_pct'] = self.signals['atr_pct']
        portfolio['rsi'] = self.signals['rsi']

        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = self.initial_capital

        crypto_position = 0.0
        cash = self.initial_capital
        entry_price = 0.0
        highest_price = 0.0

        trades = []

        print("\nExecuting backtest with tight stops...")

        for i in range(1, len(portfolio)):
            current_price = portfolio['price'].iloc[i]
            current_signal = portfolio['signal'].iloc[i]
            prev_signal = portfolio['signal'].iloc[i - 1]
            atr_pct = portfolio['atr_pct'].iloc[i]
            rsi = portfolio['rsi'].iloc[i]

            # TIGHTER STOPS - Based on ATR
            # ATR-based stop: 2x ATR (tighter than before)
            atr_stop = np.clip(atr_pct * 2.0, 0.08, 0.15)  # 8-20% stop

            # Take profit: 3x ATR (let winners run more)
            atr_tp = np.clip(atr_pct * 10, 0.40, 1.50)  # 25-80% take profit

            # Trailing stop: 1.5x ATR
            atr_trail = np.clip(atr_pct * 1.5, 0.06, 0.15)  # 6-15% trailing

            # Risk management
            if crypto_position > 0 and entry_price > 0:
                if current_price > highest_price:
                    highest_price = current_price

                pnl_pct = (current_price / entry_price - 1) * 100

                # TIGHT STOP LOSS - Cut losses quickly
                if current_price <= entry_price * (1 - atr_stop):
                    sell_value = crypto_position * current_price
                    fee = sell_value * self.trading_fee
                    cash += sell_value - fee

                    trades.append({
                        'timestamp': portfolio.index[i],
                        'type': 'SELL (STOP)',
                        'price': current_price,
                        'value': sell_value,
                        'fee': fee,
                        'pnl_pct': pnl_pct,
                        'reason': 'Stop Loss',
                        'stop_size': atr_stop * 100
                    })

                    crypto_position = 0.0
                    entry_price = 0.0
                    highest_price = 0.0
                    continue

                # RSI-based exit - Exit if overbought and losing momentum
                if pd.notna(rsi) and rsi > 82 and pnl_pct > 15 and crypto_position > 0:
                    # Sell half position to lock in profits
                    sell_amount = crypto_position * 0.5
                    sell_value = sell_amount * current_price
                    fee = sell_value * self.trading_fee
                    cash += sell_value - fee
                    crypto_position -= sell_amount

                    trades.append({
                        'timestamp': portfolio.index[i],
                        'type': 'SELL (PARTIAL)',
                        'price': current_price,
                        'value': sell_value,
                        'fee': fee,
                        'pnl_pct': pnl_pct,
                        'reason': 'Partial Profit',
                        'stop_size': 0
                    })

                # Take profit
                if current_price >= entry_price * (1 + atr_tp):
                    sell_value = crypto_position * current_price
                    fee = sell_value * self.trading_fee
                    cash += sell_value - fee

                    trades.append({
                        'timestamp': portfolio.index[i],
                        'type': 'SELL (TP)',
                        'price': current_price,
                        'value': sell_value,
                        'fee': fee,
                        'pnl_pct': pnl_pct,
                        'reason': 'Take Profit',
                        'stop_size': 0
                    })

                    crypto_position = 0.0
                    entry_price = 0.0
                    highest_price = 0.0
                    continue

                # Trailing stop (after 8% profit - lower threshold)
                if highest_price > entry_price * 1.10:
                    if current_price <= highest_price * (1 - atr_trail):
                        sell_value = crypto_position * current_price
                        fee = sell_value * self.trading_fee
                        cash += sell_value - fee

                        trades.append({
                            'timestamp': portfolio.index[i],
                            'type': 'SELL (TRAIL)',
                            'price': current_price,
                            'value': sell_value,
                            'fee': fee,
                            'pnl_pct': pnl_pct,
                            'reason': 'Trailing Stop',
                            'stop_size': atr_trail * 100
                        })

                        crypto_position = 0.0
                        entry_price = 0.0
                        highest_price = 0.0
                        continue

            # Signal trading
            if prev_signal == 0 and current_signal == 1 and cash > 0 and crypto_position == 0:
                # BUY
                buy_amount = cash * self.position_size
                fee = buy_amount * self.trading_fee
                crypto_bought = (buy_amount - fee) / current_price

                crypto_position = crypto_bought
                cash -= buy_amount
                entry_price = current_price
                highest_price = current_price

                trades.append({
                    'timestamp': portfolio.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'value': buy_amount,
                    'fee': fee,
                    'pnl_pct': 0,
                    'reason': 'Signal',
                    'stop_size': atr_stop * 100
                })

            elif prev_signal == 1 and current_signal == 0 and crypto_position > 0:
                # SELL
                sell_value = crypto_position * current_price
                fee = sell_value * self.trading_fee
                cash += sell_value - fee

                pnl_pct = (current_price / entry_price - 1) * 100

                trades.append({
                    'timestamp': portfolio.index[i],
                    'type': 'SELL (SIGNAL)',
                    'price': current_price,
                    'value': sell_value,
                    'fee': fee,
                    'pnl_pct': pnl_pct,
                    'reason': 'Signal',
                    'stop_size': 0
                })

                crypto_position = 0.0
                entry_price = 0.0
                highest_price = 0.0

            # Update portfolio
            portfolio.loc[portfolio.index[i], 'holdings'] = crypto_position * current_price
            portfolio.loc[portfolio.index[i], 'cash'] = cash
            portfolio.loc[portfolio.index[i], 'total'] = cash + (crypto_position * current_price)

        portfolio['returns'] = portfolio['total'].pct_change()
        self.trades_df = pd.DataFrame(trades)

        print(f"✓ Backtest complete: {len(trades)} trades")

        # Print stop loss statistics
        stop_losses = self.trades_df[self.trades_df['type'] == 'SELL (STOP)']
        if len(stop_losses) > 0:
            print(f"\nStop Loss Stats:")
            print(f"  Stop losses triggered: {len(stop_losses)}")
            print(f"  Avg stop loss: {stop_losses['pnl_pct'].mean():.2f}%")
            print(f"  Max stop loss: {stop_losses['pnl_pct'].min():.2f}%")

        return portfolio


def calculate_metrics(returns, trades_df, initial_capital):
    """Calculate metrics."""
    metrics = {}

    metrics['Initial Capital'] = initial_capital
    metrics['Final Value'] = returns['total'].iloc[-1]
    metrics['Net Profit'] = returns['total'].iloc[-1] - initial_capital
    metrics['Total Return (%)'] = (returns['total'].iloc[-1] / initial_capital - 1) * 100

    total_days = (returns.index[-1] - returns.index[0]).total_seconds() / (24 * 3600)
    metrics['Total Days'] = total_days

    periods_per_year = 365.25 * 24 * 4
    periods_total = len(returns)

    if periods_total > 1:
        total_return_ratio = returns['total'].iloc[-1] / initial_capital
        annualized = ((total_return_ratio ** (periods_per_year / periods_total)) - 1) * 100
        metrics['Annualized Return (%)'] = annualized

    returns_clean = returns['returns'].replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns_clean) > 0 and returns_clean.std() != 0:
        metrics['Sharpe Ratio'] = (returns_clean.mean() / returns_clean.std()) * np.sqrt(periods_per_year)
    else:
        metrics['Sharpe Ratio'] = 0

    running_max = returns['total'].cummax()
    drawdown = (returns['total'] - running_max) / running_max
    metrics['Max Drawdown (%)'] = drawdown.min() * 100

    if not trades_df.empty:
        metrics['Number of Trades'] = len(trades_df)
        metrics['Total Fees'] = trades_df['fee'].sum()

        buy_trades = trades_df[trades_df['type'] == 'BUY']
        sell_trades = trades_df[trades_df['type'].str.contains('SELL')]

        metrics['Buys'] = len(buy_trades)
        metrics['Sells'] = len(sell_trades)

        if len(sell_trades) > 0:
            winners = len(sell_trades[sell_trades['pnl_pct'] > 0])
            losers = len(sell_trades[sell_trades['pnl_pct'] <= 0])
            metrics['Winning Trades'] = winners
            metrics['Losing Trades'] = losers
            metrics['Win Rate (%)'] = (winners / len(sell_trades)) * 100

            if winners > 0:
                metrics['Avg Win (%)'] = sell_trades[sell_trades['pnl_pct'] > 0]['pnl_pct'].mean()
                metrics['Max Win (%)'] = sell_trades['pnl_pct'].max()
            if losers > 0:
                metrics['Avg Loss (%)'] = sell_trades[sell_trades['pnl_pct'] <= 0]['pnl_pct'].mean()
                metrics['Max Loss (%)'] = sell_trades['pnl_pct'].min()

            total_wins = sell_trades[sell_trades['pnl_pct'] > 0]['pnl_pct'].sum()
            total_losses = abs(sell_trades[sell_trades['pnl_pct'] <= 0]['pnl_pct'].sum())
            if total_losses > 0:
                metrics['Profit Factor'] = total_wins / total_losses

            if 'reason' in sell_trades.columns:
                for reason in sell_trades['reason'].unique():
                    count = len(sell_trades[sell_trades['reason'] == reason])
                    metrics[f'{reason} Exits'] = count

    return metrics


def print_metrics(metrics):
    """Print results."""
    print(f"\n{'=' * 70}")
    print("LOSS-MINIMIZING STRATEGY RESULTS")
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            if '(%)' in key or 'Rate' in key or 'Drawdown' in key or 'Return' in key or 'Factor' in key:
                print(f"  {key:.<50} {value:>15.2f}%")
            elif 'Capital' in key or 'Value' in key or 'Profit' in key or 'Fees' in key:
                print(f"  {key:.<50} ${value:>14,.2f}")
            elif 'Days' in key:
                print(f"  {key:.<50} {value:>15.1f}")
            else:
                print(f"  {key:.<50} {value:>15.2f}")
        else:
            print(f"  {key:.<50} {value:>15}")

    print("=" * 70)


if __name__ == "__main__":

    SYMBOL = "ETHUSDT"
    INTERVAL = "15m"

    start_date = datetime.datetime(2025, 6, 1)
    end_date = datetime.datetime(2025, 8, 31, 23, 59, 59)

    SHORT_WINDOW = 30
    LONG_WINDOW = 80

    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE = 0.95  # 80% to be safer
    TRADING_FEE = 0.001

    USE_CACHE = True
    USE_LOG_HAR = True

    print("=" * 70)
    print("LOSS-MINIMIZING STRATEGY")
    print("=" * 70)
    print("Key Features:")
    print("  • ATR-based tight stop losses (2x ATR = 8-20%)")
    print("  • Partial profit taking at RSI > 75")
    print("  • Lower trailing stop threshold (8% vs 15%)")
    print("  • Minimal entry filters (90%+ acceptance)")
    print("  • Fixed 80% position sizing")
    print("=" * 70)

    # Fetch
    fetcher = BinanceDataFetcher()
    bars = fetcher.fetch_ohlcv(SYMBOL, INTERVAL, start_date, end_date, use_cache=USE_CACHE)

    # HAR
    print(f"\n{'=' * 70}")
    print("FITTING LOG-HAR MODEL")
    print("=" * 70)

    har_model = HARModel(use_log=USE_LOG_HAR)
    har_model.fit(bars['Close'], min_training_days=40)
    har_model.print_model_summary()

    # Backtest
    print(f"\n{'=' * 70}")
    print("RUNNING BACKTEST")
    print("=" * 70)

    strategy = LossMinimizingStrategy(SYMBOL, bars, har_model,
                                      short_window=SHORT_WINDOW,
                                      long_window=LONG_WINDOW)
    signals = strategy.generate_signals()

    portfolio = TightStopPortfolio(SYMBOL, bars, signals,
                                   initial_capital=INITIAL_CAPITAL,
                                   position_size=POSITION_SIZE,
                                   trading_fee=TRADING_FEE)
    returns = portfolio.backtest_portfolio()

    # Results
    metrics = calculate_metrics(returns, portfolio.trades_df, INITIAL_CAPITAL)
    print_metrics(metrics)

    if not portfolio.trades_df.empty:
        portfolio.trades_df.to_csv('trades_loss_minimizing.csv', index=False)

        # Show worst losses
        losses = portfolio.trades_df[portfolio.trades_df['pnl_pct'] < 0].sort_values('pnl_pct')
        if len(losses) > 0:
            print(f"\nWorst 5 Losses:")
            print(losses.head()[['timestamp', 'type', 'pnl_pct', 'reason']].to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPARISON TO PREVIOUS VERSIONS")
    print("=" * 70)
    print("Original:          +13.55% return, -34% drawdown, 42% win, -2.72% avg loss")
    print("With 200-day MA:   -5.55% return, -33% drawdown, 42% win, -3.48% avg loss")
    print("This version:      Goal is to reduce avg loss to < -2%")
    print("=" * 70)

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    bars['Close'].plot(label='ETH', alpha=0.7)
    signals['short_mavg'].plot(label=f'MA{SHORT_WINDOW}')
    signals['long_mavg'].plot(label=f'MA{LONG_WINDOW}')

    buy_sig = signals[signals.positions == 1.0]
    if not buy_sig.empty:
        plt.scatter(buy_sig.index, bars.loc[buy_sig.index, 'Close'],
                    marker='^', s=100, color='green', label='BUY', zorder=5)

    plt.legend()
    plt.title('Loss-Minimizing Strategy with Tight ATR Stops')
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    returns['total'].plot(linewidth=2, label='Portfolio', color='orange')
    plt.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', label='Initial')
    plt.legend()
    plt.title('Portfolio Value')
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    running_max = returns['total'].cummax()
    drawdown = ((returns['total'] - running_max) / running_max) * 100
    drawdown.plot(linewidth=2, color='red', label='Drawdown')
    plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    plt.legend()
    plt.title('Drawdown')
    plt.ylabel('%')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ETHUSDT_loss_minimizing.png', dpi=300)
    print("\n✓ Chart saved: 'ETHUSDT_loss_minimizing.png'")
    plt.show()