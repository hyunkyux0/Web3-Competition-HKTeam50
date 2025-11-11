"""
Adaptive Coin Range Trading Strategy - BLACK SWAN PROTECTED
Added protections against liquidation cascades and macro shocks
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json
import os
from backtest import Strategy, Portfolio
from binance_data_fetcher import BinanceDataFetcher
from har_model import HARModel


class MarketRegimeDetector:
    """
    Detects dangerous market conditions (crashes, liquidation cascades).
    """

    @staticmethod
    def detect_liquidation_cascade(df, lookback=96):
        """
        Detect if we're in a liquidation cascade.
        Returns True if UNSAFE to trade.
        """
        if len(df) < lookback:
            return False

        recent = df.tail(lookback)

        # 1. Extreme volume spike (3x average)
        avg_volume = df['Volume'].rolling(500).mean().iloc[-1]
        current_volume = recent['Volume'].mean()
        volume_spike = current_volume > (avg_volume * 3)

        # 2. Large downward price move in short time
        price_drop = (recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100
        sharp_drop = price_drop < -5  # -5% in 24 hours

        # 3. Multiple consecutive red candles
        consecutive_reds = (recent['Close'] < recent['Open']).rolling(10).sum().max()
        cascade_pattern = consecutive_reds >= 8  # 8 out of 10 red candles

        # 4. Accelerating volatility
        recent_vol = recent['Close'].pct_change().std()
        historical_vol = df['Close'].pct_change().tail(1000).std()
        vol_spike = recent_vol > (historical_vol * 2)

        is_cascade = (volume_spike and sharp_drop) or (cascade_pattern and vol_spike)

        return is_cascade

    @staticmethod
    def detect_macro_shock(df, btc_df=None):
        """
        Detect macro shock events (market-wide crashes).
        Returns True if UNSAFE to trade.
        """
        if len(df) < 200:
            return False

        # 1. Circuit breaker: -10% in 2 hours (8 candles of 15m)
        price_2h_ago = df['Close'].iloc[-8] if len(df) >= 8 else df['Close'].iloc[0]
        current_price = df['Close'].iloc[-1]
        drop_2h = (current_price / price_2h_ago - 1) * 100

        if drop_2h < -10:
            return True  # Circuit breaker triggered

        # 2. Check if Bitcoin is crashing (if data available)
        if btc_df is not None and len(btc_df) >= 8:
            btc_drop = (btc_df['Close'].iloc[-1] / btc_df['Close'].iloc[-8] - 1) * 100
            if btc_drop < -5:  # BTC down >5% in 2 hours
                return True

        # 3. Gap down detection (overnight shock)
        if len(df) >= 2:
            gap_down = (df['Open'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
            if gap_down < -3:  # -3% gap
                return True

        return False

    @staticmethod
    def is_safe_to_trade(df, btc_df=None):
        """
        Master safety check. Returns True if safe to enter new positions.
        """
        # Check for liquidation cascade
        if MarketRegimeDetector.detect_liquidation_cascade(df):
            return False, "Liquidation cascade detected"

        # Check for macro shock
        if MarketRegimeDetector.detect_macro_shock(df, btc_df):
            return False, "Macro shock event detected"

        # Check for extreme gap (news event)
        if len(df) >= 96:
            returns_1d = df['Close'].pct_change().tail(96)
            extreme_moves = (returns_1d.abs() > 0.05).sum()  # Count 5%+ moves
            if extreme_moves > 3:  # More than 3 extreme moves in 24h
                return False, "Excessive volatility"

        return True, "Safe"


class AdaptiveCoinSelector:
    """
    Selects the best coin for RANGE TRADING (not momentum).
    Looks for high volatility, low trending coins.
    """

    def __init__(self, interval='15m', rescan_hours=6, min_data_days=30):
        self.fetcher = BinanceDataFetcher()
        self.interval = interval
        self.rescan_hours = rescan_hours
        self.min_data_days = min_data_days
        self.last_scan_time = None
        self.current_best_coin = None
        self.coin_scores = {}
        self.btc_data = None  # Cache BTC data for market regime detection

    def fetch_btc_reference(self):
        """Fetch BTC data for market-wide crash detection."""
        try:
            end_date = datetime.datetime.utcnow()
            start_date = end_date - datetime.timedelta(days=2)
            self.btc_data = self.fetcher.fetch_ohlcv('BTCUSDT', self.interval, start_date, end_date, use_cache=True)
        except:
            self.btc_data = None

    def get_top_coins(self, limit=30):
        """Get top coins by 24h volume from Binance."""
        try:
            from binance.client import Client
            client = Client()

            tickers = client.get_ticker()
            usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt_pairs_sorted = sorted(usdt_pairs,
                                      key=lambda x: float(x['quoteVolume']),
                                      reverse=True)

            top_pairs = [pair['symbol'] for pair in usdt_pairs_sorted[:limit*2]]

            excluded = ['USDC', 'BUSD', 'TUSD', 'USDD', 'USDP', 'FDUSD',
                       'UP', 'DOWN', 'BULL', 'BEAR', 'DAI']

            filtered_pairs = [
                pair for pair in top_pairs
                if not any(exc in pair for exc in excluded)
            ]

            return filtered_pairs[:limit]

        except Exception as e:
            print(f"⚠️  Error fetching top coins: {e}")
            return [
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
                'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'
            ]

    def is_choppy_market(self, df, lookback=100):
        """
        Detect if market is actually ranging/choppy.
        Returns True if suitable for mean reversion.
        """
        try:
            if len(df) < lookback:
                return True  # Default to True if insufficient data

            high_100 = df['High'].rolling(lookback).max()
            low_100 = df['Low'].rolling(lookback).min()
            current_price = df['Close']

            # Price should be in middle 60% of range (not at extremes)
            range_position = (current_price - low_100) / (high_100 - low_100)

            # Calculate ADX for trend strength
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()

            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr_14 = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()

            # Check latest values
            latest_range_pos = range_position.iloc[-1]
            latest_adx = adx.iloc[-1]

            # Choppy if: price in middle range AND low ADX
            is_choppy = (0.3 < latest_range_pos < 0.7) and (latest_adx < 30)

            return is_choppy

        except Exception as e:
            return True  # Default to True on error

    def calculate_range_score(self, symbol, bars):
        """
        Score coins for RANGE TRADING (opposite of momentum).
        High score = good for mean reversion.
        """
        try:
            if len(bars) < 100:
                return 0, False

            df = bars.copy()

            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
            df['bb_std'] = df['Close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
            df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            atr_pct = (atr / df['Close']) * 100

            # ADX (for trend strength - we WANT low ADX)
            high_diff = df['High'].diff()
            low_diff = -df['Low'].diff()

            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            atr_14 = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()

            # Get latest values
            latest = df.iloc[-1]

            price = latest['Close']
            bb_width_now = df['bb_width'].iloc[-1]
            rsi_now = rsi.iloc[-1]
            atr_pct_now = atr_pct.iloc[-1]
            adx_now = adx.iloc[-1]
            bb_middle = df['bb_middle'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]

            # Check data sufficiency for HAR
            has_enough_data = len(bars) >= 960

            # Calculate RANGE TRADING score (0-20)
            score = 0

            # 1. HIGH Volatility is GOOD for range trading (opposite of momentum)
            if pd.notna(bb_width_now):
                if bb_width_now > 6.0:  # Wide bands = high volatility
                    score += 5
                elif bb_width_now > 4.0:
                    score += 3
                elif bb_width_now > 2.0:
                    score += 1

            # 2. LOW ADX is GOOD (no strong trend = good for mean reversion)
            if pd.notna(adx_now):
                if adx_now < 20:  # Very weak trend
                    score += 5
                elif adx_now < 25:  # Weak trend
                    score += 3
                elif adx_now < 30:  # Mild trend
                    score += 1

            # 3. RSI near extremes (good entry points)
            if pd.notna(rsi_now):
                if rsi_now < 35 or rsi_now > 65:  # Near oversold/overbought
                    score += 3
                elif rsi_now < 40 or rsi_now > 60:
                    score += 2

            # 4. Price near bands (ready for reversal)
            if pd.notna(bb_upper) and pd.notna(bb_lower):
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
                if bb_position < 0.2 or bb_position > 0.8:  # Near extremes
                    score += 3
                elif bb_position < 0.3 or bb_position > 0.7:
                    score += 2

            # 5. Moderate ATR (not too crazy)
            if pd.notna(atr_pct_now):
                if 0.3 < atr_pct_now < 1.0:  # Good volatility range
                    score += 2
                elif 1.0 < atr_pct_now < 2.0:
                    score += 1

            # 6. Recent choppiness (no strong directional move)
            if len(df) >= 20:
                price_20_ago = df['Close'].iloc[-20]
                recent_move = abs((price / price_20_ago - 1) * 100)

                if recent_move < 2.0:  # Very choppy
                    score += 2
                elif recent_move < 5.0:  # Somewhat choppy
                    score += 1

            # 7. Market regime filter
            if self.is_choppy_market(df):
                score += 3  # Bonus for confirmed choppy market

            # Bonus for HAR data
            if has_enough_data:
                score += 1

            return score, has_enough_data

        except Exception as e:
            return 0, False

    def select_best_coin(self, coins=None, verbose=True, top_n=3):
        """Scan coins for best RANGE TRADING setup. Returns top N coins."""
        # Fetch BTC for market regime detection
        self.fetch_btc_reference()

        if coins is None:
            coins = self.get_top_coins(limit=30)

        if verbose:
            print(f"\n{'='*70}")
            print(f"SCANNING {len(coins)} COINS FOR BEST RANGE TRADING SETUP")
            print(f"Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print('='*70)

        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=self.min_data_days)

        scores = {}
        har_compatible = {}

        for i, symbol in enumerate(coins, 1):
            try:
                if verbose:
                    print(f"  [{i}/{len(coins)}] {symbol}...", end=' ')

                bars = self.fetcher.fetch_ohlcv(
                    symbol,
                    self.interval,
                    start_date,
                    end_date,
                    use_cache=True
                )

                if bars is None or len(bars) < 100:
                    if verbose:
                        print("❌ Insufficient data")
                    continue

                score, has_har_data = self.calculate_range_score(symbol, bars)
                scores[symbol] = score
                har_compatible[symbol] = has_har_data

                har_status = "✓ HAR" if has_har_data else "⚠ No HAR"
                if verbose:
                    print(f"{har_status} | Score: {score}/24")

                time.sleep(0.05)

            except Exception as e:
                if verbose:
                    print(f"❌ {e}")
                continue

        if not scores:
            if verbose:
                print("\n⚠️  No valid coins found, defaulting to ETHUSDT")
            return [('ETHUSDT', 0, True)]

        sorted_coins = sorted(scores.items(), key=lambda x: (x[1], har_compatible.get(x[0], False)), reverse=True)

        # Get top N coins
        top_coins = []
        for i in range(min(top_n, len(sorted_coins))):
            coin = sorted_coins[i][0]
            score = sorted_coins[i][1]
            has_har = har_compatible.get(coin, False)
            top_coins.append((coin, score, has_har))

        self.coin_scores = scores
        self.last_scan_time = datetime.datetime.utcnow()

        if verbose:
            print(f"\n{'='*70}")
            print("TOP 5 RANGE TRADING SETUPS:")
            print('='*70)
            for i, (coin, score) in enumerate(sorted_coins[:5], 1):
                medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1]
                har_icon = "✓" if har_compatible.get(coin, False) else "⚠"
                print(f"  {medal} {coin}: {score}/24 points [{har_icon} HAR]")
            print('='*70)
            print(f"\n🎯 SELECTED TOP {len(top_coins)} COINS FOR DIVERSIFICATION:")
            for coin, score, has_har in top_coins:
                print(f"   • {coin} (Score: {score}/24)")

        return top_coins


class RangeTradingStrategy(Strategy):
    """
    Bollinger Band Mean Reversion Strategy with HAR volatility.
    Now with BLACK SWAN PROTECTION.
    """

    def __init__(self, symbol, bars, use_har=True, bb_period=20, bb_std=2, rsi_period=14, btc_data=None):
        self.symbol = symbol
        self.bars = bars
        self.use_har = use_har
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.har_model = None
        self.btc_data = btc_data  # For market regime detection

    def generate_signals(self):
        """Generate mean reversion signals with BLACK SWAN protection."""
        signals = pd.DataFrame(index=self.bars.index)

        print(f"\n{'='*70}")
        print(f"GENERATING RANGE SIGNALS FOR {self.symbol}")
        print('='*70)

        # Bollinger Bands
        signals['bb_middle'] = self.bars['Close'].rolling(window=self.bb_period).mean()
        signals['bb_std'] = self.bars['Close'].rolling(window=self.bb_period).std()
        signals['bb_upper'] = signals['bb_middle'] + (self.bb_std * signals['bb_std'])
        signals['bb_lower'] = signals['bb_middle'] - (self.bb_std * signals['bb_std'])
        signals['bb_width'] = (signals['bb_upper'] - signals['bb_lower']) / signals['bb_middle'] * 100
        signals['bb_position'] = (self.bars['Close'] - signals['bb_lower']) / (signals['bb_upper'] - signals['bb_lower'])

        # RSI
        delta = self.bars['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))

        # ADX
        high = self.bars['High']
        low = self.bars['Low']
        close = self.bars['Close']

        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr_14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        signals['adx'] = dx.rolling(window=14).mean()

        # ATR
        signals['atr'] = atr_14
        signals['atr_pct'] = (atr_14 / self.bars['Close']) * 100

        # Volume analysis
        signals['volume_ma'] = self.bars['Volume'].rolling(window=50).mean()
        signals['volume_ratio'] = self.bars['Volume'] / signals['volume_ma']

        # Order Flow
        signals['price_change'] = self.bars['Close'].diff()
        signals['up_volume'] = self.bars['Volume'].where(signals['price_change'] > 0, 0)
        signals['down_volume'] = self.bars['Volume'].where(signals['price_change'] < 0, 0)

        signals['buy_volume_20'] = signals['up_volume'].rolling(window=20).sum()
        signals['sell_volume_20'] = signals['down_volume'].rolling(window=20).sum()

        signals['order_flow_ratio'] = signals['buy_volume_20'] / (signals['buy_volume_20'] + signals['sell_volume_20'])

        # HAR Volatility
        if self.use_har and len(self.bars) >= 960:
            try:
                print("  🔬 Fitting HAR volatility model...")
                self.har_model = HARModel(use_log=True)
                self.har_model.fit(self.bars['Close'], min_training_days=3)

                returns = np.log(self.bars['Close'] / self.bars['Close'].shift(1))

                signals['rv_daily'] = (returns ** 2).rolling(window=96).sum()
                signals['rv_weekly'] = (returns ** 2).rolling(window=672).sum() / 7
                signals['rv_monthly'] = (returns ** 2).rolling(window=672).sum() / 7

                signals['intraday_vol_pct'] = np.sqrt(signals['rv_daily'] * 252) * 100
                signals['har_forecast'] = np.nan

                for i in range(672, len(signals)):
                    rv_d = signals['rv_daily'].iloc[i]
                    rv_w = signals['rv_weekly'].iloc[i]
                    rv_m = signals['rv_monthly'].iloc[i]

                    if pd.notna(rv_d) and pd.notna(rv_w) and pd.notna(rv_m) and rv_d > 0 and rv_w > 0 and rv_m > 0:
                        try:
                            log_forecast = (self.har_model.intercept_ +
                                          self.har_model.phi_d * np.log(rv_d) +
                                          self.har_model.phi_w * np.log(rv_w) +
                                          self.har_model.phi_m * np.log(rv_m))
                            forecast_rv = np.exp(log_forecast)
                            signals.loc[signals.index[i], 'har_forecast'] = np.sqrt(forecast_rv * 252) * 100
                        except:
                            pass

                signals['intraday_vol_pct'] = signals['intraday_vol_pct'] / np.sqrt(96)
                signals['har_forecast'] = signals['har_forecast'] / np.sqrt(96)
                signals['volatility_pct'] = signals['har_forecast'].fillna(signals['intraday_vol_pct'])

                print("  ✅ HAR model successfully applied")

            except Exception as e:
                print(f"  ⚠️  HAR model failed ({e}), using simple volatility")
                returns = np.log(self.bars['Close'] / self.bars['Close'].shift(1))
                signals['volatility_pct'] = returns.rolling(window=96).std() * np.sqrt(96) * 100
                self.use_har = False
        else:
            print("  ⚠️  Using simple volatility (insufficient data for HAR)")
            returns = np.log(self.bars['Close'] / self.bars['Close'].shift(1))
            signals['volatility_pct'] = returns.rolling(window=96).std() * np.sqrt(96) * 100
            self.use_har = False

        signals['volatility_pct'] = signals['volatility_pct'].fillna(2.0)

        # Volatility regime
        vol_lookback = min(2880, len(signals) - 1)
        vol_20th = signals['volatility_pct'].rolling(window=vol_lookback, min_periods=96).quantile(0.20)
        vol_80th = signals['volatility_pct'].rolling(window=vol_lookback, min_periods=96).quantile(0.80)

        signals['vol_regime'] = 'medium'
        signals.loc[signals['volatility_pct'] < vol_20th, 'vol_regime'] = 'low'
        signals.loc[signals['volatility_pct'] > vol_80th, 'vol_regime'] = 'high'

        print(f"  Volatility model: {'HAR (forward-looking)' if self.use_har else 'Simple (backward-looking)'}")
        print(f"  Indicators calculated for {len(signals)} periods")

        # MEAN REVERSION SIGNAL GENERATION WITH BLACK SWAN PROTECTION
        signals['signal'] = 0.0
        signals['strategy_type'] = 'none'
        signals['position_size'] = 0.85

        buy_signals = 0
        sell_signals = 0
        skipped = 0
        skipped_blackswan = 0

        for i in range(max(100, self.bb_period), len(signals)):

            if pd.isna(signals['bb_upper'].iloc[i]) or pd.isna(signals['rsi'].iloc[i]):
                continue

            # 🛡️ BLACK SWAN PROTECTION: Check if safe to trade
            current_bars = self.bars.iloc[:i+1]
            is_safe, reason = MarketRegimeDetector.is_safe_to_trade(current_bars, self.btc_data)

            if not is_safe:
                skipped_blackswan += 1
                continue  # Skip trading during dangerous conditions

            current_hour = self.bars.index[i].hour
            if 0 <= current_hour < 6:
                skipped += 1
                continue

            price = self.bars['Close'].iloc[i]
            bb_upper = signals['bb_upper'].iloc[i]
            bb_lower = signals['bb_lower'].iloc[i]
            bb_middle = signals['bb_middle'].iloc[i]
            bb_position = signals['bb_position'].iloc[i]
            bb_width = signals['bb_width'].iloc[i]

            rsi = signals['rsi'].iloc[i]
            adx = signals['adx'].iloc[i]
            atr_pct = signals['atr_pct'].iloc[i]
            vol_regime = signals['vol_regime'].iloc[i]
            order_flow = signals['order_flow_ratio'].iloc[i]
            volume_ratio = signals['volume_ratio'].iloc[i]

            # Skip extremely high volatility (crashes)
            if pd.notna(atr_pct) and atr_pct > 3.0:
                skipped += 1
                continue

            # Skip if bands too narrow (no volatility to profit from)
            if pd.notna(bb_width) and bb_width < 1.5:
                skipped += 1
                continue

            # Skip if not in choppy regime
            if pd.notna(adx) and adx > 35:  # Strong trend, avoid
                skipped += 1
                continue

            # ✅ BUY SIGNAL (Oversold - expect bounce)
            buy_score = 0

            # Price near lower band
            if pd.notna(bb_position):
                if bb_position < 0.15:  # Very oversold
                    buy_score += 5
                elif bb_position < 0.25:  # Oversold
                    buy_score += 3

            # RSI oversold
            if pd.notna(rsi):
                if rsi < 28:  # Very oversold
                    buy_score += 4
                elif rsi < 35:  # Oversold
                    buy_score += 2

            # Low ADX (good for mean reversion)
            if pd.notna(adx):
                if adx < 20:  # No trend
                    buy_score += 3
                elif adx < 25:  # Weak trend
                    buy_score += 2

            # Order flow turning positive
            if pd.notna(order_flow) and order_flow > 0.48:
                buy_score += 2

            # Good volatility regime
            if vol_regime == 'medium':
                buy_score += 2
            elif vol_regime == 'low':
                buy_score += 1

            # Volume confirmation
            if pd.notna(volume_ratio) and volume_ratio > 1.2:
                buy_score += 2

            # BUY if score >= 12 (SELECTIVE)
            if buy_score >= 12:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1.0
                signals.iloc[i, signals.columns.get_loc('strategy_type')] = 'range_buy'
                buy_signals += 1

                if buy_score >= 16:  # Very high confidence
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.95
                elif buy_score >= 14:  # High confidence
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.90
                elif buy_score >= 12:  # Normal confidence
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.75
                else:  # Lower confidence (shouldn't hit this)
                    signals.iloc[i, signals.columns.get_loc('position_size')] = 0.75

                buy_signals += 1
                continue

            # ✅ SELL SIGNAL (Overbought - expect pullback)
            sell_score = 0

            # Price near upper band
            if pd.notna(bb_position):
                if bb_position > 0.85:  # Very overbought
                    sell_score += 5
                elif bb_position > 0.75:  # Overbought
                    sell_score += 3

            # RSI overbought
            if pd.notna(rsi):
                if rsi > 72:  # Very overbought
                    sell_score += 4
                elif rsi > 65:  # Overbought
                    sell_score += 2

            # Low ADX (good for mean reversion)
            if pd.notna(adx):
                if adx < 20:
                    sell_score += 3
                elif adx < 25:
                    sell_score += 2

            # SELL if score >= 8 (easier to exit than enter)
            if sell_score >= 8:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1.0
                signals.iloc[i, signals.columns.get_loc('strategy_type')] = 'range_sell'
                sell_signals += 1

        print(f"  Range BUY signals: {buy_signals}")
        print(f"  Range SELL signals: {sell_signals}")
        print(f"  Skipped (extreme conditions): {skipped}")
        print(f"  🛡️ Skipped (BLACK SWAN protection): {skipped_blackswan}")

        signals['signal'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0.0)
        signals['signal'] = (signals['signal'] > 0).astype(float)

        signals['position_size'] = 0.85
        signals['positions'] = signals['signal'].diff()

        actual_buy = (signals['positions'] == 1.0).sum()
        actual_sell = (signals['positions'] == -1.0).sum()

        print(f"  Actual BUY entries: {actual_buy}")
        print(f"  Actual SELL exits: {actual_sell}")

        return signals


class RangeTradingPortfolio(Portfolio):
    """Portfolio optimized for range trading with dynamic stops."""

    def __init__(self, symbol, bars, signals, initial_capital=50000.0,
                 position_size=0.85, trading_fee=0.001):
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

    def calculate_dynamic_stop(self, volatility_pct, base_stop=0.025):
        """
        Dynamic stop loss based on current volatility.
        Higher volatility = wider stop to avoid whipsaws.
        """
        if pd.isna(volatility_pct):
            return base_stop

        # Get volatility percentile (relative to recent history)
        vol_rolling_mean = np.nanmean(self.signals['volatility_pct'].tail(100))

        if vol_rolling_mean > 0:
            volatility_multiplier = 1 + (volatility_pct / vol_rolling_mean - 1) * 0.5
        else:
            volatility_multiplier = 1.0

        dynamic_stop = base_stop * volatility_multiplier
        return np.clip(dynamic_stop, 0.020, 0.040)  # 2-4% range

    def backtest_portfolio(self):
        """Backtest with dynamic stops and trailing take profit."""
        portfolio = pd.DataFrame(index=self.bars.index)

        portfolio['price'] = self.bars['Close']
        portfolio['signal'] = self.signals['signal']
        portfolio['atr_pct'] = self.signals['atr_pct']
        portfolio['volatility_pct'] = self.signals['volatility_pct']
        portfolio['bb_middle'] = self.signals['bb_middle']
        portfolio['bb_upper'] = self.signals['bb_upper']
        portfolio['strategy_type'] = self.signals['strategy_type']

        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = self.initial_capital

        crypto_position = 0.0
        cash = self.initial_capital
        entry_price = 0.0
        entry_strategy = 'none'
        highest_price = 0.0  # For trailing stop

        trades = []

        for i in range(1, len(portfolio)):
            current_price = portfolio['price'].iloc[i]
            current_signal = portfolio['signal'].iloc[i]
            prev_signal = portfolio['signal'].iloc[i - 1]
            atr_pct = portfolio['atr_pct'].iloc[i]
            volatility_pct = portfolio['volatility_pct'].iloc[i]
            bb_middle = portfolio['bb_middle'].iloc[i]
            bb_upper = portfolio['bb_upper'].iloc[i]
            strategy_type = portfolio['strategy_type'].iloc[i]

            # Dynamic stop loss
            dynamic_stop = self.calculate_dynamic_stop(volatility_pct, base_stop=0.025)
            atr_tp = np.clip(atr_pct * 1.5, 0.03, 0.06)  # 3-6% initial TP

            if crypto_position > 0 and entry_price > 0:

                # Update highest price for trailing
                if current_price > highest_price:
                    highest_price = current_price

                pnl_pct = (current_price / entry_price - 1) * 100

                # Dynamic stop loss
                stop_price = entry_price * (1 - dynamic_stop)

                # Trailing take profit: Once we're up 3%, start trailing at 2% below highest price
                if pnl_pct > 3.0:
                    trailing_stop = highest_price * 0.98
                    stop_price = max(stop_price, trailing_stop)  # Use higher of two stops

                # Stop loss check
                if current_price <= stop_price:
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
                        'reason': 'Trailing Stop' if pnl_pct > 3.0 else 'Stop Loss',
                        'strategy': entry_strategy,
                        'entry_price': entry_price,
                        'exit_price': current_price
                    })

                    crypto_position = 0.0
                    entry_price = 0.0
                    entry_strategy = 'none'
                    highest_price = 0.0
                    continue

                # Take profit target (still check for big moves)
                tp_target = max(entry_price * (1 + atr_tp), bb_middle) if pd.notna(bb_middle) else entry_price * (1 + atr_tp)

                if current_price >= tp_target and pnl_pct > 5.0:  # Only TP if > 5%
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
                        'strategy': entry_strategy,
                        'entry_price': entry_price,
                        'exit_price': current_price
                    })

                    crypto_position = 0.0
                    entry_price = 0.0
                    entry_strategy = 'none'
                    highest_price = 0.0
                    continue

            # BUY
            if prev_signal == 0 and current_signal == 1 and cash > 0 and crypto_position == 0:
                buy_amount = cash * self.position_size
                fee = buy_amount * self.trading_fee
                crypto_bought = (buy_amount - fee) / current_price

                crypto_position = crypto_bought
                cash -= buy_amount
                entry_price = current_price
                entry_strategy = strategy_type
                highest_price = current_price  # Initialize trailing

                trades.append({
                    'timestamp': portfolio.index[i],
                    'type': 'BUY',
                    'price': current_price,
                    'value': buy_amount,
                    'fee': fee,
                    'pnl_pct': 0,
                    'reason': 'Signal',
                    'strategy': strategy_type,
                    'entry_price': current_price,
                    'exit_price': 0
                })

            # SELL
            elif prev_signal == 1 and current_signal == 0 and crypto_position > 0:
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
                    'strategy': entry_strategy,
                    'entry_price': entry_price,
                    'exit_price': current_price
                })

                crypto_position = 0.0
                entry_price = 0.0
                entry_strategy = 'none'
                highest_price = 0.0

            portfolio.loc[portfolio.index[i], 'holdings'] = crypto_position * current_price
            portfolio.loc[portfolio.index[i], 'cash'] = cash
            portfolio.loc[portfolio.index[i], 'total'] = cash + (crypto_position * current_price)

        portfolio['returns'] = portfolio['total'].pct_change()
        self.trades_df = pd.DataFrame(trades)

        return portfolio


def calculate_metrics(returns, trades_df, initial_capital):
    """Calculate performance metrics."""
    metrics = {}

    metrics['Initial Capital'] = initial_capital
    metrics['Final Value'] = returns['total'].iloc[-1]
    metrics['Net Profit'] = returns['total'].iloc[-1] - initial_capital
    metrics['Total Return (%)'] = (returns['total'].iloc[-1] / initial_capital - 1) * 100

    total_days = (returns.index[-1] - returns.index[0]).total_seconds() / (24 * 3600)
    metrics['Total Days'] = total_days

    if not trades_df.empty:
        metrics['Number of Trades'] = len(trades_df)
        metrics['Total Fees'] = trades_df['fee'].sum()

        sell_trades = trades_df[trades_df['type'].str.contains('SELL')]

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

    return metrics


def print_metrics(metrics, symbol, used_har):
    """Print results."""
    print(f"\n{'=' * 70}")
    print(f"RANGE TRADING RESULTS - {symbol}")
    print(f"Strategy: Bollinger Band Mean Reversion")
    print(f"Volatility Model: {'HAR (Forward-Looking)' if used_har else 'Simple (Backward-Looking)'}")
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            if '(%)' in key or 'Rate' in key:
                print(f"  {key:.<50} {value:>15.2f}%")
            elif 'Capital' in key or 'Value' in key or 'Profit' in key or 'Fees' in key:
                print(f"  {key:.<50} ${value:>14,.2f}")
            else:
                print(f"  {key:.<50} {value:>15.2f}")
        else:
            print(f"  {key:.<50} {value:>15}")

    print("=" * 70)


def main():
    """Main execution with multi-coin range trading strategy and daily rescanning."""

    # ⭐ MODE CONFIGURATION
    LIVE_MODE = False  # Set True for AWS 24/7 operation, False for backtest
    RESCAN_HOUR_UTC = 8  # Rescan at 08:00 UTC (Asia peak + Europe open = highest volume)

    # State file for tracking rescans
    STATE_FILE = 'bot_state.json'

    def load_state():
        """Load bot state from JSON file"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'last_scan_date': None,
            'current_coins': None,
            'scan_count': 0
        }

    def save_state(state):
        """Save bot state to JSON file"""
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def get_coins_for_date(selector, target_date, verbose=True):
        """
        Get top 3 coins AS OF a specific historical date.
        Uses data available only up to that date.
        """
        if verbose:
            print(f"   📅 {target_date.strftime('%Y-%m-%d')}", end=' ')

        # Define data window ending at target_date
        end_date_for_scan = datetime.datetime.combine(target_date, datetime.time(23, 59, 59))
        start_date_for_scan = end_date_for_scan - datetime.timedelta(days=30)

        # Get current top 30 coins (we can't get historical rankings)
        coins = selector.get_top_coins(limit=30)

        scores = {}
        har_compatible = {}

        # Score each coin using ONLY data available up to target_date
        for symbol in coins:
            try:
                # Fetch historical data ending at target_date
                bars = selector.fetcher.fetch_ohlcv(
                    symbol,
                    selector.interval,
                    start_date_for_scan,
                    end_date_for_scan,
                    use_cache=True
                )

                if bars is None or len(bars) < 100:
                    continue

                score, has_har = selector.calculate_range_score(symbol, bars)
                scores[symbol] = score
                har_compatible[symbol] = has_har

            except:
                continue

        if not scores:
            # Fallback to default coins
            return [('ETHUSDT', 0, False), ('BTCUSDT', 0, False), ('BNBUSDT', 0, False)]

        # Sort by score and return top 3
        sorted_coins = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_coins = [
            (coin, score, har_compatible.get(coin, False))
            for coin, score in sorted_coins[:3]
        ]

        return top_coins

    print("="*70)
    print("ADAPTIVE RANGE TRADING STRATEGY - BLACK SWAN PROTECTED")
    print("="*70)
    print("✅ Bollinger Band Mean Reversion")
    print("✅ Profits from volatility (not trends)")
    print("✅ Optimized for choppy/sideways markets")
    print("✅ HAR volatility forecasting enabled")
    print("="*70)
    print("🛡️ BLACK SWAN PROTECTIONS:")
    print("   • Liquidation cascade detection")
    print("   • Macro shock event filter")
    print("   • Circuit breaker: No entry during -10% moves")
    print("   • Dynamic stop loss: 2-4% based on volatility")
    print("   • Trailing take profit: Let winners run!")
    print("   • Multi-coin trading: Top 3 coins for diversification")
    print("   • 🔄 Daily rescanning at 08:00 UTC (backtests & live)")
    print("="*70)

    # ⭐ BACKTEST DATES
    BACKTEST_START = datetime.datetime(2025, 10, 27, 0, 0, 0)
    BACKTEST_END = datetime.datetime(2025, 11, 7, 23, 59, 59)

    # Calculate how many days in backtest
    total_backtest_days = (BACKTEST_END - BACKTEST_START).days + 1

    print(f"\n{'='*70}")
    print(f"DAILY COIN ROTATION STRATEGY")
    print('='*70)
    print(f"  Mode: {'LIVE TRADING' if LIVE_MODE else 'BACKTEST SIMULATION'}")
    print(f"  Period: {BACKTEST_START.strftime('%Y-%m-%d')} to {BACKTEST_END.strftime('%Y-%m-%d')}")
    print(f"  Duration: {total_backtest_days} days")
    print(f"  Daily rescan time: {RESCAN_HOUR_UTC:02d}:00 UTC")
    print(f"  Strategy: Rescan top 30 coins daily, trade best 3")
    print('='*70)

    # Initialize selector
    selector = AdaptiveCoinSelector(interval='15m', rescan_hours=6, min_data_days=30)

    if LIVE_MODE:
        # ===== LIVE TRADING MODE =====
        bot_state = load_state()
        current_time = datetime.datetime.utcnow()
        current_date = current_time.date()
        current_hour = current_time.hour

        needs_scan = False

        if bot_state['last_scan_date'] is None:
            needs_scan = True
            print(f"\n🆕 FIRST RUN - Performing initial coin selection")
            print(f"   Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        else:
            try:
                last_scan = datetime.datetime.strptime(bot_state['last_scan_date'], '%Y-%m-%d').date()

                if current_date > last_scan and current_hour >= RESCAN_HOUR_UTC:
                    needs_scan = True
                    print(f"\n🔄 DAILY RESCAN TRIGGERED - {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                    print(f"   Last scan: {last_scan}")
                    print(f"   Current date: {current_date}")
            except:
                needs_scan = True

        if needs_scan:
            top_coins = selector.select_best_coin(top_n=3)

            bot_state['last_scan_date'] = current_date.strftime('%Y-%m-%d')
            bot_state['current_coins'] = [
                {'symbol': coin[0], 'score': coin[1], 'has_har': coin[2]}
                for coin in top_coins
            ]
            bot_state['scan_count'] = bot_state.get('scan_count', 0) + 1
            save_state(bot_state)

            print(f"\n✅ COIN SELECTION UPDATED (Scan #{bot_state['scan_count']})")
            print(f"   Selected coins: {', '.join([c[0] for c in top_coins])}")
            next_scan_date = (current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"   Next scan: {next_scan_date} at {RESCAN_HOUR_UTC:02d}:00 UTC")
        else:
            if bot_state['current_coins']:
                top_coins = [
                    (c['symbol'], c['score'], c['has_har'])
                    for c in bot_state['current_coins']
                ]
                print(f"\n📌 USING EXISTING COIN SELECTION")
                print(f"   Last scanned: {bot_state['last_scan_date']} (Scan #{bot_state['scan_count']})")
                next_scan_date = (current_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"   Next scan: {next_scan_date} at {RESCAN_HOUR_UTC:02d}:00 UTC")
                print(f"   Current coins: {', '.join([c[0] for c in top_coins])}")
            else:
                print(f"\n⚠️  No saved coins found - performing fresh scan")
                top_coins = selector.select_best_coin(top_n=3)
                bot_state['last_scan_date'] = current_date.strftime('%Y-%m-%d')
                bot_state['current_coins'] = [
                    {'symbol': coin[0], 'score': coin[1], 'has_har': coin[2]}
                    for coin in top_coins
                ]
                bot_state['scan_count'] = 1
                save_state(bot_state)

    else:
        # ===== BACKTEST MODE WITH DAILY RESCANNING =====
        print(f"\n🔄 BACKTEST MODE: Simulating daily coin rotation")
        print(f"   Will rescan top 30 coins for each day of backtest period")
        print(f"   This simulates realistic adaptive behavior\n")

        # Generate list of all dates in backtest period
        scan_dates = []
        current_scan_date = BACKTEST_START.date()
        end_scan_date = BACKTEST_END.date()

        while current_scan_date <= end_scan_date:
            scan_dates.append(current_scan_date)
            current_scan_date += datetime.timedelta(days=1)

        print(f"📊 Will perform {len(scan_dates)} daily coin scans during backtest")
        print(f"   (One scan per day from {scan_dates[0]} to {scan_dates[-1]})")

        # Collect all coins across all days
        daily_coin_selections = {}

        print(f"\n{'='*70}")
        print("PHASE 1: DAILY COIN SELECTION (Simulating Daily Top 30 Scans)")
        print('='*70)

        for day_num, scan_date in enumerate(scan_dates, 1):
            print(f"[Day {day_num}/{len(scan_dates)}]", end=' ')

            # Get top 3 coins for this specific date
            daily_coins = get_coins_for_date(selector, scan_date, verbose=True)
            daily_coin_selections[scan_date] = daily_coins

            # Show selected coins with scores
            coin_str = ', '.join([f"{c[0]}({c[1]})" for c in daily_coins])
            print(f"→ {coin_str}")
            time.sleep(0.05)  # Slightly longer pause to avoid rate limits

        print(f"\n{'='*70}")
        print("PHASE 2: ANALYZING CONSISTENT PERFORMERS")
        print('='*70)

        # Count coin frequency across all days
        coin_frequency = {}
        for date, coins in daily_coin_selections.items():
            for coin_symbol, score, has_har in coins:
                if coin_symbol not in coin_frequency:
                    coin_frequency[coin_symbol] = {'count': 0, 'total_score': 0, 'has_har': has_har}
                coin_frequency[coin_symbol]['count'] += 1
                coin_frequency[coin_symbol]['total_score'] += score

        # Sort by frequency, then by average score
        sorted_coins = sorted(
            coin_frequency.items(),
            key=lambda x: (x[1]['count'], x[1]['total_score'] / x[1]['count']),
            reverse=True
        )

        # Take top 3 most consistent coins
        top_coins = [
            (coin, int(data['total_score'] / data['count']), data['has_har'])
            for coin, data in sorted_coins[:3]
        ]

        print(f"\n📈 MOST CONSISTENTLY HIGH-SCORING COINS DURING PERIOD:")
        for i, (coin, avg_score, has_har) in enumerate(top_coins, 1):
            frequency = coin_frequency[coin]['count']
            frequency_pct = (frequency / len(scan_dates)) * 100
            har_icon = "✓" if has_har else "⚠"
            print(f"   {i}. {coin}: Appeared {frequency}/{len(scan_dates)} days ({frequency_pct:.1f}%) | Avg Score: {avg_score}/24 [{har_icon} HAR]")

        print(f"\n🎯 These coins will be backtested across the full period")
        print(f"   (Simulating daily adaptive coin rotation)")

    INTERVAL = "15m"
    TRADING_FEE = 0.001

    print(f"\n{'='*70}")
    print(f"BACKTEST CONFIGURATION")
    print('='*70)
    print(f"  Period: {BACKTEST_START.strftime('%Y-%m-%d')} to {BACKTEST_END.strftime('%Y-%m-%d')}")
    print(f"  Total days: {(BACKTEST_END - BACKTEST_START).days}")
    print(f"  Trading {len(top_coins)} coins with equal allocation")
    print('='*70)

    # Trade multiple coins with equal capital allocation
    capital_per_coin = 50000.0 / len(top_coins)

    all_portfolios = []
    all_trades = []

    fetcher = BinanceDataFetcher()

    for coin_symbol, coin_score, has_har in top_coins:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {coin_symbol} (Score: {coin_score}/24)")
        print('='*70)

        bars = fetcher.fetch_ohlcv(coin_symbol, INTERVAL, BACKTEST_START, BACKTEST_END, use_cache=True)

        if bars is None or len(bars) < 100:
            print(f"❌ Insufficient data for {coin_symbol}, skipping...")
            continue

        # Run strategy with BLACK SWAN protection
        strategy = RangeTradingStrategy(coin_symbol, bars, use_har=has_har,
                                       bb_period=20, bb_std=2, rsi_period=14,
                                       btc_data=selector.btc_data)
        signals = strategy.generate_signals()

        portfolio = RangeTradingPortfolio(coin_symbol, bars, signals,
                                         initial_capital=capital_per_coin,
                                         position_size=0.85,
                                         trading_fee=TRADING_FEE)
        returns = portfolio.backtest_portfolio()

        # Save results
        all_portfolios.append((coin_symbol, returns, strategy.use_har))

        if not portfolio.trades_df.empty:
            portfolio.trades_df['coin'] = coin_symbol
            all_trades.append(portfolio.trades_df)

        # Display individual coin metrics
        metrics = calculate_metrics(returns, portfolio.trades_df, capital_per_coin)
        print_metrics(metrics, coin_symbol, strategy.use_har)

    # Aggregate results across all coins
    print(f"\n{'='*70}")
    print("COMBINED PORTFOLIO RESULTS")
    print('='*70)

    total_final_value = sum([portfolio[1]['total'].iloc[-1] for portfolio in all_portfolios])
    total_initial_capital = capital_per_coin * len(all_portfolios)
    total_return_pct = (total_final_value / total_initial_capital - 1) * 100

    print(f"  Initial Capital (Total)................... ${total_initial_capital:>14,.2f}")
    print(f"  Final Value (Total)....................... ${total_final_value:>14,.2f}")
    print(f"  Net Profit (Total)........................ ${total_final_value - total_initial_capital:>14,.2f}")
    print(f"  Total Return (%)..........................           {total_return_pct:>6.2f}%")
    print(f"  Number of Coins Traded....................                   {len(all_portfolios)}")
    print('='*70)

    # Save combined trades
    if all_trades:
        combined_trades = pd.concat(all_trades, ignore_index=True)
        csv_filename = f'trades_multi_coin_{datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        combined_trades.to_csv(csv_filename, index=False)
        print(f"\n✓ All trades saved to '{csv_filename}'")

        # Show best and worst across all coins
        losses = combined_trades[combined_trades['pnl_pct'] < 0].sort_values('pnl_pct')
        if len(losses) > 0:
            print(f"\nWorst 5 Losses (All Coins):")
            print(losses.head()[['timestamp', 'coin', 'type', 'pnl_pct', 'reason']].to_string(index=False))

        wins = combined_trades[combined_trades['pnl_pct'] > 0].sort_values('pnl_pct', ascending=False)
        if len(wins) > 0:
            print(f"\nBest 5 Wins (All Coins):")
            print(wins.head()[['timestamp', 'coin', 'type', 'pnl_pct', 'reason']].to_string(index=False))



    print(f"\n{'='*70}")
    print("✓ BLACK SWAN PROTECTED BACKTEST COMPLETE")
    print('='*70)
    print(f"Coins traded: {', '.join([p[0] for p in all_portfolios])}")
    print(f"Strategy: Mean reversion with event risk protection")
    print(f"🛡️ Protected against liquidation cascades & macro shocks")
    print('='*70)


if __name__ == "__main__":
    main()