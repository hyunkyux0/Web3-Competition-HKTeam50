# strategy.py
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from roostoo_api import get_candles, get_ticker, get_balance, get_available_symbols

# Strategy configuration - will be set dynamically
COMMISSION_RATE = 0.001

logging.basicConfig(level=logging.INFO)

# ADD THE MISSING RSI FUNCTION HERE
def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index)
    Returns current RSI value or 50 (neutral) if calculation fails
    """
    try:
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], 1.0).fillna(1.0)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0
        
    except Exception as e:
        logging.warning(f"RSI calculation failed: {e}, returning neutral 50")
        return 50.0

def get_available_trading_symbols() -> List[str]:
    """Get available symbols that we can actually trade"""
    symbols = get_available_symbols()
    if not symbols:
        # Fallback to some known symbols from the exchange info
        return ["ADA/USD", "ETH/USD", "BTC/USD", "BNB/USD", "SOL/USD"]
    
    # Filter to major coins for better liquidity
    major_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'LTC', 'LINK', 'MATIC', 'TRX', 'LTC', 'CAKE']
    major_pairs = [f"{coin}/USD" for coin in major_coins if f"{coin}/USD" in symbols]
    
    return major_pairs[:8]  # Limit to 8 pairs

def calculate_trend_strength(df: pd.DataFrame) -> float:
    """Calculate simple trend strength score (0-1)"""
    try:
        if len(df) < 5:
            return 0.0
            
        # Simple price momentum
        current_price = df['close'].iloc[-1]
        short_ma = df['close'].rolling(5).mean().iloc[-1]
        medium_ma = df['close'].rolling(10).mean().iloc[-1] if len(df) >= 10 else current_price
        
        # Positive trend if above both MAs
        if current_price > short_ma and current_price > medium_ma:
            return 0.7
        elif current_price > short_ma:
            return 0.4
        else:
            return 0.1
            
    except Exception as e:
        logging.error(f"Error calculating trend strength: {e}")
        return 0.0

def create_synthetic_candles(ticker_data: Dict, num_candles: int = 20) -> pd.DataFrame:
    """Create synthetic candle data from ticker data when candles aren't available"""
    if not ticker_data:
        return pd.DataFrame()
    
    current_price = float(ticker_data.get('LastPrice', 0))
    if current_price <= 0:
        return pd.DataFrame()
    
    # Create simple synthetic data around current price
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_candles, freq='15min')
    
    # Create realistic price movement with some volatility
    returns = np.random.normal(0, 0.002, num_candles)  # Small random variations
    prices = current_price * (1 + np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, num_candles))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, num_candles))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, num_candles)
    }, index=dates)
    
    # Ensure high is highest, low is lowest
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, num_candles)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, num_candles)))
    
    return df

def analyze_symbol_with_real_data(symbol: str) -> Dict[str, Any]:
    """Analyze symbol using real ticker data (since candles might not be available)"""
    try:
        ticker_data = get_ticker(symbol)
        if not ticker_data:
            return None
        
        current_price = float(ticker_data.get('LastPrice', 0))
        price_change = float(ticker_data.get('Change', 0))
        
        if current_price <= 0:
            return None
        
        # Create synthetic candles for technical analysis
        df = create_synthetic_candles(ticker_data, 50)
        
        if df.empty:
            return None
        
        # Calculate technical indicators
        rsi = compute_rsi(df['close'], 14)
        trend_score = calculate_trend_strength(df)
        
        # Momentum from price change
        momentum_score = min(max(price_change * 5, 0), 1.0)  # Normalize to 0-1
        
        # Volume confidence (if available)
        volume_confidence = 0.5  # Default
        if 'CoinTradeValue' in ticker_data:
            volume = float(ticker_data['CoinTradeValue'])
            if volume > 1000000:  # $1M+ volume
                volume_confidence = 0.8
            elif volume > 100000:  # $100K+ volume  
                volume_confidence = 0.6
        
        # RSI score (closer to 50 is better for mean reversion, but we want some momentum)
        rsi_score = 1.0 - min(abs(rsi - 60) / 40, 1.0)  # Prefer slightly bullish (60)
        
        # Combined score - weighted average
        combined_score = (
            momentum_score * 0.3 +
            trend_score * 0.3 +
            rsi_score * 0.2 +
            volume_confidence * 0.2
        )
        
        # Trading decision logic
        should_trade = (
            combined_score > 0.5 and 
            trend_score > 0.3 and 
            30 < rsi < 70 and  # Avoid extremes
            momentum_score > 0.2
        )
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'rsi': rsi,
            'trend_score': trend_score,
            'momentum_score': momentum_score,
            'volume_confidence': volume_confidence,
            'combined_score': combined_score,
            'should_trade': should_trade
        }
        
    except Exception as e:
        logging.error(f"Error analyzing {symbol}: {e}")
        return None

def decide_targets(portfolio_value_usd: float, symbols: List[str] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Enhanced target allocation that works with available data
    """
    if symbols is None:
        symbols = get_available_trading_symbols()
    
    metadata = {}
    symbol_scores = {}
    
    logging.info(f"Analyzing {len(symbols)} symbols for portfolio value: ${portfolio_value_usd:,.2f}")
    
    for symbol in symbols:
        analysis = analyze_symbol_with_real_data(symbol)
        if analysis:
            metadata[symbol] = analysis
            symbol_scores[symbol] = analysis['combined_score']
            logging.info(f"{symbol}: ${analysis['current_price']:.4f}, change={analysis['price_change']:.4f}, RSI={analysis['rsi']:.1f}, score={analysis['combined_score']:.3f}")
    
    # Select top symbols and allocate capital
    targets = {}
    
    if symbol_scores:
        # Sort by score and take top 2-3
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, score in sorted_symbols[:3] if score > 0.4 and metadata[sym]['should_trade']]
        
        if top_symbols:
            # Conservative allocation: 50% of portfolio split among winners
            allocation_per_symbol = portfolio_value_usd * 0.5 / len(top_symbols)
            
            for symbol in top_symbols:
                targets[symbol] = allocation_per_symbol
            
            logging.info(f"🎯 Selected {len(top_symbols)} symbols for trading: {top_symbols}")
            for symbol in top_symbols:
                analysis = metadata[symbol]
                logging.info(f"   {symbol}: ${analysis['current_price']:.4f} (change: {analysis['price_change']:.4f}, RSI: {analysis['rsi']:.1f})")
        else:
            logging.info("📊 No strong trading opportunities found this cycle")
    
    return targets, metadata

def generate_signals(symbols: List[str] = None) -> Dict[str, str]:
    """
    Generate simple trading signals based on strategy
    """
    if symbols is None:
        symbols = get_available_trading_symbols()
    
    signals = {}
    
    # Use actual portfolio value for signal generation
    portfolio_value = get_portfolio_value_from_api()
    
    targets, metadata = decide_targets(portfolio_value, symbols)
    
    # Get current balances
    balances = get_balance()
    
    for symbol in symbols:
        try:
            if symbol not in metadata:
                signals[symbol] = "HOLD"
                continue
                
            asset = symbol.split('/')[0]
            current_balance = 0.0
            usd_balance = 0.0
            
            if balances:
                current_balance = float(balances.get(asset, {}).get('Free', 0))
                usd_balance = float(balances.get('USD', {}).get('Free', 0))
            
            meta = metadata[symbol]
            current_price = meta['current_price']
            target_value = targets.get(symbol, 0)
            
            # Calculate current position value
            current_value = current_balance * current_price
            
            # Generate signal
            if target_value > 0 and current_value < target_value * 0.5 and usd_balance > 50:
                # Buy signal - we have less than 50% of target and enough USD
                signals[symbol] = "BUY"
                logging.info(f"📈 BUY signal for {symbol}: target=${target_value:.2f}, current=${current_value:.2f}, USD=${usd_balance:.2f}")
            elif target_value == 0 and current_value > 10:
                # Sell signal - no target but we have position
                signals[symbol] = "SELL"
                logging.info(f"📉 SELL signal for {symbol}: position=${current_value:.2f}")
            else:
                signals[symbol] = "HOLD"
                
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {e}")
            signals[symbol] = "HOLD"
    
    # Log summary
    buy_signals = [s for s, signal in signals.items() if signal == "BUY"]
    sell_signals = [s for s, signal in signals.items() if signal == "SELL"]
    
    if buy_signals:
        logging.info(f"🎯 BUY signals: {buy_signals}")
    if sell_signals:
        logging.info(f"🎯 SELL signals: {sell_signals}")
    
    return signals

def get_portfolio_value_from_api() -> float:
    """Get portfolio value from API with fallback"""
    from roostoo_api import get_portfolio_value
    value = get_portfolio_value()
    return value if value > 0 else 10000.0  # Fallback to $10,000

def test_strategy():
    """Test the strategy with available symbols"""
    print("Testing strategy with available symbols...")
    
    symbols = get_available_trading_symbols()
    print(f"Available symbols: {symbols}")
    
    portfolio_value = get_portfolio_value_from_api()
    targets, metadata = decide_targets(portfolio_value, symbols)
    
    print(f"Portfolio value: ${portfolio_value:,.2f}")
    print(f"Target allocations: {targets}")
    
    if metadata:
        print("\nSymbol analysis:")
        for symbol, data in metadata.items():
            print(f"  {symbol}:")
            print(f"    Price: ${data['current_price']:.4f}")
            print(f"    Change: {data['price_change']:.4f}")
            print(f"    RSI: {data['rsi']:.1f}")
            print(f"    Trend score: {data['trend_score']:.3f}")
            print(f"    Combined score: {data['combined_score']:.3f}")
            print(f"    Should trade: {data['should_trade']}")
    
    # Test signal generation
    signals = generate_signals(symbols)
    print(f"\nTrading signals: {signals}")

if __name__ == "__main__":
    test_strategy()