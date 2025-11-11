# quick_test.py
from roostoo_api import get_balance, get_portfolio_value, get_available_symbols
from strategy import decide_targets, generate_signals, compute_rsi
import pandas as pd
import numpy as np

def test_rsi_function():
    """Test that the RSI function works correctly"""
    print("🧪 Testing RSI function...")
    
    # Create sample price data
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 
                       110, 112, 111, 113, 115, 114, 116, 118, 117, 119])
    
    rsi = compute_rsi(prices, 14)
    print(f"RSI calculated: {rsi:.2f}")
    
    if 0 <= rsi <= 100:
        print("✅ RSI function working correctly!")
    else:
        print("❌ RSI function has issues")

def quick_test():
    print("🚀 QUICK TRADING TEST")
    print("=" * 50)
    
    # Test RSI first
    test_rsi_function()
    print()
    
    # Test balance
    balance = get_balance()
    print(f"💰 Current Balance: {balance}")
    
    # Test portfolio value
    portfolio_value = get_portfolio_value()
    print(f"📊 Portfolio Value: ${portfolio_value:,.2f}")
    
    # Test strategy
    targets, metadata = decide_targets(portfolio_value)
    print(f"🎯 Trading Targets: {targets}")
    
    # Test signals
    signals = generate_signals()
    print(f"📡 Trading Signals: {signals}")
    
    print("=" * 50)
    print("✅ Ready to trade!")

if __name__ == "__main__":
    quick_test()