# live_trader.py
import time
import logging
import csv
import os
from datetime import datetime, timezone
from typing import Dict, List
from roostoo_api import (
    get_balance, get_ticker, place_market_order, get_portfolio_value,
    test_api_connection
)
from strategy import generate_signals, decide_targets

# Configuration
LOOP_INTERVAL = 900  # 15 minutes in seconds
MAX_DRAWDOWN = 0.15  # 15% max drawdown
MAX_API_ERRORS = 5
COMMISSION_RATE = 0.001  # 0.1%

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# Initialize data files
def initialize_files():
    """Initialize CSV files for logging"""
    os.makedirs('logs', exist_ok=True)
    
    # Trade ledger
    if not os.path.exists('trade_ledger.csv'):
        with open('trade_ledger.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'fee', 'reason'])
    
    # Portfolio value log
    if not os.path.exists('pv_log.csv'):
        with open('pv_log.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'portfolio_value'])

def log_trade(symbol: str, side: str, quantity: float, price: float, reason: str = ""):
    """Log a trade to the ledger"""
    value = quantity * price
    fee = value * COMMISSION_RATE
    
    with open('trade_ledger.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol,
            side,
            quantity,
            price,
            value,
            fee,
            reason
        ])
    
    logging.info(f"Logged trade: {side} {quantity:.4f} {symbol} @ ${price:.2f}")

def log_portfolio_value(value: float):
    """Log portfolio value"""
    with open('pv_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            value
        ])

def execute_trade(symbol: str, side: str, quantity: float, reason: str = "") -> bool:
    """Execute a trade with proper error handling"""
    try:
        if quantity <= 0:
            logging.warning(f"Invalid quantity {quantity} for {symbol}")
            return False
        
        # Get current price for logging
        ticker = get_ticker(symbol)
        if not ticker:
            logging.error(f"Could not get ticker for {symbol}")
            return False
        
        current_price = float(ticker.get('LastPrice', 0))
        if current_price <= 0:
            logging.error(f"Invalid price for {symbol}: {current_price}")
            return False
        
        # Place order
        order_result = place_market_order(symbol, side, quantity)
        if not order_result:
            logging.error(f"Failed to place {side} order for {symbol}")
            return False
        
        # Log the trade
        log_trade(symbol, side, quantity, current_price, reason)
        logging.info(f"Successfully executed {side} order for {quantity:.4f} {symbol}")
        return True
        
    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")
        return False

def rebalance_portfolio():
    """Main portfolio rebalancing logic"""
    consecutive_errors = 0
    peak_portfolio_value = 0
    symbols = ["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "ADA/USD"]
    
    logging.info("Starting trading bot...")
    
    # Test API connection first
    if not test_api_connection():
        logging.error("API connection test failed. Exiting.")
        return
    
    while True:
        try:
            # Get current portfolio value
            portfolio_value = get_portfolio_value()
            if portfolio_value <= 0:
                logging.warning("Could not get portfolio value, skipping iteration")
                consecutive_errors += 1
                time.sleep(60)
                continue
            
            # Update peak and check drawdown
            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value
            
            drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value
            logging.info(f"Portfolio: ${portfolio_value:,.2f}, Peak: ${peak_portfolio_value:,.2f}, Drawdown: {drawdown:.2%}")
            
            # Check stop conditions
            if drawdown >= MAX_DRAWDOWN:
                logging.error(f"Max drawdown ({MAX_DRAWDOWN:.1%}) reached. Stopping bot.")
                break
            
            if consecutive_errors >= MAX_API_ERRORS:
                logging.error(f"Too many consecutive errors ({consecutive_errors}). Stopping bot.")
                break
            
            # Generate trading signals
            signals = generate_signals(symbols)
            logging.info(f"Generated signals: {signals}")
            
            # Get current balances
            balances = get_balance()
            if not balances:
                consecutive_errors += 1
                time.sleep(60)
                continue
            
            # Execute trades based on signals
            trades_executed = 0
            for symbol, signal in signals.items():
                if signal == "BUY":
                    # Simple buy logic - use 10% of available USD
                    usd_balance = float(balances.get('USD', {}).get('Free', 0))
                    if usd_balance > 100:  # Minimum $100
                        buy_amount = usd_balance * 0.1
                        
                        # Get current price to calculate quantity
                        ticker = get_ticker(symbol)
                        if ticker and 'LastPrice' in ticker:
                            price = float(ticker['LastPrice'])
                            quantity = buy_amount / price
                            
                            if execute_trade(symbol, "BUY", quantity, "Signal-based buy"):
                                trades_executed += 1
                
                elif signal == "SELL":
                    # Sell entire position
                    asset = symbol.split('/')[0]
                    asset_balance = float(balances.get(asset, {}).get('Free', 0))
                    
                    if asset_balance > 0:
                        if execute_trade(symbol, "SELL", asset_balance, "Signal-based sell"):
                            trades_executed += 1
            
            if trades_executed > 0:
                logging.info(f"Executed {trades_executed} trades this cycle")
            
            # Log portfolio value
            log_portfolio_value(portfolio_value)
            
            # Reset error counter on successful iteration
            consecutive_errors = 0
            
            # Wait for next cycle
            logging.info(f"Waiting {LOOP_INTERVAL} seconds for next cycle...")
            time.sleep(LOOP_INTERVAL)
            
        except Exception as e:
            consecutive_errors += 1
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)  # Shorter wait on error

def main():
    """Main entry point"""
    print("Roostoo Trading Bot Starting...")
    print("=" * 50)
    
    initialize_files()
    
    try:
        rebalance_portfolio()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
    finally:
        logging.info("Trading bot stopped")

if __name__ == "__main__":
    main()