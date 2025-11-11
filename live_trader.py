# live_trader.py
import time
import logging
import csv
import os
from datetime import datetime, timezone
from typing import Dict, List
from decimal import Decimal, getcontext

from roostoo_api import (
    get_balance, get_ticker, place_market_order, get_portfolio_value,
    test_api_connection
)
from strategy import generate_signals

# ---- Precision for Decimal math ----
getcontext().prec = 28

# ---- Config ----
LOOP_INTERVAL = 900        # 15 minutes in seconds
MAX_DRAWDOWN = 0.15        # 15% max drawdown
MAX_API_ERRORS = 5
COMMISSION_RATE = 0.001    # 0.1%
MIN_BUY_USD = Decimal("100")   # minimum USD for normal strategy buys
BUY_BUDGET_FRACTION = Decimal("0.10")  # 10% of USD on normal buys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# ----------------- Helpers (only what we need) -----------------
def _d(x) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))

def _floor_to_step(qty: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return qty
    return (qty // step) * step

def _get_rules(symbol: str) -> Dict[str, Decimal]:
    """
    Extract trading filters from ticker payload; fall back to safe defaults.
    """
    t = get_ticker(symbol) or {}
    step = t.get("StepSize") or t.get("stepSize") or t.get("LotStep") or "0.00001"
    min_qty = t.get("MinQty") or t.get("minQty") or "0.00001"
    min_notional = t.get("MinNotional") or t.get("minNotional") or "5"   # $5 default
    tick = t.get("TickSize") or t.get("tickSize") or "0.01"
    return {
        "step": _d(step),
        "min_qty": _d(min_qty),
        "min_notional": _d(min_notional),
        "tick": _d(tick),
    }

def _normalize_qty_for_market(price: Decimal, raw_qty: Decimal, rules: Dict[str, Decimal]) -> Decimal:
    """
    Return a qty that satisfies step/min filters (floored to step). Decimal('0') if not possible.
    """
    step = rules["step"]
    min_qty = rules["min_qty"]
    min_notional = rules["min_notional"]

    qty = _floor_to_step(raw_qty, step)

    if qty < min_qty:
        qty = _floor_to_step(min_qty, step)

    if qty * price < min_notional:
        needed = (min_notional / price)
        qty = _floor_to_step(needed, step)

    if qty < min_qty or (qty * price) < min_notional:
        return Decimal("0")

    return qty

# ----------------- Files & logging -----------------
def initialize_files():
    os.makedirs('logs', exist_ok=True)
    if not os.path.exists('trade_ledger.csv'):
        with open('trade_ledger.csv', 'w', newline='') as f:
            csv.writer(f).writerow(
                ['timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'fee', 'reason']
            )
    if not os.path.exists('pv_log.csv'):
        with open('pv_log.csv', 'w', newline='') as f:
            csv.writer(f).writerow(['timestamp', 'portfolio_value'])

def log_trade(symbol: str, side: str, quantity: float, price: float, reason: str = ""):
    value = quantity * price
    fee = value * COMMISSION_RATE
    with open('trade_ledger.csv', 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol, side, quantity, price, value, fee, reason
        ])
    logging.info(f"Logged trade: {side} {quantity:.6f} {symbol} @ ${price:.4f}")

def log_portfolio_value(value: float):
    with open('pv_log.csv', 'a', newline='') as f:
        csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(), value])

# ----------------- Trading primitives -----------------
def execute_trade(symbol: str, side: str, quantity: float, reason: str = "") -> bool:
    """
    Normalize to exchange filters and place the order.
    """
    try:
        if quantity <= 0:
            logging.warning(f"Invalid quantity {quantity} for {symbol}")
            return False

        ticker = get_ticker(symbol)
        if not ticker:
            logging.error(f"Could not get ticker for {symbol}")
            return False

        current_price = float(ticker.get('LastPrice', 0))
        if current_price <= 0:
            logging.error(f"Invalid price for {symbol}: {current_price}")
            return False

        rules = _get_rules(symbol)
        normalized_qty = _normalize_qty_for_market(_d(current_price), _d(quantity), rules)
        if normalized_qty <= 0:
            logging.error(f"Order blocked by filters for {symbol}: qty={quantity}, price={current_price}, rules={rules}")
            return False

        # Place order
        order_result = place_market_order(symbol, side, float(normalized_qty))
        if not order_result:
            logging.error(f"Failed to place {side} order for {symbol}")
            return False

        log_trade(symbol, side, float(normalized_qty), current_price, reason)
        logging.info(f"Executed {side} {float(normalized_qty):.6f} {symbol}")
        return True

    except Exception as e:
        logging.error(f"Error executing trade for {symbol}: {e}")
        return False

# ----------------- One-time bootstrap BUY -----------------
def bootstrap_first_trade(amount_usd: Decimal, preferred_symbol: str, symbols: List[str]) -> bool:
    """
    Try to BUY 'amount_usd' of 'preferred_symbol'. If filters/limits block it,
    try the rest of 'symbols'. Returns True if any trade is executed.
    """
    try:
        balances = get_balance() or {}
        usd_free = _d(balances.get('USD', {}).get('Free', 0))
        if usd_free < amount_usd:
            logging.warning(f"Bootstrap skipped: USD free {usd_free} < requested {amount_usd}")
            return False

        trial_list = [preferred_symbol] + [s for s in symbols if s != preferred_symbol]

        for symbol in trial_list:
            t = get_ticker(symbol)
            if not t or 'LastPrice' not in t:
                logging.warning(f"No ticker for {symbol}, skipping bootstrap attempt.")
                continue

            price = _d(t['LastPrice'])
            net_amount = amount_usd * (Decimal("1") - _d(COMMISSION_RATE))
            raw_qty = net_amount / price

            rules = _get_rules(symbol)
            qty = _normalize_qty_for_market(price, raw_qty, rules)

            if qty > 0:
                logging.warning(f"BOOTSTRAP BUY: attempting ~${amount_usd} of {symbol} (qty={qty})")
                ok = execute_trade(symbol, "BUY", float(qty), reason=f"Bootstrap ${amount_usd} buy")
                if ok:
                    # Write a marker so we never repeat accidentally
                    try:
                        with open('logs/bootstrap_done.flag', 'w') as f:
                            f.write(datetime.now(timezone.utc).isoformat())
                    except Exception:
                        pass
                    return True
                else:
                    logging.warning(f"Bootstrap attempt failed for {symbol}. Trying next symbol...")
            else:
                logging.warning(f"Bootstrap qty not viable for {symbol} (min/step/notional). Trying next symbol...")

        logging.error("Bootstrap failed for all candidates.")
        return False

    except Exception as e:
        logging.error(f"Bootstrap error: {e}")
        return False

# ----------------- Main loop -----------------
def rebalance_portfolio():
    consecutive_errors = 0
    peak_portfolio_value = 0.0
    symbols = ["ETH/USD", "BTC/USD", "BNB/USD", "SOL/USD", "ADA/USD"]  # ETH first for bootstrap

    logging.info("Starting trading bot...")

    # Test API first
    if not test_api_connection():
        logging.error("API connection test failed. Exiting.")
        return

    # One-time forced BUY to initialize (env-controlled, idempotent via flag file)
    if os.environ.get("FORCE_BOOTSTRAP", "0").lower() not in {"", "0", "false"}:
        if not os.path.exists("logs/bootstrap_done.flag"):
            amount = _d(os.environ.get("BOOTSTRAP_USD", "10"))
            preferred = os.environ.get("BOOTSTRAP_SYMBOL", "ETH/USD").strip() or "ETH/USD"
            did = bootstrap_first_trade(amount, preferred, symbols)
            if did:
                logging.warning(f"Bootstrap succeeded: bought ~${amount} of {preferred} (or fallback).")
            else:
                logging.warning("Bootstrap did not execute a trade.")
        else:
            logging.info("Bootstrap flag exists; skipping forced buy.")

    while True:
        try:
            # Portfolio state
            portfolio_value = get_portfolio_value()
            if portfolio_value <= 0:
                logging.warning("Could not get portfolio value, skipping iteration")
                consecutive_errors += 1
                time.sleep(60)
                continue

            if portfolio_value > peak_portfolio_value:
                peak_portfolio_value = portfolio_value

            drawdown = (peak_portfolio_value - portfolio_value) / peak_portfolio_value if peak_portfolio_value else 0.0
            logging.info(f"Portfolio: ${portfolio_value:,.2f}, Peak: ${peak_portfolio_value:,.2f}, Drawdown: {drawdown:.2%}")

            # Stops
            if peak_portfolio_value and drawdown >= MAX_DRAWDOWN:
                logging.error(f"Max drawdown ({MAX_DRAWDOWN:.1%}) reached. Stopping bot.")
                break
            if consecutive_errors >= MAX_API_ERRORS:
                logging.error(f"Too many consecutive errors ({consecutive_errors}). Stopping bot.")
                break

            # Signals
            signals = generate_signals(symbols)
            logging.info(f"Generated signals: {signals}")

            # Balances
            balances = get_balance()
            if not balances:
                consecutive_errors += 1
                time.sleep(60)
                continue

            # Execute
            trades_executed = 0
            for symbol, signal in signals.items():
                if signal == "BUY":
                    usd_balance = _d(balances.get('USD', {}).get('Free', 0))
                    if usd_balance > MIN_BUY_USD:
                        t = get_ticker(symbol)
                        if t and 'LastPrice' in t:
                            price = _d(t['LastPrice'])
                            buy_amount = usd_balance * BUY_BUDGET_FRACTION
                            net_amount = buy_amount * (Decimal("1") - _d(COMMISSION_RATE))
                            raw_qty = net_amount / price
                            rules = _get_rules(symbol)
                            qty = _normalize_qty_for_market(price, raw_qty, rules)
                            if qty > 0 and execute_trade(symbol, "BUY", float(qty), "Signal-based buy"):
                                trades_executed += 1
                            elif qty <= 0:
                                logging.error(f"Cannot meet filters for {symbol} (BUY).")

                elif signal == "SELL":
                    asset = symbol.split('/')[0]
                    asset_balance = _d(balances.get(asset, {}).get('Free', 0))
                    if asset_balance > 0:
                        t = get_ticker(symbol)
                        if t and 'LastPrice' in t:
                            price = _d(t['LastPrice'])
                            rules = _get_rules(symbol)
                            qty = _normalize_qty_for_market(price, asset_balance, rules)
                            if qty > 0 and execute_trade(symbol, "SELL", float(qty), "Signal-based sell"):
                                trades_executed += 1
                            elif qty <= 0:
                                logging.error(f"Cannot meet filters for {symbol} (SELL).")

            if trades_executed > 0:
                logging.info(f"Executed {trades_executed} trades this cycle")

            log_portfolio_value(portfolio_value)
            consecutive_errors = 0

            logging.info(f"Waiting {LOOP_INTERVAL} seconds for next cycle...")
            time.sleep(LOOP_INTERVAL)

        except Exception as e:
            consecutive_errors += 1
            logging.error(f"Error in main loop: {e}")
            time.sleep(60)

def main():
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