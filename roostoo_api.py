# roostoo_api.py
import time
import hmac
import hashlib
import requests
import logging
import os
from typing import Dict, Optional, List


# ======= CONFIG =======
import os
BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.environ.get("ROOSTOO_API_KEY", "")
SECRET_KEY = os.environ.get("ROOSTOO_SECRET_KEY", "")

# Rate limiting
_last_request_time = 0
MIN_REQUEST_INTERVAL = 1.0  # 1 second between requests

logging.basicConfig(level=logging.INFO)

def _rate_limit():
    """Enforce rate limiting"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
    _last_request_time = time.time()

def _get_timestamp():
    """Return a 13-digit timestamp in milliseconds as string."""
    return str(int(time.time() * 1000))

def _sign_payload(payload: dict) -> tuple:
    """Return (headers, payload_str) with correct HMAC SHA256 signature."""
    payload['timestamp'] = _get_timestamp()
    sorted_keys = sorted(payload.keys())
    total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
    
    print(f"Signing payload: {total_params}")  # Debug
    
    signature = hmac.new(
        SECRET_KEY.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature
    }
    return headers, total_params

# ---------- PUBLIC ENDPOINTS ----------
def check_server_time():
    """GET /v3/serverTime"""
    _rate_limit()
    try:
        r = requests.get(f"{BASE_URL}/v3/serverTime", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"check_server_time failed: {e}")
        return None

def get_exchange_info():
    """GET /v3/exchangeInfo"""
    _rate_limit()
    try:
        r = requests.get(f"{BASE_URL}/v3/exchangeInfo", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"get_exchange_info failed: {e}")
        return None

def get_ticker(pair: str = None):
    """GET /v3/ticker"""
    _rate_limit()
    params = {"timestamp": _get_timestamp()}
    if pair:
        params["pair"] = pair
    
    try:
        r = requests.get(f"{BASE_URL}/v3/ticker", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        print(f"Ticker response: {data}")  # Debug
        
        if data.get("Success"):
            if pair and "Data" in data:
                return data["Data"].get(pair)
            elif "Data" in data:
                return data["Data"]
        else:
            logging.warning(f"Ticker API returned error: {data.get('ErrMsg', 'Unknown error')}")
            return None
            
    except Exception as e:
        logging.error(f"get_ticker failed for {pair}: {e}")
        return None

def get_candles(pair: str, interval: str = "15m", limit: int = 100):
    """
    GET /v3/candles
    Note: The API might have different endpoint structure
    """
    _rate_limit()
    params = {
        "pair": pair,
        "interval": interval,
        "limit": limit,
        "timestamp": _get_timestamp()
    }
    
    try:
        r = requests.get(f"{BASE_URL}/v3/candles", params=params, timeout=10)
        print(f"Candles request URL: {r.url}")  # Debug
        print(f"Candles response status: {r.status_code}")  # Debug
        
        if r.status_code == 404:
            logging.warning(f"Candles endpoint not found for {pair}. Trying alternative approach.")
            return get_ticker(pair)  # Fallback to ticker data
            
        r.raise_for_status()
        data = r.json()
        
        print(f"Candles response: {data}")  # Debug
        
        if data.get("Success") and "Data" in data:
            return data["Data"]
        else:
            logging.warning(f"Candles API error for {pair}: {data.get('ErrMsg', 'Unknown error')}")
            return None
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Candles not available for {pair}, using ticker data")
            return get_ticker(pair)
        else:
            logging.error(f"get_candles failed for {pair}: {e}")
            return None
    except Exception as e:
        logging.error(f"get_candles failed for {pair}: {e}")
        return None

# ---------- SIGNED ENDPOINTS ----------
def get_balance():
    """GET /v3/balance - Get current wallet balance"""
    _rate_limit()
    url = f"{BASE_URL}/v3/balance"
    
    try:
        payload = {}
        headers, payload_str = _sign_payload(payload)
        
        r = requests.get(url, headers=headers, params=payload, timeout=10)
        
        print(f"Balance response status: {r.status_code}")
        
        r.raise_for_status()
        data = r.json()
        
        print(f"Balance response: {data}")
        
        if data.get("Success"):
            # FIX: Use SpotWallet instead of Wallet
            spot_wallet = data.get("SpotWallet", {})
            print(f"✓ Balance loaded successfully: {spot_wallet}")
            return spot_wallet
        else:
            error_msg = data.get('ErrMsg', 'Unknown error')
            logging.warning(f"Balance API error: {error_msg}")
            return None
            
    except Exception as e:
        logging.error(f"get_balance failed: {e}")
        return None

def place_market_order(pair: str, side: str, quantity: float):
    """
    Place MARKET order
    """
    _rate_limit()
    url = f"{BASE_URL}/v3/place_order"
    payload = {
        "pair": pair,
        "side": side.upper(),
        "type": "MARKET",
        "quantity": str(quantity)
    }
    
    try:
        headers, payload_str = _sign_payload(payload)
        r = requests.post(url, headers=headers, data=payload_str, timeout=10)
        
        print(f"Order request data: {payload_str}")  # Debug
        print(f"Order response status: {r.status_code}")  # Debug
        
        r.raise_for_status()
        data = r.json()
        
        if data.get("Success"):
            logging.info(f"Order placed: {side} {quantity} {pair}")
            return data.get("OrderDetail", {})
        else:
            error_msg = data.get('ErrMsg', 'Unknown error')
            logging.error(f"Order failed: {error_msg}")
            return None
            
    except Exception as e:
        logging.error(f"place_market_order failed: {e}")
        return None

# Utility functions
def get_portfolio_value() -> float:
    """Calculate total portfolio value in USD - UPDATED"""
    balances = get_balance()
    if not balances:
        return 0.0
    
    total_value = 0.0
    
    print(f"Calculating portfolio from: {balances}")
    
    for asset, balance in balances.items():
        free = float(balance.get("Free", 0))
        locked = float(balance.get("Lock", 0))
        total_balance = free + locked
        
        if total_balance <= 0:
            continue
            
        print(f"Processing asset: {asset}, balance: {total_balance}")
            
        # USD is already in USD
        if asset == "USD":
            total_value += total_balance
            print(f"  USD: {total_balance} -> total: {total_value}")
            continue
            
        # For crypto assets, get current price
        pair = f"{asset}/USD"
        ticker = get_ticker(pair)
        
        if ticker and "LastPrice" in ticker:
            price = float(ticker["LastPrice"])
            asset_value = total_balance * price
            total_value += asset_value
            print(f"  {asset}: {total_balance} * {price} = {asset_value} -> total: {total_value}")
        else:
            logging.warning(f"Could not get price for {asset}")
    
    print(f"Final portfolio value: ${total_value:,.2f}")
    return total_value

def get_available_symbols() -> List[str]:
    """Get list of available trading pairs"""
    info = get_exchange_info()
    if info and "TradePairs" in info:
        pairs = list(info["TradePairs"].keys())
        # Filter only tradable pairs
        tradable_pairs = [
            pair for pair in pairs 
            if info["TradePairs"][pair].get("CanTrade", False)
        ]
        return tradable_pairs
    return []

def test_api_connection():
    """Test all API endpoints with better error reporting"""
    print("Testing Roostoo API connection...")
    
    # Test server time
    server_time = check_server_time()
    if server_time:
        print(f"✓ Server time: {server_time.get('ServerTime')}")
    else:
        print("✗ Failed to get server time")
        return False
    
    # Test exchange info
    exchange_info = get_exchange_info()
    if exchange_info:
        pairs = list(exchange_info.get("TradePairs", {}).keys())
        print(f"✓ Exchange info loaded: {len(pairs)} pairs available")
        print(f"  First 10 pairs: {pairs[:10]}")
        
        # Show which pairs are tradable
        tradable_pairs = [
            pair for pair in pairs 
            if exchange_info["TradePairs"][pair].get("CanTrade", False)
        ]
        print(f"  Tradable pairs: {len(tradable_pairs)}")
        print(f"  Sample tradable: {tradable_pairs[:5]}")
    else:
        print("✗ Failed to get exchange info")
        return False
    
    # Test balance with detailed error reporting
    print("Testing balance endpoint...")
    balance = get_balance()
    if balance:
        print(f"✓ Balance loaded: {len(balance)} assets")
        for asset, bal in list(balance.items())[:5]:
            print(f"  {asset}: Free={bal.get('Free', 0)}, Locked={bal.get('Lock', 0)}")
    else:
        print("✗ Failed to get balance - this may be normal if no trades yet")
        # Don't return False here - balance might be empty initially
    
    # Test ticker
    if exchange_info:
        # Test with first available pair
        first_pair = list(exchange_info.get("TradePairs", {}).keys())[0]
        ticker = get_ticker(first_pair)
        if ticker:
            print(f"✓ Ticker working - {first_pair}: ${ticker.get('LastPrice', 'N/A')}")
        else:
            print(f"✗ Failed to get ticker for {first_pair}")
            return False
    
    print("✓ Basic API connectivity confirmed!")
    return True

if __name__ == "__main__":
    test_api_connection()