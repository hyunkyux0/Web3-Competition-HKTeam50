# binance_data_fetcher.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import pickle


class BinanceDataFetcher:
    """
    Fetches cryptocurrency OHLCV data from Binance public API.
    Includes caching to avoid re-downloading data.
    """

    def __init__(self, cache_dir='data_cache'):
        """Initialize the Binance data fetcher with caching."""
        self.base_url = "https://api.binance.com/api/v3"
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_filename(self, symbol, interval, start_date, end_date):
        """Generate cache filename based on parameters."""
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        return os.path.join(self.cache_dir, f"{symbol}_{interval}_{start_str}_{end_str}.pkl")

    def _load_from_cache(self, cache_file):
        """Load data from cache file."""
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded data from cache: {cache_file}")
            return data
        except Exception as e:
            print(f"⚠️  Could not load cache: {e}")
            return None

    def _save_to_cache(self, data, cache_file):
        """Save data to cache file."""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved data to cache: {cache_file}")
        except Exception as e:
            print(f"⚠️  Could not save cache: {e}")

    def fetch_ohlcv(self, symbol, interval, start_date, end_date, use_cache=True):
        """
        Fetch OHLCV data from Binance API with caching support.

        Parameters:
        symbol (str): Trading pair symbol (e.g., 'ETHUSDT')
        interval (str): Candle interval (e.g., '15m')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        use_cache (bool): Whether to use cached data if available

        Returns:
        pd.DataFrame: DataFrame with OHLCV data indexed by timestamp
        """

        # Check cache first
        cache_file = self._get_cache_filename(symbol, interval, start_date, end_date)

        if use_cache and os.path.exists(cache_file):
            print(f"\n{'=' * 70}")
            print("USING CACHED DATA")
            print("=" * 70)
            cached_data = self._load_from_cache(cache_file)
            if cached_data is not None:
                print(f"  Total candles: {len(cached_data)}")
                print(f"  Date range   : {cached_data.index[0]} to {cached_data.index[-1]}")
                print("=" * 70)
                return cached_data

        # Fetch from Binance API
        print(f"\n{'=' * 70}")
        print("FETCHING DATA FROM BINANCE API")
        print("=" * 70)

        endpoint = f"{self.base_url}/klines"

        # Convert dates to millisecond timestamps
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        all_candles = []
        current_start = start_ts

        # Binance has a limit of 1000 candles per request
        limit = 1000

        print(f"Symbol        : {symbol}")
        print(f"Interval      : {interval}")
        print(f"Start date    : {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End date      : {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        request_count = 0

        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit
            }

            try:
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                candles = response.json()

                if not candles:
                    break

                all_candles.extend(candles)
                request_count += 1

                # Update start time to the last candle's close time + 1ms
                current_start = candles[-1][6] + 1

                print(f"  Request {request_count}: Fetched {len(candles)} candles (Total: {len(all_candles)})")

                # Respect rate limits
                time.sleep(0.1)

                # If we got less than the limit, we've reached the end
                if len(candles) < limit:
                    break

            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching data from Binance API: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response status: {e.response.status_code}")
                    print(f"Response body: {e.response.text}")
                raise

        if not all_candles:
            raise ValueError("No data retrieved from Binance API")

        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Volume', 'Trades', 'Taker_Buy_Base',
            'Taker_Buy_Quote', 'Ignore'
        ])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['Open_Time'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Keep only OHLCV columns and convert to numeric
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)

        print("-" * 70)
        print(f"✓ Successfully fetched {len(df)} candles")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print("=" * 70)

        # Save to cache
        if use_cache:
            self._save_to_cache(df, cache_file)

        return df

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            print(f"✓ Cache cleared: {self.cache_dir}")

    def test_connection(self):
        """Test the API connection."""
        try:
            endpoint = f"{self.base_url}/ping"
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            print("✓ Successfully connected to Binance API")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Binance API: {e}")
            return False

    def get_server_time(self):
        """Get Binance server time."""
        try:
            endpoint = f"{self.base_url}/time"
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            server_time = response.json()['serverTime']
            dt = datetime.fromtimestamp(server_time / 1000)
            print(f"Binance server time: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            return dt
        except Exception as e:
            print(f"Error getting server time: {e}")
            return None