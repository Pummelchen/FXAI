#!/usr/bin/env python3
"""
Pull historical daily OHLCV data from Yahoo Finance for major tech stocks.

This script is designed to be friendly to Yahoo Finance:
- Sequential requests (one symbol at a time)
- 3-second delay between requests
- Goes back as far as possible (1980s-1990s depending on stock)
- Saves results to CSV files in FXAI data directory

Symbols to download:
- AAPL (Apple) - IPO: 1980-12-12
- MSFT (Microsoft) - IPO: 1986-03-13
- NVDA (NVIDIA) - IPO: 1999-01-22
- GOOGL (Alphabet/Google) - IPO: 2004-08-19
- META (Meta/Facebook) - IPO: 2012-05-18

Usage:
    python3 fetch_tech_stocks.py
"""

import time
import json
import ssl
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
import sys

# Try to use certifi for SSL certificates (fixes macOS Python SSL issues)
try:
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # Fallback: create default context (may fail on some macOS Python installations)
    ssl_context = ssl.create_default_context()

# Configuration
SYMBOLS = [
    ("AAPL", "Apple", "1980-12-12"),
    ("MSFT", "Microsoft", "1986-03-13"),
    ("NVDA", "NVIDIA", "1999-01-22"),
    ("GOOGL", "Alphabet (Google)", "2004-08-19"),
    ("META", "Meta (Facebook)", "2012-05-18"),
]

# Yahoo Finance API endpoint
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Rate limiting - be friendly to Yahoo
DELAY_BETWEEN_REQUESTS = 3  # seconds
RETRY_DELAY = 5  # seconds
MAX_RETRIES = 3

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "Data" / "YahooFinance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def date_to_unix(date_str: str) -> int:
    """Convert YYYY-MM-DD to Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp())


def fetch_yahoo_data(symbol: str, from_date: str) -> dict:
    """
    Fetch daily OHLCV data from Yahoo Finance chart API.
    
    Args:
        symbol: Yahoo Finance symbol (e.g., AAPL, MSFT)
        from_date: Start date in YYYY-MM-DD format
    
    Returns:
        Parsed JSON response dict
    
    Raises:
        urllib.error.HTTPError: If HTTP request fails
        Exception: If parsing fails
    """
    period1 = date_to_unix(from_date)
    period2 = int(datetime.now().timestamp())
    
    url = (
        f"{YAHOO_CHART_URL.format(symbol=symbol)}"
        f"?period1={period1}"
        f"&period2={period2}"
        f"&interval=1d"
        f"&events=history"
        f"&includeAdjustedClose=true"
    )
    
    # Set headers to be a good citizen
    headers = {
        "User-Agent": "FXAI-Research/1.0 (Educational/Research Use)",
        "Accept": "application/json",
    }
    
    request = urllib.request.Request(url, headers=headers)
    
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES}...")
            with urllib.request.urlopen(request, timeout=30, context=ssl_context) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            print(f"  HTTP Error {e.code}: {e.reason}")
            if e.code == 429:  # Rate limited
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif e.code >= 500:  # Server error, retry
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Server error. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise
    
    raise Exception(f"Failed to fetch data for {symbol} after {MAX_RETRIES} attempts")


def extract_bars(response: dict, symbol: str) -> list[dict]:
    """
    Extract OHLCV bars from Yahoo Finance chart response.
    
    Args:
        response: Parsed JSON response
        symbol: Symbol name for error messages
    
    Returns:
        List of bar dicts with keys: date, open, high, low, close, volume, adjclose
    """
    chart = response.get("chart", {})
    
    # Check for errors
    error = chart.get("error")
    if error:
        raise Exception(f"Yahoo API error for {symbol}: {error}")
    
    results = chart.get("result", [])
    if not results:
        raise Exception(f"No chart results for {symbol}")
    
    result = results[0]
    timestamps = result.get("timestamp", [])
    
    indicators = result.get("indicators", {})
    quote = indicators.get("quote", [{}])[0]
    adjclose_data = indicators.get("adjclose", [{}])[0]
    
    opens = quote.get("open", [])
    highs = quote.get("high", [])
    lows = quote.get("low", [])
    closes = quote.get("close", [])
    volumes = quote.get("volume", [])
    adjcloses = adjclose_data.get("adjclose", [])
    
    bars = []
    for i, ts in enumerate(timestamps):
        # Skip bars with missing data
        if (opens[i] is None or highs[i] is None or 
            lows[i] is None or closes[i] is None):
            continue
        
        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        
        bar = {
            "date": date_str,
            "open": round(opens[i], 2),
            "high": round(highs[i], 2),
            "low": round(lows[i], 2),
            "close": round(closes[i], 2),
            "volume": int(volumes[i]) if volumes[i] is not None else 0,
        }
        
        if i < len(adjcloses) and adjcloses[i] is not None:
            bar["adjclose"] = round(adjcloses[i], 2)
        else:
            bar["adjclose"] = bar["close"]
        
        bars.append(bar)
    
    return bars


def save_to_csv(bars: list[dict], symbol: str, company_name: str):
    """
    Save bars to CSV file.
    
    Args:
        bars: List of bar dicts
        symbol: Stock symbol
        company_name: Company name for header comment
    """
    output_file = OUTPUT_DIR / f"{symbol}_daily.csv"
    
    with open(output_file, "w") as f:
        # Header comment
        f.write(f"# {company_name} ({symbol}) - Daily OHLCV Data\n")
        f.write(f"# Source: Yahoo Finance\n")
        f.write(f"# Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Bars: {len(bars)}\n")
        if bars:
            f.write(f"# Date Range: {bars[0]['date']} to {bars[-1]['date']}\n")
        f.write("#\n")
        
        # CSV header
        f.write("date,open,high,low,close,volume,adjclose\n")
        
        # Data rows
        for bar in bars:
            f.write(
                f"{bar['date']},"
                f"{bar['open']},"
                f"{bar['high']},"
                f"{bar['low']},"
                f"{bar['close']},"
                f"{bar['volume']},"
                f"{bar['adjclose']}\n"
            )
    
    print(f"  ✓ Saved to {output_file}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("FXAI Yahoo Finance Data Fetcher")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Symbols: {len(SYMBOLS)} tech stocks")
    print(f"Rate limiting: {DELAY_BETWEEN_REQUESTS}s between requests")
    print("=" * 60)
    print()
    
    total_bars = 0
    success_count = 0
    error_count = 0
    
    for i, (symbol, company_name, ipo_date) in enumerate(SYMBOLS, 1):
        print(f"[{i}/{len(SYMBOLS)}] Fetching {symbol} ({company_name})...")
        print(f"  IPO date: {ipo_date}")
        
        try:
            # Fetch data
            response = fetch_yahoo_data(symbol, ipo_date)
            
            # Extract bars
            bars = extract_bars(response, symbol)
            
            if not bars:
                print(f"  ⚠ No bars found for {symbol}")
                error_count += 1
                continue
            
            # Save to CSV
            save_to_csv(bars, symbol, company_name)
            
            # Summary
            total_bars += len(bars)
            success_count += 1
            print(f"  📊 {len(bars)} bars downloaded")
            print(f"     Range: {bars[0]['date']} → {bars[-1]['date']}")
            print(f"     Latest close: ${bars[-1]['close']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
            error_count += 1
        
        # Rate limiting delay (except after last symbol)
        if i < len(SYMBOLS):
            print(f"  Waiting {DELAY_BETWEEN_REQUESTS}s before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        print()
    
    # Final summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Successful: {success_count}/{len(SYMBOLS)}")
    print(f"Errors: {error_count}/{len(SYMBOLS)}")
    print(f"Total bars: {total_bars:,}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
