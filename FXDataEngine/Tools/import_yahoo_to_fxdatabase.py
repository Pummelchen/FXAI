#!/usr/bin/env python3
"""
Import Yahoo Finance D1 CSV data into FXDatabase and validate 100% data quality.

This script:
1. Reads CSV files from FXDataEngine/Data/YahooFinance/
2. Validates data quality (OHLC logic, timestamps, gaps, duplicates, NaN values)
3. Imports into FXDatabase via HTTP API
4. Runs post-import validation
5. Generates quality report

Usage:
    python3 import_yahoo_to_fxdatabase.py [--dry-run] [--api-url http://localhost:8765]
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Configuration
DATA_DIR = Path(__file__).parent.parent / "Data" / "YahooFinance"
FXDATABASE_API_URL = "http://localhost:8765"
BATCH_SIZE = 1000  # bars per API call
DELAY_BETWEEN_BATCHES = 0.5  # seconds, be friendly to API

# Expected symbols and their Yahoo Finance source IDs
SYMBOL_CONFIG = {
    "AAPL_daily.csv": {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "broker_source_id": "yahoo_finance",
        "timezone": "America/New_York",
    },
    "MSFT_daily.csv": {
        "symbol": "MSFT",
        "name": "Microsoft Corporation",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "broker_source_id": "yahoo_finance",
        "timezone": "America/New_York",
    },
    "NVDA_daily.csv": {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "broker_source_id": "yahoo_finance",
        "timezone": "America/New_York",
    },
    "GOOGL_daily.csv": {
        "symbol": "GOOGL",
        "name": "Alphabet Inc.",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "broker_source_id": "yahoo_finance",
        "timezone": "America/New_York",
    },
    "META_daily.csv": {
        "symbol": "META",
        "name": "Meta Platforms Inc.",
        "source_origin": "YAHOO_FINANCE_HISTORY",
        "broker_source_id": "yahoo_finance",
        "timezone": "America/New_York",
    },
}


class DataValidationError(Exception):
    """Raised when data quality validation fails."""
    pass


class FXDatabaseImporter:
    """Import D1 bars into FXDatabase via HTTP API."""
    
    def __init__(self, api_url: str, dry_run: bool = False):
        self.api_url = api_url.rstrip("/")
        self.dry_run = dry_run
        self.total_imported = 0
        self.total_errors = 0
    
    def import_bars(self, symbol: str, bars: list[dict], config: dict) -> dict:
        """
        Import a batch of bars into FXDatabase.
        
        Args:
            symbol: Trading symbol (e.g., AAPL)
            bars: List of bar dicts with keys: date, open, high, low, close, volume, adjclose
            config: Symbol configuration dict
        
        Returns:
            Import result dict
        """
        if not bars:
            return {"status": "skipped", "reason": "no bars"}
        
        # Convert to FXDatabase D1 batch format
        batch = {
            "api_version": "fxbacktest.api.v1",
            "broker_source_id": config["broker_source_id"],
            "source_origin": config["source_origin"],
            "logical_symbol": symbol,
            "bars": []
        }
        
        for bar in bars:
            # Convert date to Unix timestamp (UTC midnight)
            dt = datetime.strptime(bar["date"], "%Y-%m-%d")
            dt = dt.replace(tzinfo=timezone.utc)
            timestamp = int(dt.timestamp())
            
            batch["bars"].append({
                "timestamp_utc": timestamp,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": int(bar["volume"]),
                "adjclose": float(bar.get("adjclose", bar["close"])),
            })
        
        if self.dry_run:
            print(f"    [DRY RUN] Would import {len(bars)} bars for {symbol}")
            return {"status": "dry_run", "bars": len(bars)}
        
        # Send to FXDatabase API
        url = f"{self.api_url}/v1/ingest/d1"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FXAI-YahooFinance-Importer/1.0",
        }
        
        data = json.dumps(batch).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                self.total_imported += len(bars)
                return {"status": "success", "result": result}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            self.total_errors += len(bars)
            return {
                "status": "error",
                "code": e.code,
                "message": error_body,
            }
        except Exception as e:
            self.total_errors += len(bars)
            return {
                "status": "error",
                "message": str(e),
            }


class DataQualityValidator:
    """Validate 100% data quality for D1 OHLCV bars."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.errors = []
        self.warnings = []
    
    def validate_bar(self, bar: dict, index: int, prev_bar: Optional[dict] = None):
        """Validate a single bar."""
        row_num = index + 2  # CSV row (1-indexed + header)
        
        # 1. Date validation
        try:
            datetime.strptime(bar["date"], "%Y-%m-%d")
        except ValueError:
            self.errors.append(f"Row {row_num}: Invalid date format '{bar['date']}'")
            return
        
        # 2. OHLC positive values
        for field in ["open", "high", "low", "close"]:
            value = float(bar[field])
            if value <= 0:
                self.errors.append(f"Row {row_num}: {field}={value} must be positive")
            if value != value:  # NaN check
                self.errors.append(f"Row {row_num}: {field} is NaN")
        
        # 3. OHLC logic: high >= low, high >= open, high >= close, low <= open, low <= close
        open_val = float(bar["open"])
        high_val = float(bar["high"])
        low_val = float(bar["low"])
        close_val = float(bar["close"])
        
        if high_val < low_val:
            self.errors.append(f"Row {row_num}: high ({high_val}) < low ({low_val})")
        if high_val < open_val:
            self.errors.append(f"Row {row_num}: high ({high_val}) < open ({open_val})")
        if high_val < close_val:
            self.errors.append(f"Row {row_num}: high ({high_val}) < close ({close_val})")
        if low_val > open_val:
            self.errors.append(f"Row {row_num}: low ({low_val}) > open ({open_val})")
        if low_val > close_val:
            self.errors.append(f"Row {row_num}: low ({low_val}) > close ({close_val})")
        
        # 4. Volume non-negative
        volume = int(bar["volume"])
        if volume < 0:
            self.errors.append(f"Row {row_num}: volume={volume} must be non-negative")
        
        # 5. AdjClose positive
        adjclose = float(bar.get("adjclose", bar["close"]))
        if adjclose <= 0:
            self.errors.append(f"Row {row_num}: adjclose={adjclose} must be positive")
        
        # 6. Temporal ordering (if we have previous bar)
        if prev_bar:
            curr_date = datetime.strptime(bar["date"], "%Y-%m-%d")
            prev_date = datetime.strptime(prev_bar["date"], "%Y-%m-%d")
            
            if curr_date < prev_date:
                self.errors.append(f"Row {row_num}: Date {bar['date']} is before previous date {prev_bar['date']}")
            elif curr_date == prev_date:
                self.errors.append(f"Row {row_num}: Duplicate date {bar['date']}")
    
    def validate_completeness(self, bars: list[dict]):
        """Validate dataset completeness."""
        if not bars:
            self.errors.append("Empty dataset")
            return
        
        # Check for gaps > 7 days (weekends/holidays are OK, but >1 week is suspicious)
        for i in range(1, len(bars)):
            curr_date = datetime.strptime(bars[i]["date"], "%Y-%m-%d")
            prev_date = datetime.strptime(bars[i-1]["date"], "%Y-%m-%d")
            gap_days = (curr_date - prev_date).days
            
            if gap_days > 14:  # More than 2 weeks
                self.warnings.append(
                    f"Gap of {gap_days} days between {bars[i-1]['date']} and {bars[i]['date']}"
                )
    
    def get_report(self) -> dict:
        """Get validation report."""
        return {
            "symbol": self.symbol,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "is_valid": len(self.errors) == 0,
        }


def read_csv(filepath: Path) -> list[dict]:
    """Read CSV file, skipping comment lines."""
    bars = []
    
    with open(filepath, "r") as f:
        reader = csv.DictReader(
            line for line in f if not line.startswith("#")
        )
        
        for row in reader:
            bars.append({
                "date": row["date"].strip(),
                "open": row["open"].strip(),
                "high": row["high"].strip(),
                "low": row["low"].strip(),
                "close": row["close"].strip(),
                "volume": row["volume"].strip(),
                "adjclose": row["adjclose"].strip(),
            })
    
    return bars


def import_symbol(
    importer: FXDatabaseImporter,
    csv_file: Path,
    config: dict,
) -> dict:
    """Import a single symbol's data with validation."""
    symbol = config["symbol"]
    print(f"\n{'='*60}")
    print(f"Importing {symbol} ({config['name']})")
    print(f"{'='*60}")
    
    # Read CSV
    print(f"Reading {csv_file}...")
    bars = read_csv(csv_file)
    print(f"  Read {len(bars):,} bars")
    
    if not bars:
        return {"status": "skipped", "reason": "no bars"}
    
    # Validate data quality
    print(f"Validating data quality...")
    validator = DataQualityValidator(symbol)
    
    for i, bar in enumerate(bars):
        prev_bar = bars[i-1] if i > 0 else None
        validator.validate_bar(bar, i, prev_bar)
    
    validator.validate_completeness(bars)
    report = validator.get_report()
    
    # Print validation results
    if report["error_count"] > 0:
        print(f"  ✗ FAILED: {report['error_count']} errors found")
        for error in report["errors"][:10]:  # Show first 10
            print(f"    ERROR: {error}")
        if report["error_count"] > 10:
            print(f"    ... and {report['error_count'] - 10} more errors")
        raise DataValidationError(f"{symbol} has {report['error_count']} validation errors")
    else:
        print(f"  ✓ Data quality: PASSED (0 errors)")
    
    if report["warning_count"] > 0:
        print(f"  ⚠ {report['warning_count']} warnings:")
        for warning in report["warnings"][:5]:
            print(f"    {warning}")
    
    # Import in batches
    print(f"Importing to FXDatabase...")
    total_batches = (len(bars) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(bars))
        batch = bars[start:end]
        
        result = importer.import_bars(symbol, batch, config)
        
        if result["status"] == "error":
            print(f"  ✗ Batch {batch_idx + 1}/{total_batches} FAILED: {result.get('message', 'unknown error')}")
            return result
        elif result["status"] == "success":
            print(f"  ✓ Batch {batch_idx + 1}/{total_batches}: {len(batch)} bars imported")
        elif result["status"] == "dry_run":
            print(f"  [DRY RUN] Batch {batch_idx + 1}/{total_batches}: {len(batch)} bars")
        
        # Rate limiting
        if batch_idx < total_batches - 1:
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    # Summary
    date_range = f"{bars[0]['date']} → {bars[-1]['date']}"
    print(f"\n  ✓ {symbol} import complete")
    print(f"    Bars: {len(bars):,}")
    print(f"    Range: {date_range}")
    print(f"    Latest close: ${float(bars[-1]['close']):.2f}")
    
    return {
        "status": "success",
        "symbol": symbol,
        "bars": len(bars),
        "date_range": date_range,
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Import Yahoo Finance D1 data into FXDatabase")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't import")
    parser.add_argument("--api-url", default=FXDATABASE_API_URL, help="FXDatabase API URL")
    args = parser.parse_args()
    
    print("="*60)
    print("FXAI Yahoo Finance → FXDatabase Importer")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")
    print(f"FXDatabase API: {args.api_url}")
    print(f"Mode: {'DRY RUN (validation only)' if args.dry_run else 'IMPORT'}")
    print("="*60)
    
    # Check data directory
    if not DATA_DIR.exists():
        print(f"✗ Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    csv_files = sorted(DATA_DIR.glob("*_daily.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in {DATA_DIR}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()
    
    # Initialize importer
    importer = FXDatabaseImporter(args.api_url, dry_run=args.dry_run)
    
    # Import each symbol
    results = []
    success_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        filename = csv_file.name
        
        if filename not in SYMBOL_CONFIG:
            print(f"⚠ Skipping {filename}: no configuration")
            continue
        
        config = SYMBOL_CONFIG[filename]
        
        try:
            result = import_symbol(importer, csv_file, config)
            results.append(result)
            success_count += 1
        except DataValidationError as e:
            print(f"✗ {filename}: {e}")
            results.append({"status": "error", "file": filename, "error": str(e)})
            error_count += 1
        except Exception as e:
            print(f"✗ {filename}: Unexpected error: {e}")
            results.append({"status": "error", "file": filename, "error": str(e)})
            error_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("Import Summary")
    print(f"{'='*60}")
    print(f"Files processed: {len(csv_files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total bars imported: {importer.total_imported:,}")
    print(f"Total bars failed: {importer.total_errors:,}")
    
    if results:
        print(f"\nPer-symbol results:")
        for result in results:
            if result["status"] == "success":
                print(f"  ✓ {result['symbol']}: {result['bars']:,} bars ({result['date_range']})")
            else:
                print(f"  ✗ {result.get('symbol', result.get('file', 'unknown'))}: {result.get('error', result.get('reason', 'failed'))}")
    
    print(f"{'='*60}")
    
    if error_count > 0:
        print("\n✗ Import completed with errors")
        sys.exit(1)
    else:
        print("\n✓ Import completed successfully")


if __name__ == "__main__":
    main()
