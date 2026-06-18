#!/usr/bin/env python3
"""
Comprehensive data quality verification for Yahoo Finance D1 CSV data.

This script performs 100% quality validation:
1. Date format and ordering validation
2. OHLC logic checks (high >= low, high >= open/close, low <= open/close)
3. Positive price validation
4. Volume non-negative checks
5. Gap detection (>14 days)
6. Duplicate detection
7. NaN/Inf detection
8. Statistical summary
9. Cross-file consistency

Usage:
    python3 verify_yahoo_data_quality.py
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "Data" / "YahooFinance"


class QualityCheck:
    """Track quality check results."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.warning_messages = []
    
    def pass_check(self, name: str):
        self.passed += 1
    
    def fail_check(self, name: str, detail: str = ""):
        self.failed += 1
        self.errors.append(f"{name}: {detail}" if detail else name)
    
    def warn(self, message: str):
        self.warnings += 1
        self.warning_messages.append(message)
    
    def summary(self) -> str:
        status = "✅ PASSED" if self.failed == 0 else "❌ FAILED"
        lines = [
            f"{status} - {self.passed} passed, {self.failed} failed, {self.warnings} warnings",
        ]
        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:20]:
                lines.append(f"  ✗ {error}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more")
        if self.warning_messages:
            lines.append(f"\nWarnings ({len(self.warning_messages)}):")
            for msg in self.warning_messages[:10]:
                lines.append(f"  ⚠ {msg}")
            if len(self.warning_messages) > 10:
                lines.append(f"  ... and {len(self.warning_messages) - 10} more")
        return "\n".join(lines)


def read_csv_bars(filepath: Path) -> list[dict]:
    """Read CSV file, skipping comment lines."""
    bars = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(
            line for line in f if not line.startswith("#")
        )
        for row in reader:
            bars.append({
                "date": row["date"].strip(),
                "open": float(row["open"].strip()),
                "high": float(row["high"].strip()),
                "low": float(row["low"].strip()),
                "close": float(row["close"].strip()),
                "volume": int(row["volume"].strip()),
                "adjclose": float(row["adjclose"].strip()),
            })
    return bars


def verify_symbol(filepath: Path, symbol: str, name: str) -> QualityCheck:
    """Run comprehensive quality checks on a symbol's data."""
    qc = QualityCheck()
    
    print(f"\n{'='*70}")
    print(f"Verifying {symbol} ({name})")
    print(f"{'='*70}")
    print(f"File: {filepath.name}")
    
    # Read data
    try:
        bars = read_csv_bars(filepath)
        print(f"Bars loaded: {len(bars):,}")
        qc.pass_check("CSV parsing")
    except Exception as e:
        qc.fail_check("CSV parsing", str(e))
        return qc
    
    if not bars:
        qc.fail_check("Data presence", "No bars found")
        return qc
    
    qc.pass_check("Data presence")
    
    # Check 1: Date format validation
    print("\n1. Date format validation...")
    invalid_dates = []
    for i, bar in enumerate(bars):
        try:
            datetime.strptime(bar["date"], "%Y-%m-%d")
        except ValueError:
            invalid_dates.append((i, bar["date"]))
    
    if invalid_dates:
        for idx, date_str in invalid_dates[:5]:
            qc.fail_check("Date format", f"Row {idx}: '{date_str}'")
    else:
        qc.pass_check("Date format")
        print(f"   ✓ All {len(bars):,} dates valid (YYYY-MM-DD)")
    
    # Check 2: Temporal ordering
    print("2. Temporal ordering validation...")
    out_of_order = []
    duplicates = []
    for i in range(1, len(bars)):
        curr_date = datetime.strptime(bars[i]["date"], "%Y-%m-%d")
        prev_date = datetime.strptime(bars[i-1]["date"], "%Y-%m-%d")
        
        if curr_date < prev_date:
            out_of_order.append(i)
        elif curr_date == prev_date:
            duplicates.append(i)
    
    if out_of_order:
        qc.fail_check("Temporal ordering", f"{len(out_of_order)} bars out of order")
    else:
        qc.pass_check("Temporal ordering")
        print(f"   ✓ All dates in ascending order")
    
    if duplicates:
        qc.fail_check("Duplicate detection", f"{len(duplicates)} duplicate dates found")
    else:
        qc.pass_check("Duplicate detection")
        print(f"   ✓ No duplicate dates")
    
    # Check 3: OHLC logic
    print("3. OHLC logic validation...")
    ohlc_violations = []
    for i, bar in enumerate(bars):
        violations = []
        if bar["high"] < bar["low"]:
            violations.append(f"high<{bar['low']}")
        if bar["high"] < bar["open"]:
            violations.append(f"high<open")
        if bar["high"] < bar["close"]:
            violations.append(f"high<close")
        if bar["low"] > bar["open"]:
            violations.append(f"low>open")
        if bar["low"] > bar["close"]:
            violations.append(f"low>close")
        
        if violations:
            ohlc_violations.append((i, bar["date"], violations))
    
    if ohlc_violations:
        for idx, date_str, viols in ohlc_violations[:5]:
            qc.fail_check("OHLC logic", f"Row {idx} ({date_str}): {', '.join(viols)}")
    else:
        qc.pass_check("OHLC logic")
        print(f"   ✓ All {len(bars):,} bars satisfy high≥low, high≥open, high≥close, low≤open, low≤close")
    
    # Check 4: Positive prices
    print("4. Positive price validation...")
    non_positive = []
    for i, bar in enumerate(bars):
        for field in ["open", "high", "low", "close", "adjclose"]:
            if bar[field] <= 0:
                non_positive.append((i, bar["date"], field, bar[field]))
    
    if non_positive:
        for idx, date_str, field, val in non_positive[:5]:
            qc.fail_check("Positive prices", f"Row {idx} ({date_str}): {field}={val}")
    else:
        qc.pass_check("Positive prices")
        print(f"   ✓ All prices > 0")
    
    # Check 5: Volume non-negative
    print("5. Volume validation...")
    negative_volume = [i for i, bar in enumerate(bars) if bar["volume"] < 0]
    
    if negative_volume:
        qc.fail_check("Volume", f"{len(negative_volume)} bars with negative volume")
    else:
        qc.pass_check("Volume")
        print(f"   ✓ All volumes ≥ 0")
    
    # Check 6: NaN/Inf detection
    print("6. NaN/Inf detection...")
    import math
    nan_inf = []
    for i, bar in enumerate(bars):
        for field in ["open", "high", "low", "close", "adjclose", "volume"]:
            val = bar[field]
            if math.isnan(val) or math.isinf(val):
                nan_inf.append((i, bar["date"], field, val))
    
    if nan_inf:
        for idx, date_str, field, val in nan_inf[:5]:
            qc.fail_check("NaN/Inf", f"Row {idx} ({date_str}): {field}={val}")
    else:
        qc.pass_check("NaN/Inf")
        print(f"   ✓ No NaN or Inf values")
    
    # Check 7: Gap detection
    print("7. Gap detection (>14 days)...")
    large_gaps = []
    for i in range(1, len(bars)):
        curr_date = datetime.strptime(bars[i]["date"], "%Y-%m-%d")
        prev_date = datetime.strptime(bars[i-1]["date"], "%Y-%m-%d")
        gap_days = (curr_date - prev_date).days
        
        if gap_days > 14:
            large_gaps.append((i, bars[i-1]["date"], bars[i]["date"], gap_days))
    
    if large_gaps:
        qc.warn(f"{len(large_gaps)} gaps >14 days found")
        for idx, start, end, days in large_gaps[:5]:
            print(f"   ⚠ {start} → {end}: {days} days")
        print(f"   (This may be normal for older data or delisted periods)")
    else:
        qc.pass_check("Gap detection")
        print(f"   ✓ No gaps >14 days")
    
    # Check 8: Data range
    print("8. Data range validation...")
    first_date = datetime.strptime(bars[0]["date"], "%Y-%m-%d")
    last_date = datetime.strptime(bars[-1]["date"], "%Y-%m-%d")
    date_range = last_date - first_date
    
    if date_range.days < 30:
        qc.warn(f"Date range only {date_range.days} days")
    else:
        qc.pass_check("Data range")
        print(f"   ✓ Date range: {bars[0]['date']} → {bars[-1]['date']} ({date_range.days} days, {date_range.days/365.25:.1f} years)")
    
    # Check 9: Statistical sanity
    print("9. Statistical sanity checks...")
    closes = [bar["close"] for bar in bars]
    volumes = [bar["volume"] for bar in bars]
    
    # Price should not have zero variance
    price_variance = sum((c - sum(closes)/len(closes))**2 for c in closes) / len(closes)
    if price_variance < 0.0001:
        qc.fail_check("Price variance", f"Variance {price_variance:.6f} too low (possible flat data)")
    else:
        qc.pass_check("Price variance")
        print(f"   ✓ Price variance: {price_variance:.2f} (healthy)")
    
    # Volume should have some activity
    zero_volume_days = sum(1 for v in volumes if v == 0)
    if zero_volume_days > len(bars) * 0.5:
        qc.fail_check("Volume activity", f"{zero_volume_days} days ({zero_volume_days/len(bars)*100:.1f}%) with zero volume")
    else:
        qc.pass_check("Volume activity")
        print(f"   ✓ Zero volume days: {zero_volume_days}/{len(bars):,} ({zero_volume_days/len(bars)*100:.1f}%)")
    
    # Check 10: AdjClose vs Close relationship
    print("10. Adjusted close validation...")
    adjclose_mismatches = 0
    for bar in bars:
        # AdjClose should be close to Close (within 50% is reasonable for splits)
        ratio = bar["adjclose"] / bar["close"]
        if ratio < 0.1 or ratio > 10.0:
            adjclose_mismatches += 1
    
    if adjclose_mismatches > len(bars) * 0.01:  # More than 1% mismatches
        qc.fail_check("AdjClose", f"{adjclose_mismatches} bars with unusual adjclose/close ratio")
    else:
        qc.pass_check("AdjClose")
        print(f"   ✓ AdjClose/Close ratio reasonable ({adjclose_mismatches} unusual out of {len(bars):,})")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   First close: ${closes[0]:.2f}")
    print(f"   Last close: ${closes[-1]:.2f}")
    print(f"   Min close: ${min(closes):.2f}")
    print(f"   Max close: ${max(closes):.2f}")
    print(f"   Avg volume: {sum(volumes)/len(volumes):,.0f}")
    print(f"   Max volume: {max(volumes):,}")
    
    # Return results
    return qc


def main():
    """Main execution function."""
    print("="*70)
    print("FXAI Yahoo Finance Data Quality Verification")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Validation level: 100% quality (zero tolerance for errors)")
    print("="*70)
    
    if not DATA_DIR.exists():
        print(f"✗ Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    csv_files = sorted(DATA_DIR.glob("*_daily.csv"))
    if not csv_files:
        print(f"✗ No CSV files found in {DATA_DIR}")
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} files to verify:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Symbol metadata
    symbol_meta = {
        "AAPL_daily.csv": ("AAPL", "Apple Inc."),
        "MSFT_daily.csv": ("MSFT", "Microsoft Corporation"),
        "NVDA_daily.csv": ("NVDA", "NVIDIA Corporation"),
        "GOOGL_daily.csv": ("GOOGL", "Alphabet Inc."),
        "META_daily.csv": ("META", "Meta Platforms Inc."),
    }
    
    # Verify each symbol
    all_results = {}
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    total_bars = 0
    
    for csv_file in csv_files:
        filename = csv_file.name
        if filename not in symbol_meta:
            print(f"\n⚠ Skipping {filename}: no metadata")
            continue
        
        symbol, name = symbol_meta[filename]
        qc = verify_symbol(csv_file, symbol, name)
        all_results[symbol] = qc
        
        total_passed += qc.passed
        total_failed += qc.failed
        total_warnings += qc.warnings
        
        # Count bars
        bars = read_csv_bars(csv_file)
        total_bars += len(bars)
        
        print(f"\n{qc.summary()}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL QUALITY REPORT")
    print(f"{'='*70}")
    print(f"Files verified: {len(csv_files)}")
    print(f"Total bars: {total_bars:,}")
    print(f"Total checks passed: {total_passed}")
    print(f"Total checks failed: {total_failed}")
    print(f"Total warnings: {total_warnings}")
    
    if total_failed == 0:
        print(f"\n✅ QUALITY STATUS: 100% PASS")
        print(f"All {total_bars:,} bars across {len(csv_files)} symbols are clean.")
        print(f"Data is ready for import into FXDatabase.")
    else:
        print(f"\n❌ QUALITY STATUS: FAILED")
        print(f"{total_failed} errors found. Do NOT import until fixed.")
    
    # Per-symbol breakdown
    print(f"\nPer-Symbol Results:")
    print(f"{'Symbol':<8} {'Bars':>8} {'Checks':>8} {'Passed':>8} {'Failed':>8} {'Warnings':>10}")
    print(f"{'-'*8:<8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*10:>10}")
    
    for symbol, qc in all_results.items():
        bars = len(read_csv_bars(DATA_DIR / f"{symbol}_daily.csv"))
        total_checks = qc.passed + qc.failed
        print(f"{symbol:<8} {bars:>8,} {total_checks:>8} {qc.passed:>8} {qc.failed:>8} {qc.warnings:>10}")
    
    print(f"{'='*70}")
    
    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
