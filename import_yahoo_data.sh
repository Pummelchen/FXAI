#!/bin/bash
# Import Yahoo Finance data into FXDatabase
# Run this on the machine where FXDatabase and ClickHouse are running

set -e

echo "=============================================="
echo "FXAI Yahoo Finance Data Import"
echo "=============================================="

# Check if FXDatabase is running
if curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "✓ FXDatabase is running"
else
    echo "✗ FXDatabase is not running on port 8765"
    echo ""
    echo "Start FXDatabase first:"
    echo "  cd FXDatabase"
    echo "  swift run FXDatabase --serve"
    exit 1
fi

# Run the importer
python3 FXDataEngine/Tools/import_yahoo_to_fxdatabase.py

echo ""
echo "=============================================="
echo "Import complete!"
echo "=============================================="
