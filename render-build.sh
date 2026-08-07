#!/usr/bin/env bash
# ===================================================================
# Gold Scalper -- Render Build Script
#
# Render free tier: 512 MB RAM, shared CPU
# XGBoost candle models are committed to git; the build only verifies
# they are present (no on-build training, no torch/onnx runtime).
# ===================================================================
set -euo pipefail

echo "=============================================="
echo "  Gold Scalper -- Render Build"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "=============================================="

# -- Step 1: Install dependencies --
echo ""
echo "-> Step 1/4: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -- Step 2: Create required directories --
echo ""
echo "-> Step 2/4: Creating directories..."
mkdir -p data

# -- Step 3: Verify committed XGBoost candle models --
echo ""
echo "-> Step 3/4: Verifying XGBoost candle models..."
REQUIRED_MODELS=(
    "models/candle_h1/XAUUSD.joblib"
    "models/candle_h1/US100.joblib"
    "models/candle_h1/JP225.joblib"
    "models/candle_h1/DE40.joblib"
    "models/candle_h1/US500.joblib"
    "models/candle_h1/US30.joblib"
)
MISSING=0
for model in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "  MISS $model"
        MISSING=1
    else
        echo "  OK   $model"
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo "ERROR: committed candle models missing -- deploy cannot proceed."
    exit 1
fi

# -- Step 4: Set permissions --
echo ""
echo "-> Step 4/4: Setting permissions..."
chmod -R 755 models/ 2>/dev/null || true
chmod -R 755 data/ 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Build complete!"
echo "=============================================="
