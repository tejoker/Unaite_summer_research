#!/bin/bash

# Quick verification script to check if the p-value fix is working
# Usage: bash verify_p_values.sh <test_output_log_file>

LOG_FILE=${1:-"test_output.log"}

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Error: Log file not found: $LOG_FILE"
    echo "Usage: bash verify_p_values.sh <test_output_log_file>"
    exit 1
fi

echo "=============================================================================="
echo "Verifying p-value consistency in test run"
echo "=============================================================================="

echo -e "\n📋 Searching for p-value declarations..."

# Extract all p-value declarations
grep "CRITICAL PARAMETER: max_lag p" "$LOG_FILE" > /tmp/p_values.txt

if [ ! -s /tmp/p_values.txt ]; then
    echo "⚠️  Warning: No p-value declarations found in log file"
    echo "   Make sure you're using the updated dbn_dynotears_fixed_lambda.py"
    exit 1
fi

echo -e "\n📊 Found p-value declarations:\n"
cat /tmp/p_values.txt | nl

echo -e "\n"

# Extract just the p values
P_VALUES=$(grep -oP "max_lag p = \K\d+" "$LOG_FILE")
UNIQUE_P_VALUES=$(echo "$P_VALUES" | sort -u)
COUNT=$(echo "$UNIQUE_P_VALUES" | wc -l)

echo "=============================================================================="
if [ "$COUNT" -eq 1 ]; then
    echo "✅ SUCCESS: All runs use the same p value!"
    echo "   p = $UNIQUE_P_VALUES"
    echo ""
    echo "   This confirms the fix is working. Both Golden and Anomaly runs"
    echo "   are now using identical lag parameters."
else
    echo "❌ FAILURE: Different p values detected!"
    echo "   Unique p values found:"
    echo "$UNIQUE_P_VALUES" | nl
    echo ""
    echo "   This suggests the preprocessing script is still recalculating lags"
    echo "   instead of using the provided Golden lags file."
fi
echo "=============================================================================="

echo -e "\n📋 Checking lags file usage..."
grep "Using pre-calculated lags from\|Will calculate new lags" "$LOG_FILE" | nl

echo -e "\n📋 Diagnostic results (Temporal Onset):"
grep "VERDICT (TEMPORAL ONSET)" "$LOG_FILE"
grep "VERDICT (MAX IMPACT)" "$LOG_FILE"

echo -e "\n✅ Verification complete!"

# Cleanup
rm -f /tmp/p_values.txt
