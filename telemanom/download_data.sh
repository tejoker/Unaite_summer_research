#!/bin/bash
#
# Download and prepare Telemanom dataset for Tucker-CAM pipeline
#
# This script:
# 1. Downloads NASA Telemanom data from Kaggle
# 2. Converts .npy files to CSV format
# 3. Prepares clean datasets (NaN handling)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "TELEMANOM DATA DOWNLOAD AND PREPARATION"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Download NASA Telemanom dataset from Kaggle (~63MB)"
echo "  2. Convert train/test .npy files to CSV format"
echo "  3. Apply NaN handling to create clean datasets"
echo ""
echo "Requirements:"
echo "  - Kaggle API key configured (~/.kaggle/kaggle.json)"
echo "  - pip install kaggle"
echo ""
echo "================================================================================"
echo ""

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: Kaggle CLI not found!"
    echo ""
    echo "Please install it:"
    echo "  pip install kaggle"
    echo ""
    echo "And configure your API key:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Create new API token (downloads kaggle.json)"
    echo "  3. Move to ~/.kaggle/kaggle.json"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Check if API key is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "ERROR: Kaggle API key not found!"
    echo ""
    echo "Please configure your API key:"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Create new API token (downloads kaggle.json)"
    echo "  3. Move to ~/.kaggle/kaggle.json"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "[STEP 1/4] Downloading NASA Telemanom dataset from Kaggle"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Download and extract (using command from README.md)
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl && \
    mv nasa-anomaly-detection-dataset-smap-msl.zip data.zip && \
    unzip -o data.zip && \
    rm data.zip && \
    mv data/data tmp && \
    rm -r data && \
    mv tmp data

echo ""
echo "Download complete!"
echo ""

# Verify download
TRAIN_FILES=$(ls data/train/*.npy 2>/dev/null | wc -l)
TEST_FILES=$(ls data/test/*.npy 2>/dev/null | wc -l)

echo "Files extracted:"
echo "  Train: $TRAIN_FILES .npy files"
echo "  Test:  $TEST_FILES .npy files"
echo ""

if [ "$TRAIN_FILES" -eq 0 ] || [ "$TEST_FILES" -eq 0 ]; then
    echo "ERROR: Download failed or incomplete!"
    exit 1
fi

echo "[STEP 2/4] Converting train data to golden_period_dataset.csv"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

python3 merge_npy.py

if [ ! -f golden_period_dataset.csv ]; then
    echo "ERROR: Failed to create golden_period_dataset.csv"
    exit 1
fi

GOLDEN_SIZE=$(du -h golden_period_dataset.csv | cut -f1)
echo ""
echo "Created golden_period_dataset.csv ($GOLDEN_SIZE)"
echo ""

echo "[STEP 3/4] Converting test data to test_dataset_merged.csv"
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

# Run convert_test_to_csv.py in mode 2 (single merged CSV)
python3 convert_test_to_csv.py 2

if [ ! -f test_dataset_merged.csv ]; then
    echo "ERROR: Failed to create test_dataset_merged.csv"
    exit 1
fi

TEST_SIZE=$(du -h test_dataset_merged.csv | cut -f1)
echo ""
echo "Created test_dataset_merged.csv ($TEST_SIZE)"
echo ""

echo ""
echo "================================================================================"
echo "DATA DOWNLOAD COMPLETE!"
echo "================================================================================"
echo ""
echo "Datasets ready for Tucker-CAM pipeline:"
echo "  golden_period_dataset.csv             ($GOLDEN_SIZE)"
echo "  test_dataset_merged.csv               ($TEST_SIZE)"
echo "  labeled_anomalies.csv                 (3.9K)"
echo ""
echo "Note: NaN handling will be done automatically during preprocessing"
echo ""
echo "Next steps:"
echo "  cd .."
echo "  bash run_tucker_cam_benchmark.sh"
echo ""
echo "================================================================================"
echo ""

exit 0
