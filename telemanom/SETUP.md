# Telemanom Dataset Setup

Quick setup guide for downloading and preparing NASA Telemanom spacecraft data for Tucker-CAM pipeline.

## One-Command Setup

After cloning the repository:

```bash
cd telemanom
bash download_data.sh
```

This will:
1. Download NASA Telemanom dataset from Kaggle (~63MB)
2. Convert .npy files to CSV format
3. Apply NaN handling (ffill + dropna)
4. Create clean datasets ready for Tucker-CAM

## Requirements

### Kaggle API Setup

You need a Kaggle account and API key:

1. Create account at https://www.kaggle.com
2. Go to Account Settings: https://www.kaggle.com/settings/account
3. Click "Create New API Token" (downloads kaggle.json)
4. Move to your home directory:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Python Dependencies

```bash
pip install kaggle pandas numpy
```

## What You Get

After running `download_data.sh`, you'll have:

```
telemanom/
├── golden_period_dataset.csv          (33M, original with NaN)
├── golden_period_dataset_clean.csv    (51M, clean, ready to use)
├── test_dataset_merged.csv            (73M, original with NaN)
├── test_dataset_merged_clean.csv      (103M, clean, ready to use)
├── labeled_anomalies.csv              (3.9K, 105 anomaly labels)
└── data/
    ├── train/  (82 .npy files)
    └── test/   (82 .npy files)
```

## Dataset Details

### Golden Baseline (Normal Operation)
- **File**: `golden_period_dataset_clean.csv`
- **Shape**: 4,308 timesteps × 2,889 variables
- **Purpose**: Training data representing normal spacecraft behavior
- **NaN**: None (cleaned with ffill + dropna)

### Test Timeline (With Anomalies)
- **File**: `test_dataset_merged_clean.csv`
- **Shape**: 8,640 timesteps × 2,889 variables
- **Purpose**: Test data containing 105 labeled anomalies
- **NaN**: None (cleaned with ffill + dropna)

### Anomaly Labels
- **File**: `labeled_anomalies.csv`
- **Content**: 105 anomalies from SMAP (69) and MSL (36) spacecraft
- **Columns**: channel_id, spacecraft, anomaly_sequences, class, num_values

## Running the Pipeline

After setup is complete:

```bash
cd ..  # Return to project root
bash run_tucker_cam_benchmark.sh
```

This processes both datasets with Tucker-CAM (~3 hours runtime).

## Troubleshooting

### "Kaggle CLI not found"
Install Kaggle CLI:
```bash
pip install kaggle
```

### "Kaggle API key not found"
Follow the Kaggle API Setup section above.

### "401 Unauthorized"
Your kaggle.json permissions may be wrong:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### "Download failed"
Check your internet connection and verify your Kaggle account is active.

## Manual Setup (If Script Fails)

If `download_data.sh` fails, you can run steps manually:

```bash
# Step 1: Download from Kaggle
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
mv nasa-anomaly-detection-dataset-smap-msl.zip data.zip
unzip -o data.zip
rm data.zip
mv data/data tmp && rm -r data && mv tmp data

# Step 2: Convert train data to CSV
python3 merge_npy.py

# Step 3: Convert test data to CSV
python3 convert_test_to_csv.py 2

# Step 4: Prepare clean datasets
python3 prepare_datasets.py
```

## Data Source

Dataset: [NASA Anomaly Detection Dataset (SMAP/MSL)](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)

Paper: Hundman et al. (2018), "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"

## Next Steps

See [../README.md](../README.md) for Tucker-CAM pipeline usage and benchmarking instructions.
