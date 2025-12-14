#!/bin/bash
# Run chunked nearest-neighbor detection
cd ~/program_internship_paul_wurth

python executable/chunked_nn_detector.py \
  --golden results/golden_baseline/weights/weights_enhanced.csv \
  --test results/test_timeline/weights/weights_enhanced.csv \
  --output results/nn_detection_chunked.csv \
  --chunk-size 50 \
  --sample-rate 10 \
  --lag 0 \
  2>&1 | tee logs/nn_detection_chunked.log
