#!/bin/bash
# Test script to verify launcher.py variants functionality

cd /home/nicolasbigeard/program_internship_paul_wurth/executable

echo "Testing launcher.py --variants functionality"
echo "========================================="

# Test dry-run with variants
echo "1. Testing dry-run with --variants no_mi"
python launcher.py --folder final_pipeline --csv-file ../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv --variants no_mi --dry-run

echo ""
echo "2. Testing dry-run with --variants no_rolling"
python launcher.py --folder final_pipeline --csv-file ../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv --variants no_rolling --dry-run

echo ""
echo "3. Testing dry-run with --variants no_mi no_rolling"
python launcher.py --folder final_pipeline --csv-file ../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv --variants no_mi no_rolling --dry-run

echo ""
echo "4. Testing dry-run with normal mode (no variants)"
python launcher.py --folder final_pipeline --csv-file ../data/Anomaly/output_of_the_1th_chunk__Temperatur_Exzenterlager_links__amplitude_change.csv --dry-run

echo "========================================="
echo "Tests completed!"