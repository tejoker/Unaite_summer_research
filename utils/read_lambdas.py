#!/usr/bin/env python3
"""
Simple script to read lambda values from JSON without requiring jq
"""
import json
import sys

if len(sys.argv) != 2:
    print("Usage: python3 read_lambdas.py <json_file>")
    sys.exit(1)

json_file = sys.argv[1]

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"{data.get('lambda_w', 0.1)} {data.get('lambda_a', 0.1)}")
except Exception as e:
    print("0.1 0.1")  # Default values if file doesn't exist
    sys.exit(0)
