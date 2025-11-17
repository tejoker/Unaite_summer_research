#!/usr/bin/env python3
"""
Simple script to read a field from JSON without requiring jq
"""
import json
import sys

if len(sys.argv) < 3:
    print("N/A")
    sys.exit(0)

json_file = sys.argv[1]
field = sys.argv[2]
default = sys.argv[3] if len(sys.argv) > 3 else "N/A"

try:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Support nested fields or fallback fields
    if '//' in field:
        fields = [f.strip() for f in field.split('//')]
        for f in fields:
            if f in data and data[f] is not None:
                print(data[f])
                sys.exit(0)
        print(default)
    else:
        value = data.get(field, default)
        print(value if value is not None else default)
except Exception:
    print(default)
