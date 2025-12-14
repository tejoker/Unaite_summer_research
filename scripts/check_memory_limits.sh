#!/bin/bash
echo "=== System Memory Info ==="
free -h
echo ""
echo "=== Process Limits ==="
ulimit -a
echo ""
echo "=== OOM Killer Logs ==="
dmesg | grep -i "killed process" | tail -10
echo ""
echo "=== Recent Python Process Kills ==="
dmesg | grep -i "python" | grep -i "kill" | tail -5
