#!/bin/bash
#
# Super Monitor - Comprehensive System Health Monitoring Script (No sudo required)
#
# Monitors every 10 seconds using only built-in Linux tools:
# - Memory usage (total, used, available, swap)
# - CPU usage per core
# - Process-level CPU and memory usage
# - Disk usage
# - System load averages
#
# Usage: ./super_monitor.sh [output_file]
#   Default output: system_monitor_<timestamp>.log
#

INTERVAL=10  # Monitoring interval in seconds
OUTPUT_FILE="${1:-logs/system_monitor_$(date '+%Y%m%d_%H%M%S').log}"

# Print header
print_header() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "================================================================================"
    echo "[$timestamp] SYSTEM HEALTH MONITOR"
    echo "================================================================================"
    echo "Hostname: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "Uptime: $(uptime)"
    echo "Monitoring interval: ${INTERVAL}s"
    echo "Output file: $OUTPUT_FILE"
    echo "================================================================================"
    echo ""
}

# Get power consumption (if available)
get_power_consumption() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] POWER CONSUMPTION"
    echo "────────────────────────────────────────────────────────────────────────────────"

    # Method 1: RAPL (Intel Running Average Power Limit)
    if [ -d /sys/class/powercap/intel-rapl ]; then
        echo "  Intel RAPL Energy Consumption:"
        for rapl in /sys/class/powercap/intel-rapl/intel-rapl:*; do
            if [ -f "$rapl/name" ] && [ -f "$rapl/energy_uj" ]; then
                name=$(cat "$rapl/name")
                energy_uj=$(cat "$rapl/energy_uj")
                energy_w=$(awk "BEGIN {printf \"%.2f\", $energy_uj / 1000000}")
                echo "    $name: ${energy_w}W (cumulative: ${energy_uj}uJ)"
            fi
        done
    else
        echo "  RAPL power metrics not available"
    fi

    # Method 2: Battery status (for laptops)
    if [ -d /sys/class/power_supply ]; then
        for battery in /sys/class/power_supply/BAT*; do
            if [ -d "$battery" ]; then
                echo "  Battery: $(basename $battery)"
                [ -f "$battery/status" ] && echo "    Status: $(cat $battery/status)"
                [ -f "$battery/capacity" ] && echo "    Capacity: $(cat $battery/capacity)%"
                [ -f "$battery/power_now" ] && echo "    Power: $(awk '{printf "%.2f W", $1/1000000}' $battery/power_now)"
                [ -f "$battery/current_now" ] && echo "    Current: $(awk '{printf "%.2f mA", $1/1000}' $battery/current_now)"
            fi
        done
    fi

    echo ""
}

# Get memory statistics
get_memory_stats() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] MEMORY USAGE"
    echo "────────────────────────────────────────────────────────────────────────────────"

    # Overall memory using /proc/meminfo
    echo "  Overall Memory:"
    awk '/MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Dirty|Slab/ {
        printf "    %-20s %10s %s\n", $1, $2, $3
    }' /proc/meminfo

    echo ""
    echo "  Memory Pressure:"

    # Calculate memory pressure
    local mem_total=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    local mem_available=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local mem_used=$((mem_total - mem_available))
    local mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used / $mem_total) * 100}")

    # Convert to human readable
    local mem_total_mb=$((mem_total / 1024))
    local mem_used_mb=$((mem_used / 1024))
    local mem_available_mb=$((mem_available / 1024))

    echo "    Total:     ${mem_total_mb} MB"
    echo "    Used:      ${mem_used_mb} MB (${mem_percent}%)"
    echo "    Available: ${mem_available_mb} MB"

    # Status indicator
    if (( $(echo "$mem_percent > 90" | bc -l 2>/dev/null || echo 0) )); then
        echo "    Status: CRITICAL - Memory heavily utilized"
    elif (( $(echo "$mem_percent > 75" | bc -l 2>/dev/null || echo 0) )); then
        echo "    Status: WARNING - Memory moderately utilized"
    else
        echo "    Status: OK - Memory usage normal"
    fi

    echo ""
}

# Get CPU statistics using /proc/stat
get_cpu_stats() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] CPU USAGE"
    echo "────────────────────────────────────────────────────────────────────────────────"

    # Read CPU stats
    echo "  CPU Statistics (averaged over ${INTERVAL}s interval):"

    # First reading
    local cpu_stats1=$(grep '^cpu' /proc/stat)
    sleep 1
    # Second reading
    local cpu_stats2=$(grep '^cpu' /proc/stat)

    # Calculate CPU usage
    echo "$cpu_stats1" > /tmp/cpu_stat1_$$
    echo "$cpu_stats2" > /tmp/cpu_stat2_$$

    awk '
    NR==FNR {
        cpu=$1
        user1[cpu]=$2; nice1[cpu]=$3; system1[cpu]=$4; idle1[cpu]=$5
        iowait1[cpu]=$6; irq1[cpu]=$7; softirq1[cpu]=$8
        total1[cpu]=$2+$3+$4+$5+$6+$7+$8
        next
    }
    {
        cpu=$1
        user2=$2; nice2=$3; system2=$4; idle2=$5
        iowait2=$6; irq2=$7; softirq2=$8
        total2=$2+$3+$4+$5+$6+$7+$8

        total_diff = total2 - total1[cpu]
        idle_diff = idle2 - idle1[cpu]

        if (total_diff > 0) {
            usage = 100 * (total_diff - idle_diff) / total_diff
            user_pct = 100 * (user2 - user1[cpu]) / total_diff
            sys_pct = 100 * (system2 - system1[cpu]) / total_diff
            iowait_pct = 100 * (iowait2 - iowait1[cpu]) / total_diff
            idle_pct = 100 * idle_diff / total_diff

            if (cpu == "cpu") {
                printf "    Overall:  Usage: %5.1f%%  User: %5.1f%%  System: %5.1f%%  IOWait: %5.1f%%  Idle: %5.1f%%\n",
                    usage, user_pct, sys_pct, iowait_pct, idle_pct
            } else {
                core_num = substr(cpu, 4)
                printf "    %-8s  Usage: %5.1f%%  User: %5.1f%%  System: %5.1f%%  IOWait: %5.1f%%  Idle: %5.1f%%\n",
                    cpu, usage, user_pct, sys_pct, iowait_pct, idle_pct
            }
        }
    }
    ' /tmp/cpu_stat1_$$ /tmp/cpu_stat2_$$

    rm -f /tmp/cpu_stat1_$$ /tmp/cpu_stat2_$$

    echo ""
    echo "  CPU Frequencies:"

    # CPU frequency per core
    if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
            if [ -f "$cpu" ]; then
                cpu_num=$(echo $cpu | grep -oP 'cpu\K[0-9]+')
                freq_khz=$(cat $cpu)
                freq_mhz=$(awk "BEGIN {printf \"%.0f\", $freq_khz / 1000}")
                printf "    CPU%-2s  %4d MHz" "$cpu_num" "$freq_mhz"

                # Add max frequency
                max_freq_file="/sys/devices/system/cpu/cpu${cpu_num}/cpufreq/scaling_max_freq"
                if [ -f "$max_freq_file" ]; then
                    max_khz=$(cat $max_freq_file)
                    max_mhz=$(awk "BEGIN {printf \"%.0f\", $max_khz / 1000}")
                    printf " / %4d MHz max" "$max_mhz"
                fi
                echo ""
            fi
        done
    else
        echo "    CPU frequency information not available"
    fi

    echo ""
    echo "  CPU Temperature:"

    # CPU temperature
    if [ -d /sys/class/thermal ]; then
        for thermal in /sys/class/thermal/thermal_zone*/temp; do
            if [ -f "$thermal" ]; then
                zone=$(basename $(dirname $thermal))
                temp_milli=$(cat $thermal)
                temp_c=$(awk "BEGIN {printf \"%.1f\", $temp_milli / 1000}")

                # Get thermal zone type
                type_file=$(dirname $thermal)/type
                if [ -f "$type_file" ]; then
                    zone_type=$(cat $type_file)
                    echo "    $zone_type: ${temp_c}°C"
                else
                    echo "    $zone: ${temp_c}°C"
                fi
            fi
        done
    else
        echo "    Temperature sensors not available"
    fi

    echo ""
    echo "  Load Averages:"
    echo "    $(cat /proc/loadavg | awk '{printf "1min: %.2f  5min: %.2f  15min: %.2f  Running/Total: %s", $1, $2, $3, $4}')"

    echo ""
}

# Get top processes by CPU and memory
get_top_processes() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] TOP PROCESSES"
    echo "────────────────────────────────────────────────────────────────────────────────"

    echo "  Top 10 Processes by CPU Usage:"
    ps aux --sort=-%cpu | head -n 11 | awk 'NR==1 {
        printf "    %-8s %6s %5s %5s %10s %-10s %s\n", "USER", "PID", "%CPU", "%MEM", "VSZ", "STAT", "COMMAND"
    } NR>1 {
        cmd = ""
        for (i=11; i<=NF; i++) cmd = cmd $i " "
        printf "    %-8s %6s %5.1f %5.1f %10s %-10s %s\n", $1, $2, $3, $4, $5, $8, substr(cmd, 1, 60)
    }'

    echo ""
    echo "  Top 10 Processes by Memory Usage:"
    ps aux --sort=-%mem | head -n 11 | awk 'NR==1 {
        printf "    %-8s %6s %5s %5s %10s %-10s %s\n", "USER", "PID", "%CPU", "%MEM", "RSS", "STAT", "COMMAND"
    } NR>1 {
        cmd = ""
        for (i=11; i<=NF; i++) cmd = cmd $i " "
        printf "    %-8s %6s %5.1f %5.1f %10s %-10s %s\n", $1, $2, $3, $4, $6, $8, substr(cmd, 1, 60)
    }'

    echo ""
    echo "  Process Count by State:"
    ps aux | awk 'NR>1 {
        state=substr($8,1,1)
        count[state]++
        total++
    } END {
        printf "    Running (R):     %4d\n", count["R"]+0
        printf "    Sleeping (S):    %4d\n", count["S"]+0
        printf "    Disk Sleep (D):  %4d\n", count["D"]+0
        printf "    Zombie (Z):      %4d\n", count["Z"]+0
        printf "    Stopped (T):     %4d\n", count["T"]+0
        printf "    Total:           %4d\n", total
    }'

    echo ""
}

# Get disk usage
get_disk_usage() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] DISK USAGE & STORAGE"
    echo "────────────────────────────────────────────────────────────────────────────────"

    echo "  Disk Usage:"
    df -h | awk 'NR==1 {
        printf "    %-25s %8s %8s %8s %5s  %s\n", "Filesystem", "Size", "Used", "Avail", "Use%", "Mounted"
    } NR>1 {
        printf "    %-25s %8s %8s %8s %5s  %s\n", $1, $2, $3, $4, $5, $6
    }'

    echo ""
    echo "  Inode Usage:"
    df -i | awk 'NR==1 {
        printf "    %-25s %10s %10s %10s %5s\n", "Filesystem", "Inodes", "IUsed", "IFree", "IUse%"
    } NR>1 && $1 ~ /^\/dev\// {
        printf "    %-25s %10s %10s %10s %5s\n", $1, $2, $3, $4, $5
    }'

    echo ""
}

# Get network I/O statistics
get_network_io() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] NETWORK I/O"
    echo "────────────────────────────────────────────────────────────────────────────────"

    echo "  Network Interfaces:"

    # Get network stats from /proc/net/dev
    awk 'NR>2 {
        iface=$1
        sub(/:/, "", iface)
        rx_bytes=$2
        tx_bytes=$10
        rx_packets=$3
        tx_packets=$11

        # Convert bytes to human readable
        rx_mb = rx_bytes / 1024 / 1024
        tx_mb = tx_bytes / 1024 / 1024

        printf "    %-10s  RX: %10.2f MB (%s packets)  TX: %10.2f MB (%s packets)\n",
            iface, rx_mb, rx_packets, tx_mb, tx_packets
    }' /proc/net/dev

    echo ""
    echo "  Active Network Connections:"

    # Count connection states from /proc/net/tcp and /proc/net/tcp6
    (cat /proc/net/tcp /proc/net/tcp6 2>/dev/null || cat /proc/net/tcp) | awk '
        NR>1 {
            # State is in field 4 (hex value)
            state_hex = $4

            # TCP states (hex values)
            if (state_hex == "01") state="ESTABLISHED"
            else if (state_hex == "02") state="SYN_SENT"
            else if (state_hex == "03") state="SYN_RECV"
            else if (state_hex == "04") state="FIN_WAIT1"
            else if (state_hex == "05") state="FIN_WAIT2"
            else if (state_hex == "06") state="TIME_WAIT"
            else if (state_hex == "07") state="CLOSE"
            else if (state_hex == "08") state="CLOSE_WAIT"
            else if (state_hex == "09") state="LAST_ACK"
            else if (state_hex == "0A") state="LISTEN"
            else if (state_hex == "0B") state="CLOSING"
            else state="UNKNOWN"

            count[state]++
            total++
        }
        END {
            printf "    ESTABLISHED:  %4d\n", count["ESTABLISHED"]+0
            printf "    TIME_WAIT:    %4d\n", count["TIME_WAIT"]+0
            printf "    CLOSE_WAIT:   %4d\n", count["CLOSE_WAIT"]+0
            printf "    LISTEN:       %4d\n", count["LISTEN"]+0
            printf "    SYN_SENT:     %4d\n", count["SYN_SENT"]+0
            printf "    SYN_RECV:     %4d\n", count["SYN_RECV"]+0
            printf "    Total:        %4d\n", total
        }
    '

    echo ""
}

# Get Python/Tucker-CAM specific processes
get_pipeline_processes() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] PIPELINE PROCESSES"
    echo "────────────────────────────────────────────────────────────────────────────────"

    # Find Python processes
    echo "  Python/Tucker-CAM Processes:"
    local count=$(ps aux | grep -E "python|tucker|dynotears|launcher" | grep -v grep | wc -l)

    if [ $count -eq 0 ]; then
        echo "    No pipeline processes currently running"
    else
        ps aux | grep -E "python|tucker|dynotears|launcher" | grep -v grep | awk '{
            cmd = ""
            for (i=11; i<=NF; i++) cmd = cmd $i " "
            printf "    PID: %-6s  CPU: %5s%%  MEM: %5s%%  RSS: %8sKB  TIME: %-10s  CMD: %s\n",
            $2, $3, $4, $6, $10, substr(cmd, 1, 80)
        }' | head -n 20
    fi

    echo ""
}

# Main monitoring loop
main() {
    # Print initial header to both stdout and file
    print_header | tee "$OUTPUT_FILE"

    echo "Press Ctrl+C to stop monitoring"
    echo ""

    # Counter for iterations
    iteration=0

    # Trap Ctrl+C for clean exit
    trap 'echo ""; echo "Monitoring stopped. Output saved to: $OUTPUT_FILE"; exit 0' INT

    while true; do
        iteration=$((iteration + 1))

        {
            echo ""
            echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
            echo "║ MONITORING CYCLE #${iteration}"
            echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
            echo ""

            get_memory_stats
            get_cpu_stats
            get_top_processes
            get_pipeline_processes
            get_disk_usage
            get_network_io
            get_power_consumption

            echo "================================================================================"
            echo "Next update in ${INTERVAL} seconds..."
            echo "================================================================================"

        } | tee -a "$OUTPUT_FILE"

        sleep $INTERVAL
    done
}

# Run main function
main
