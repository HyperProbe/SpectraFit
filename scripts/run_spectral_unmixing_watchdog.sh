#!/bin/bash

# Simple bash wrapper for the Python watchdog script
# This script starts the watchdog and handles any additional setup

echo "Starting Python Watchdog"
echo "=============================================="

# Navigate to the scripts directory
cd "$(dirname "$0")"

# Default values
WATCHED_SCRIPT="spectral_unmixing.py"
TIMEOUT=120      # Default 2 minutes timeout
MAX_RESTARTS=50  # Default 50 restarts

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--script)
      WATCHED_SCRIPT="$2"; shift 2;;
    -t|--timeout)
      TIMEOUT="$2"; shift 2;;
    -r|--max-restarts)
      MAX_RESTARTS="$2"; shift 2;;
    -*|--*)
      echo "Unknown option: $1"; exit 1;;
    *)
      break;;
  esac
done

echo "Configuration:"
echo "- Monitored script: $(pwd)/$WATCHED_SCRIPT"
echo "- Timeout: ${TIMEOUT} seconds"
echo "- Max restarts: ${MAX_RESTARTS}"
echo ""

# Check if monitored script exists
if [ ! -f "$WATCHED_SCRIPT" ]; then
    echo "ERROR: Monitored script '$WATCHED_SCRIPT' not found in current directory!"
    exit 1
fi

echo "Starting watchdog..."
echo "Press Ctrl+C to stop"
echo ""

# Start the watchdog
python3 spectral_unmixing_watchdog.py "$WATCHED_SCRIPT" "$TIMEOUT" "$MAX_RESTARTS"

echo ""
echo "Watchdog finished."
