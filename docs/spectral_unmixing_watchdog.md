# Script Monitoring Watchdog

This watchdog script monitors Python scripts (especially `spectral_unmixing.py`) and automatically restarts them if they hang or stop producing output.

## Problem

Python scripts, particularly `spectral_unmixing.py`, sometimes hang during multiprocessing operations when processing large HSI datasets. This watchdog monitors the script's output and restarts it when necessary.

## Files

- `spectral_unmixing_watchdog.py` - Main Python watchdog script (can monitor any Python script)
- `run_spectral_unmixing_watchdog.sh` - Convenient bash wrapper with argument parsing
- `spectral_unmixing.py` - Default monitored script (other scripts can be specified)

## Usage

### Option 1: Using the bash wrapper (recommended)

```bash
cd /home/macht/thesis/hsi-biopsy/scripts
./run_spectral_unmixing_watchdog.sh [OPTIONS]
```

**Available Options:**
- `-s, --script SCRIPT_NAME` - Specify which script to monitor (default: `spectral_unmixing.py`)
- `-t, --timeout SECONDS` - Timeout in seconds before restart (default: 120)
- `-r, --max-restarts NUM` - Maximum number of restarts (default: 50)

Examples:
```bash
# Monitor the default spectral_unmixing.py script with default settings
./run_spectral_unmixing_watchdog.sh

# Monitor with custom timeout and restart limits
./run_spectral_unmixing_watchdog.sh --script spectral_unmixing.py --timeout 300 --max-restarts 30

# Monitor another script with 3-minute timeout
./run_spectral_unmixing_watchdog.sh -s wavelength_selector.py -t 180
```

### Option 2: Using Python directly

```bash
cd /home/macht/thesis/hsi-biopsy/scripts
python3 spectral_unmixing_watchdog.py [script_to_monitor] [timeout_seconds] [max_restarts]
```

Examples:
```bash
# Monitor default spectral_unmixing.py with default settings
python3 spectral_unmixing_watchdog.py


# Monitor with custom timeout and restart limits
python3 spectral_unmixing_watchdog.py spectral_unmixing.py 300 25

# Monitor another script with custom settings
python3 spectral_unmixing_watchdog.py wavelength_selector.py 180 10
```

## Common Use Cases

### 1. Monitoring Spectral Unmixing (Default)
```bash
# Standard spectral unmixing with default settings
./run_spectral_unmixing_watchdog.sh

# With longer timeout for large datasets
./run_spectral_unmixing_watchdog.sh --timeout 300
```

### 2. Monitoring Wavelength Selection
```bash
# Wavelength selection scripts
./run_spectral_unmixing_watchdog.sh --script run_wavelength_selector_biopsy1.py --timeout 240
./run_spectral_unmixing_watchdog.sh --script run_wavelength_selector_biopsy2.py --timeout 240
```


## How it works

1. **Process Monitoring**: The watchdog starts the specified Python script (default: `spectral_unmixing.py`) as a subprocess
2. **Output Tracking**: It monitors both stdout and stderr for any output
3. **Timeout Detection**: If no output is received for the specified timeout period (default: 2 minutes), the process is considered hung
4. **Automatic Restart**: The hung process is terminated and restarted automatically
5. **Progress Preservation**: Most scripts (including `spectral_unmixing.py`) skip already-processed samples, so restarts don't lose progress
6. **Completion Detection**: The watchdog exits when the script completes successfully

## Configuration

### Script Selection
- **Default**: `spectral_unmixing.py` (if no script specified)
- **Custom**: Any Python script in the same directory can be monitored
- **Examples**: `spectral_unmixing_with_reduced_wl_set.py`, `wavelength_selector.py`, etc.

### Timeout (seconds)
- **Default**: 120 seconds (2 minutes)
- **Recommended**: 120-300 seconds depending on your system and data size

### Max Restarts
- **Default**: 50 restarts

## Script Compatibility

The watchdog can monitor any Python script, but works best with scripts that:
- Print regular progress updates to stdout or stderr
- Can be safely interrupted and restarted
- Skip already-processed data on restart (like `spectral_unmixing.py`)

**Note**: The watchdog monitors for ANY output. Scripts that don't produce output for extended periods may be restarted unnecessarily. For such scripts, consider increasing the timeout value.
