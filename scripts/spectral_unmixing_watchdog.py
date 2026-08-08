#!/usr/bin/env python3
"""
Watchdog script for spectral_unmixing.py

This script monitors the spectral_unmixing.py process and restarts it if it hangs
or stops producing output for more than the specified timeout period.
"""

import subprocess
import time
import os
import sys
import signal
import threading
from pathlib import Path
from datetime import datetime


class SpectralUnmixingWatchdog:
    def __init__(self, script_path, timeout_seconds=120, max_restarts=50):
        """
        Initialize the watchdog.

        Args:
            script_path (str): Path to the spectral_unmixing.py script
            timeout_seconds (int): Timeout in seconds before restarting (default: 120 = 2 minutes)
            max_restarts (int): Maximum number of restarts before giving up (default: 50)
        """
        self.script_path = script_path
        self.timeout_seconds = timeout_seconds
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.process = None
        self.last_output_time = None
        self.output_lock = threading.Lock()

    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] WATCHDOG: {message}")

    def output_reader(self, pipe, pipe_name):
        """Read output from subprocess and update last_output_time."""
        try:
            for line in iter(pipe.readline, b""):
                if line:
                    with self.output_lock:
                        self.last_output_time = time.time()
                    # Print the output from the script
                    decoded_line = line.decode("utf-8", errors="replace").rstrip()
                    print(f"[SCRIPT {pipe_name}] {decoded_line}")
        except Exception as e:
            self.log(f"Error reading {pipe_name}: {e}")

    def start_process(self):
        """Start the spectral_unmixing.py process."""
        self.log(f"Starting {self.script_path} (attempt {self.restart_count + 1})")

        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(self.script_path))

        # Start the process
        self.process = subprocess.Popen(
            [sys.executable, os.path.basename(self.script_path)],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line buffered
            universal_newlines=False,
            start_new_session=True,  # run in its own process group
        )

        # Update last output time
        with self.output_lock:
            self.last_output_time = time.time()

        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=self.output_reader, args=(self.process.stdout, "STDOUT")
        )
        stderr_thread = threading.Thread(
            target=self.output_reader, args=(self.process.stderr, "STDERR")
        )

        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()

        return stdout_thread, stderr_thread

    def kill_process(self):
        """Kill the current process if it's running."""
        if self.process and self.process.poll() is None:
            self.log("Terminating hung process...")
            try:
                # Try graceful termination first
                self.process.terminate()
                time.sleep(5)

                # Force kill if still running
                if self.process.poll() is None:
                    self.log("Force killing process...")
                    self.process.kill()

                # Wait for process to actually die
                self.process.wait(timeout=10)

            except subprocess.TimeoutExpired:
                self.log("Failed to kill process - it may be completely hung")
            except Exception as e:
                self.log(f"Error killing process: {e}")

        # Kill any remaining processes in the child's process group
        if self.process:
            try:
                # Only try process group kill if process is still alive
                if self.process.poll() is None:
                    pgid = os.getpgid(self.process.pid)
                    self.log(f"Sending SIGTERM to process group {pgid}...")
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(3)
                    if self.process.poll() is None:
                        self.log(f"Force killing process group {pgid} with SIGKILL...")
                        os.killpg(pgid, signal.SIGKILL)
                        time.sleep(2)
            except (ProcessLookupError, OSError) as e:
                self.log(f"Process group already dead or not accessible: {e}")
            except Exception as e:
                self.log(f"Error killing process group: {e}")

        # Fallback: Use targeted pkill to catch any remaining processes
        # Use full path to avoid matching the watchdog itself
        script_name = os.path.basename(self.script_path)
        watchdog_pid = os.getpid()
        self.log(
            f"Running targeted pkill to remove any {script_name} leftovers (excluding watchdog PID {watchdog_pid})..."
        )
        try:
            # Get all processes matching the script name
            result = subprocess.run(
                ["pgrep", "-f", script_name],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                for pid_str in pids:
                    if pid_str.strip():
                        pid = int(pid_str.strip())
                        # Don't kill the watchdog itself
                        if pid != watchdog_pid:
                            self.log(f"Killing leftover process PID {pid}")
                            try:
                                os.kill(pid, signal.SIGTERM)
                                time.sleep(1)
                                # Check if still alive and force kill
                                try:
                                    os.kill(pid, 0)  # Test if process exists
                                    os.kill(pid, signal.SIGKILL)
                                except OSError:
                                    pass  # Process already dead
                            except OSError as e:
                                self.log(f"Could not kill PID {pid}: {e}")
        except Exception as e:
            self.log(f"Error during targeted pkill fallback: {e}")

    def run(self):
        """Main watchdog loop."""
        self.log("Starting spectral unmixing watchdog")
        self.log(f"Script: {self.script_path}")
        self.log(f"Timeout: {self.timeout_seconds} seconds")
        self.log(f"Max restarts: {self.max_restarts}")

        while self.restart_count <= self.max_restarts:
            try:
                # Start the process
                stdout_thread, stderr_thread = self.start_process()

                # Monitor the process
                while True:
                    # Check if process has finished
                    if self.process.poll() is not None:
                        return_code = self.process.returncode
                        if return_code == 0:
                            self.log("Script completed successfully!")
                            return True
                        else:
                            self.log(f"Script exited with error code: {return_code}")
                            break

                    # Check for timeout
                    with self.output_lock:
                        time_since_output = time.time() - self.last_output_time

                    if time_since_output > self.timeout_seconds:
                        self.log(
                            f"No output for {time_since_output:.1f} seconds - process appears hung"
                        )
                        break

                    # Wait before next check
                    time.sleep(10)

                # If we reach here, the process died or hung
                self.kill_process()
                self.restart_count += 1

                if self.restart_count <= self.max_restarts:
                    self.log(
                        f"Will restart in 5 seconds... (restart {self.restart_count}/{self.max_restarts})"
                    )
                    time.sleep(5)
                else:
                    self.log(
                        f"Maximum restart limit ({self.max_restarts}) reached. Giving up."
                    )
                    return False

            except KeyboardInterrupt:
                self.log("Received interrupt signal - shutting down")
                self.kill_process()
                return False
            except Exception as e:
                self.log(f"Unexpected error: {e}")
                self.kill_process()
                time.sleep(5)

        return False


def main():
    """Main entry point."""
    # Determine directory of this watchdog script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Default monitored script
    default_script = os.path.join(script_dir, "spectral_unmixing.py")

    # Extract command-line arguments
    args = sys.argv[1:]

    # Determine which script to monitor
    if len(args) >= 1 and os.path.exists(args[0]):
        script_path = args[0]
        arg_index = 1
    else:
        script_path = default_script
        arg_index = 0

    # Verify the monitored script exists
    if not os.path.exists(script_path):
        print(f"Error: Script to monitor '{script_path}' not found!")
        sys.exit(1)

    # Default timeout and restart settings
    timeout_seconds = 120
    max_restarts = 50

    # Parse timeout argument if provided
    if len(args) > arg_index:
        try:
            timeout_seconds = int(args[arg_index])
        except ValueError:
            print(
                f"Invalid timeout value '{args[arg_index]}'. Using default {timeout_seconds} seconds."
            )
        arg_index += 1

    # Parse max_restarts argument if provided
    if len(args) > arg_index:
        try:
            max_restarts = int(args[arg_index])
        except ValueError:
            print(
                f"Invalid max_restarts value '{args[arg_index]}'. Using default {max_restarts}."
            )

    # Create and run the watchdog
    watchdog = SpectralUnmixingWatchdog(
        script_path=script_path,
        timeout_seconds=timeout_seconds,
        max_restarts=max_restarts,
    )
    success = watchdog.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
