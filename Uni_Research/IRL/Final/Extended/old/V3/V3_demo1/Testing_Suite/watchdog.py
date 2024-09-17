import os
import time
import subprocess
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Watchdog Configuration
PROGRESS_CHECK_INTERVAL = 20  # seconds
PROGRESS_TIMEOUT = 30  # seconds

def monitor_progress(progress_file, script_command):
    logging.info(f"Starting to monitor progress file: {progress_file}")
    logging.info(f"Script command: {script_command}")

    last_update_time = os.path.getmtime(progress_file) if os.path.exists(progress_file) else None
    notdone = True
    while notdone == True:
        try:
            time.sleep(PROGRESS_CHECK_INTERVAL)
            if not os.path.exists(progress_file):
                logging.warning(f"Progress file {progress_file} does not exist.")
                continue

            current_update_time = os.path.getmtime(progress_file)

            if last_update_time is None or last_update_time != current_update_time:
                logging.debug(f"Progress file updated. Last update: {last_update_time}, Current update: {current_update_time}")
                last_update_time = current_update_time
                continue

            if last_update_time and time.time() - last_update_time > PROGRESS_TIMEOUT:
                logging.warning("No progress detected, restarting the script...")
                
                try:
                    # Kill the existing process (if any)
                    subprocess.run("pkill -f 'python3 testing_script_gen_plots.py'", shell=True)
                    time.sleep(2)  # Wait a bit for the process to be killed
                    
                    # Start the new process
                    subprocess.Popen(script_command, shell=True)
                    logging.info(f"Restarted script with command: {script_command}")
                    notdone = False
                    
                    # Reset the last update time
                    last_update_time = None
                except Exception as e:
                    logging.error(f"Error restarting script: {e}")

        except Exception as e:
            logging.error(f"Unexpected error in watchdog loop: {e}")
            time.sleep(10)  # Wait before retrying to avoid rapid error loops

if __name__ == "__main__":
    if len(sys.argv) > 2:
        progress_file = sys.argv[1]
        script_command = " ".join(sys.argv[2:])
        script_command = "python3 testing_script_gen_plots.py --resume"
        monitor_progress(progress_file, script_command)
    else:
        logging.error("Insufficient arguments provided.")
        print("Usage: python watchdog.py <progress_file> <script_command>")
        sys.exit(1)