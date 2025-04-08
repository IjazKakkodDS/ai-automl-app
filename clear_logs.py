# clear_logs.py

import os
import logging

def clear_app_log(log_file_path: str = "app.log"):
    """
    Clear the contents of the log file specified by log_file_path.
    """
    # Shut down logging to flush and close any open file handlers.
    logging.shutdown()
    
    try:
        # Open the log file in write mode, which truncates the file.
        with open(log_file_path, "w") as f:
            pass  # Opening in "w" mode clears the file.
        print(f"Log file '{log_file_path}' has been cleared.")
    except Exception as e:
        print(f"Error clearing log file '{log_file_path}': {e}")

if __name__ == "__main__":
    clear_app_log("app.log")
