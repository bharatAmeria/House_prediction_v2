import sys
import subprocess
import logging
from pathlib import Path
from src.constants import REQUIREMENTS_FILE, REQUIRED_PYTHON

def check_python_version():
    """Ensure the correct Python version is being used."""
    system_major = sys.version_info.major
    required_major = 3 if REQUIRED_PYTHON == "python3" else 2

    if system_major != required_major:
        msg = f"This project requires Python {required_major}. Found: Python {sys.version}"
        logging.error(msg)
        raise TypeError(msg)
    else:
        logging.info("Python version check passed.")
        print(">>> Development environment passes all tests!")


def install_requirements():
    """Install dependencies from requirements.txt."""
    req_file = Path(REQUIREMENTS_FILE)

    if not req_file.exists():
        logging.warning(f"Requirements file '{REQUIREMENTS_FILE}' not found. Skipping installation.")
        print(f"Warning: '{REQUIREMENTS_FILE}' not found. Skipping dependencies installation.")
        return

    try:
        logging.info(f"Installing dependencies from {REQUIREMENTS_FILE}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
        logging.info("Dependencies installed successfully.")
        print(">>> Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to install dependencies: {e}")
        print(f"Error: Failed to install dependencies. Check Logs for details.")


def main():
    try:
        check_python_version()
        install_requirements()
    except Exception as e:
        logging.error(f"Setup failed: {e}")
        print(f"Error: {e}. Check Logs for details.")


if __name__ == '__main__':
    main()
