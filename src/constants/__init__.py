from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

"""
---------------------------------------------------------------
Training Pipeline related constant start with DATA_INGESTION VAR NAME
---------------------------------------------------------------
"""
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

"""
---------------------------------------------------------------
Environment Test and Dependency related constant 
---------------------------------------------------------------
"""
REQUIRED_PYTHON = "python3"
REQUIREMENTS_FILE = "requirements.txt"