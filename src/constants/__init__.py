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
TRAINING_STAGE_NAME = "Training Pipeline"

"""
---------------------------------------------------------------
Environment Test and Dependency related constant 
---------------------------------------------------------------
"""
REQUIRED_PYTHON = "python3"
REQUIREMENTS_FILE = "requirements.txt"

"""
---------------------------------------------------------------
Data ingestion related constant 
---------------------------------------------------------------
"""
INGESTION_STAGE_NAME = "Data Ingestion"

"""
---------------------------------------------------------------
Data PrePrecessing related constant 
---------------------------------------------------------------
"""
PRE_PROCESSING_STAGE_NAME = "Data Pre-Processing"
N_CLUSTERS = 3
RANDOM_STATE = 42
WCSS_REDUCED = []