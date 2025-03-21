from datetime import date
from pathlib import Path

"""
---------------------------------------------------------------
Training Pipeline related constant start with DATA_INGESTION VAR NAME
---------------------------------------------------------------
"""
# For training pipeline
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"




CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

class ModelNameConfig:
    """Model Configurations"""
    model_name: str = "lightgbm"
    fine_tuning: bool = False
