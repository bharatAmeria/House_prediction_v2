import os
from src.constants import *
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP
    
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str
    local_data_file: Path
    unzip_dir: Path 

@dataclass
class DataCleaningConfig:
    cleaned_data_dir: Path
    cleaned_gurgaon_data: Path
    missing_value_imputed: Path
    # cleaned_gurgaon_appartments_data: Path

    gurgaon_data_path: Path
    gurgaon_flats_data: Path
    gurgaon_appartments_data: Path
    gurgaon_houses_data: Path

@dataclass
class DataVisualizationConfig:
    viz_dir: Path
    feature_text: Path
    data_viz: Path
    latlong: Path
    missing_value_imputed: Path
    gurgaon_properties: Path

# @dataclass
# class RecommendSysConfig:
#     recommend_dir: Path
#     appartments_path: Path
#     cosine1: Path
#     cosine2: Path
#     cosine3: Path
