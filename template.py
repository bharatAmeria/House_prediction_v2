import os
from pathlib import Path

project_name = "src"

list_of_files = [

    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/data_ingestion.py",
    f"{project_name}/data/data_processing_lv2.py",
    f"{project_name}/data/flats_data_processing.py",
    f"{project_name}/data/house_data_processing.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/feature_engg/__init__.py",
    f"{project_name}/feature_engg/feature_engg.py",
    f"{project_name}/feature_engg/feature_selection.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/model/__init__.py",
    f"{project_name}/model/model_selection.py",
    f"{project_name}/outlier/__init__.py",
    f"{project_name}/outlier/missing_value_imputation.py",
    f"{project_name}/pipline/__init__.py",
    f"{project_name}/pipeline/stage01_data_ingestion.py",
    f"{project_name}/pipeline/stage02_data_cleaning.py",
    f"{project_name}/pipeline/stage03_feature_selection.py",
    f"{project_name}/pipeline/stage04_data_visualiztion.py",
    f"{project_name}/pipeline/stage05_recommender_system.py",
    f"{project_name}/pipeline/stage06_model_training.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",
    f"{project_name}/__init__.py",
    f"{project_name}/visualization/__init__.py",
    f"{project_name}/visualization/data_visualization.py",
    f"{project_name}/pipline/prediction_pipeline.py",
    f"{project_name}/src/recommender_system.py",
    "deployments/st-argocd.yaml",
    "deployments/st-deployment.yaml",
    "deployments/st-podmonitor.yaml",
    "deployments/st-services.yaml",
    "config/__init__.py",
    "config/config.yaml",
    "requirements.txt",
    "Dockerfile",
    "Jenkinsfile",
    "params.yaml",
    "runPipeline.py",
    "setup.py",
    "pyproject.toml",
    "testEnvironment.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")