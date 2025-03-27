import sys
from src.constants import *
from src.logger import logging
from src.exception import MyException
# from src.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
# from src.pipeline.stage02_data_cleaning import DataProcessingPipeline
# from src.pipeline.stage03_feature_selection import FeatureSelectionPipeline
# from src.pipeline.stage04_data_visualiztion import DataVizPipeline
from src.pipeline.stage05_recommender_system import RecoomendationSystemPipeline

# try:
#     logging.info(f">>>>>> stage {INGESTION_STAGE_NAME} started <<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.main()
#     logging.info(f">>>>>> stage {INGESTION_STAGE_NAME} completed <<<<<<\n\nx==========x")
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

# try:
#     logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} started <<<<<<")
#     data_processing= DataProcessingPipeline()
#     data_processing.main()
#     logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} completed <<<<<<\n\nx==========x")
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

# try:
#     logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} started <<<<<<")
#     selection = FeatureSelectionPipeline()
#     selection.main()
#     logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} completed <<<<<<\n\nx==========x")
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

# try:
#     logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} started <<<<<<")
#     selection = DataVizPipeline()
#     selection.main()
#     logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} completed <<<<<<\n\nx==========x")
# except MyException as e:
#     logging.exception(e, sys)
#     raise e

try:
    logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} started <<<<<<")
    selection = RecoomendationSystemPipeline()
    selection.main()
    logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} completed <<<<<<\n\nx==========x")
except MyException as e:
    logging.exception(e, sys)
    raise e

try:
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {TRAINING_STAGE_NAME} completed <<<<<<")
    logging.info(f"*******************x\nx==========x\n")
except MyException as e:
    logging.exception(e, sys)
    raise e
