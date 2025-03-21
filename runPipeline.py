import sys
from src.logger import logging
from src.exception import MyException
from src.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except MyException as e:
    logging.exception(e, sys)
    raise e

# STAGE_NAME = "Data Cleaning Stage"

# try:
#     logger.info(f"*******************")
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     cleaning_stage = CleaningStage()
#     cleaned_data = cleaning_stage.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Feature Engineering Stage"

# try:
#     logger.info(f"*******************")
#     logger.info(f">>>>>> stage {STAGE_NAME} 1: FeatureEngineeringConfig started <<<<<<")

#     # Instantiate FeatureEngineering
#     fe_strategy = FeatureEngineeringStrategy()
#     fe = FeatureEngineering(data=cleaned_data, strategy=fe_strategy)

#     # Feature Engineering Stage 1: FeatureEngineeringConfig
#     fe_config = FeatureEngineeringConfig()
#     cleaned_data = fe_config.handle_FE(cleaned_data)

#     logger.info(f">>>>>> stage {STAGE_NAME} 1: FeatureEngineeringConfig completed <<<<<<")
#     logger.info(f">>>>>> stage {STAGE_NAME} 2: OutlierTreatment started <<<<<<")

#     # Feature Engineering Stage 2: OutlierTreatment
#     outlier_treatment = OutlierTreatment()
#     cleaned_data = outlier_treatment.handle_FE(cleaned_data)

#     logger.info(f">>>>>> stage {STAGE_NAME} 2: OutlierTreatment completed <<<<<<")
#     logger.info(f">>>>>> stage {STAGE_NAME} 3: MissingValueImputation started <<<<<<")

#     # Feature Engineering Stage 3: MissingValueImputation
#     missing_value_imputation = MissingValueImputation()
#     cleaned_data = missing_value_imputation.handle_FE(cleaned_data)

#     logger.info(f">>>>>> stage {STAGE_NAME} 3: MissingValueImputation completed <<<<<<")
#     logger.info(f">>>>>> stage {STAGE_NAME} 4: FeatureSelection started <<<<<<")

#     # Feature Engineering Stage 4: FeatureSelection
#     feature_selection = FeatureSelection()
#     cleaned_data = feature_selection.handle_FE(cleaned_data)

#     logger.info(f">>>>>> stage {STAGE_NAME} 4: FeatureSelection completed <<<<<<")
#     logger.info(f">>>>>> stage {STAGE_NAME} 5: DataDivideStrategy started <<<<<<")

#     # Feature Engineering Stage 5: DataDivideStrategy
#     data_divide_strategy = DataDivideStrategy()
#     X_train, y_train, X_test, y_test = data_divide_strategy.handle_FE(cleaned_data)

#     logger.info(f">>>>>> stage {STAGE_NAME} 5: DataDivideStrategy completed <<<<<<")
#     logger.info(f">>>>>> All Feature Engineering Stages completed successfully! <<<<<<")
#     logger.info(f"*******************\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e

TAGE_NAME = "Model Training Stage"

try:
    logging.info(f"*******************")
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    logging.info(f"*******************x\nx==========x\n")
except MyException as e:
    logging.exception(e, sys)
    raise e
