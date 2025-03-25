import sys
import pandas as pd
import pickle
from src.utils.common import save_csv
from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.data.data_processing_lv2 import DataPreProcessing, DataPreprocessStrategy
from src.data.house_data_processing import HouseDataCleaning, HouseDataPreProcessingStrategy
from src.data.flats_data_processing import FlatsDataCleaning, FlatsDataPreProcessingStrategy
from src.outlier.outlier_treatment import RemovingOutlier, OutlierProcessStrategy
from src.outlier.missing_value_imputation import RemovingMissingValues, MissingValueStrategy
from src.feature_engg.feature_engg import FeatureEngineering, FeatureEngineeringConfig
from src.feature_engg.feature_selection import FeatureSelection, FeatureSelectionConfig
from src.config.configuration import ConfigurationManager

class DataProcessingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()

        logging.info(">>>>>Housing Data Preprocessing Started<<<<<")
        house_data = pd.read_csv(data_cleaning_config.gurgaon_houses_data)
        house_data_cleaning = HouseDataCleaning(data=house_data, strategy=HouseDataPreProcessingStrategy(), config=data_cleaning_config) # completed with no error
        house_cleaned_data = house_data_cleaning.handle_data()

        logging.info(">>>>>Flats Data Preprocessing Started...<<<<<")
        flats_data = pd.read_csv(data_cleaning_config.gurgaon_flats_data)
        flats_data_cleaning = FlatsDataCleaning(data=flats_data, strategy=FlatsDataPreProcessingStrategy(), config=data_cleaning_config) # completed with no error
        flats_cleaned_data = flats_data_cleaning.handle_data()
        logging.info(">>>>>Flats Data Preprocessing Completed<<<<<\n")

        logging.info("Merging Flats and House cleaned data")
        merged_data = pd.concat([flats_cleaned_data, house_cleaned_data],ignore_index=True)

        print(merged_data.info())

        logging.info(">>>>>Levael-2 Data Preprocessing Started...<<<<<")
        data_cleaning = DataPreProcessing(data=merged_data,strategy=DataPreprocessStrategy())
        cleaned_data = data_cleaning.handle_data()
        logging.info(">>>>>Levael-2 Data Preprocessing Completed<<<<<\n")

        logging.info(">>>>>Feature Engg. Started...<<<<<")
        fe_strategy = FeatureEngineering(data=cleaned_data, strategy=FeatureEngineeringConfig())
        fe = fe_strategy.handle_FE()
        logging.info(">>>>>Feature Engg. Completed<<<<<\n") 

        logging.info(">>>>>Outlier Removing Started...<<<<<")
        outlier_strategy = RemovingOutlier(data=fe, strategy=OutlierProcessStrategy())
        outlier = outlier_strategy.handle_outlier()
        logging.info(">>>>>Outlier Removing Completed<<<<<\n")
        

        logging.info(">>>>>Missing Value Imputation Started...<<<<<")
        missing_strategy = RemovingMissingValues(data=outlier, strategy=MissingValueStrategy())
        missing_imputed_df = missing_strategy.handle_missing_values()
        logging.info(">>>>>Missing Value Imputation Completed<<<<<\n")

        # logging.info(">>>>>Feature Selection Started...<<<<<")
        # feature_selection_strategy = FeatureSelection(data=missing_imputed_df, strategy=FeatureSelectionConfig())
        # final_df = feature_selection_strategy.handle_FS()
        # logging.info(">>>>>Feature Selection Completed<<<<<\n")
        print(type(missing_imputed_df))

        # missing_imputed_df.to_csv(data_cleaning_config.cleaned_gurgaon_data,index=False)

        return 


if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} started <<<<<<")
        obj = DataProcessingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
