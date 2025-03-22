import sys
import pandas as pd

from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.data.house_data_cleaning import DataCleaning, DataPreprocessStrategy
from src.config.configuration import ConfigurationManager
from src.utils.common import read_csv

class DataCleaningPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()
        house_data = pd.read_csv(data_cleaning_config.gurgaon_houses_data_path)
        house_data_cleaning = DataCleaning(data=house_data , strategy=DataPreprocessStrategy(), config=data_cleaning_config)
        house_cleaned_data = house_data_cleaning.handle_data()
        return house_cleaned_data


if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {CLEANING_STAGE_NAME} started <<<<<<")
        obj = DataCleaningPipeline()
        obj.main()
        logging.info(f">>>>>> stage {CLEANING_STAGE_NAME} completed <<<<<<\n\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
