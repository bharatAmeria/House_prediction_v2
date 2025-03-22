import sys
import pandas as pd

from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.data.house_data_cleaning import HouseDataCleaning, HouseDataPreProcessingStrategy
from src.data.flats_data_cleaning import FlatsDataCleaning, FlatsDataPreProcessingStrategy
from src.config.configuration import ConfigurationManager

class DataCleaningPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()

        logging.info(">>>>>Housing Data Preprocessing Started<<<<<")
        house_data = pd.read_csv(data_cleaning_config.gurgaon_houses_data)
        house_data_cleaning = HouseDataCleaning(data=house_data, 
                                           strategy=HouseDataPreProcessingStrategy(), 
                                           config=data_cleaning_config
                                           )
        house_cleaned_data = house_data_cleaning.handle_data()

        logging.info(">>>>>Flats Data Preprocessing Started<<<<<")
        flats_data = pd.read_csv(data_cleaning_config.gurgaon_flats_data)
        flats_data_cleaning = FlatsDataCleaning(data=flats_data,
                                                strategy=FlatsDataPreProcessingStrategy(),
                                                config=data_cleaning_config
                                                )
        flats_cleaned_data = flats_data_cleaning.handle_data()

        merged_data = pd.concat([flats_cleaned_data, house_cleaned_data],ignore_index=True)

        
        return merged_data


if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} started <<<<<<")
        obj = DataCleaningPipeline()
        obj.main()
        logging.info(f">>>>>> stage {PRE_PROCESSING_STAGE_NAME} completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
