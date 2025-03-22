import sys

from src.constants import *
from src.logger import logging
from src.exception import MyException
from src.data.data_ingestion import IngestData
from src.config.configuration import ConfigurationManager

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = IngestData(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {INGESTION_STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except MyException as e:
            raise MyException(e, sys)
