import os
import sys
import zipfile

import gdown

from src.logger import logging
from src.exception import MyException

from src.entity.config_entity import DataIngestionConfig


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, config: DataIngestionConfig):
        """Initialize the data ingestion class."""
        self.config = config
        logging.info("Data Ingestion class initialized.")

    def download_file(self):
        """ Fetch data from the URL """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logging.info(f"Successfully downloaded data from {dataset_url} into file {zip_download_dir}")
        except Exception as e:
            logging.error("Error occurred while downloading file", exc_info=True)
            raise MyException(e, sys)

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            logging.info(f"Extracting zip file {self.config.local_data_file} to {unzip_path}")
            
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            
            logging.info(f"Successfully extracted zip file to {unzip_path}")
        except Exception as e:
            logging.error("Error occurred while extracting zip file", exc_info=True)
            raise MyException(e, sys)