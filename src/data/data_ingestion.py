import os
import sys
import zipfile

import gdown
import pandas as pd

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

    def download_file(self):
        """ Fetch data from the url """

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, zip_download_dir)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise MyException(e, sys)

    def extract_zip_file(self):
        """
            zip_file_path: str
            Extracts the zip file into the data directory
            Function returns None
            """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        return df

