import re
import sys
import pandas as pd

from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class FlatsDataStrategy(ABC):
    """
    Abstract Class defining strategy for handling flats data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FlatsDataPreProcessingStrategy(FlatsDataStrategy):
    """
    Data preprocessing strategy for cleaning and transforming flat-related datasets.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting flats data Pre-Processing...")
        try:
            df = data.copy()
            logging.debug("Original DataFrame shape: %s", df.shape)

            df.drop(columns=['link', 'property_id'], inplace=True)
            logging.debug("Dropped unnecessary columns")

            df.rename(columns={'area': 'price_per_sqft'}, inplace=True)
            logging.debug("Renamed 'area' to 'price_per_sqft'")

            df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()
            logging.debug("Cleaned 'society' column")

            df = df[df['price'] != 'Price on Request']
            logging.debug("Filtered out 'Price on Request'")

            df['price'] = df['price'].str.split(' ').apply(self.treat_price)
            logging.debug("Processed 'price' column")

            df['price_per_sqft'] = (
                df['price_per_sqft']
                .str.split('/')
                .str.get(0)
                .str.replace('₹', '')
                .str.replace(',', '')
                .str.strip()
                .astype(float)
            )
            logging.debug("Processed 'price_per_sqft' column")

            df = df[~df['bedRoom'].isnull()]
            df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype(int)
            logging.debug("Processed 'bedRoom' column")

            df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype(int)
            logging.debug("Processed 'bathroom' column")

            df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')
            logging.debug("Processed 'balcony' column")

            df['additionalRoom'].fillna('not available', inplace=True)
            df['additionalRoom'] = df['additionalRoom'].str.lower()
            logging.debug("Processed 'additionalRoom' column")

            df['floorNum'] = (
                df['floorNum']
                .str.split(' ')
                .str.get(0)
                .replace('Ground', '0')
                .str.replace('Basement', '-1')
                .str.replace('Lower', '0')
                .str.extract(r'(\d+)')
            )
            logging.debug("Processed 'floorNum' column")

            df['facing'].fillna('NA', inplace=True)
            logging.debug("Handled missing values in 'facing'")

            df.insert(loc=4, column='area', value=round((df['price'] * 10000000) / df['price_per_sqft']))
            logging.debug("Calculated 'area'")

            df.insert(loc=1, column='property_type', value='flat')
            logging.debug("Added 'property_type' column")

            logging.info("Flats data pre-processing completed successfully.")
            return df
        except Exception as e:
            logging.error("Error occurred in cleaning flats data", exc_info=True)
            raise MyException(e)

    def treat_price(self, x):
        if isinstance(x, float):
            return x
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)
        return round(float(x[0]), 2)

class FlatsDataCleaning(FlatsDataStrategy):
    def __init__(self, data: pd.DataFrame, strategy: FlatsDataStrategy, config) -> None:
        self.config = config
        self.df = data
        self.strategy = strategy
        logging.debug("Initialized FlatsDataCleaning with strategy: %s", type(strategy).__name__)
    
    def handle_data(self) -> pd.DataFrame:
        logging.info("Starting data cleaning process...")
        cleaned_data = self.strategy.handle_data(self.df)
        logging.info("Data cleaning process completed successfully.")
        return cleaned_data
