import re
import sys
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from src.logger import logging
from src.exception import MyException


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values,
        and converts the data type to float.
        """
        try:
            logging.info("Starting data preprocessing")
            df = data.copy()
            df.insert(loc=3, column='sector',
                      value=df['property_name'].str.split('in').str.get(1).str.replace('Gurgaon', '').str.strip())
            
            df['sector'] = df['sector'].str.lower()
            logging.info("Converted sector names to lowercase")

            replacements = {
                'dharam colony': 'sector 12',
                'krishna colony': 'sector 7',
                'suncity': 'sector 54',
                'prem nagar': 'sector 13',
                'mg road': 'sector 28',
                'gandhi nagar': 'sector 28',
                'laxmi garden': 'sector 11',
                'shakti nagar': 'sector 11',
                'baldev nagar': 'sector 7',
                'shivpuri': 'sector 7',
                'garhi harsaru': 'sector 17',
                'imt manesar': 'manesar',
                'adarsh nagar': 'sector 12',
                'shivaji nagar': 'sector 11',
                'bhim nagar': 'sector 6',
                'madanpuri': 'sector 7'
            }
            df['sector'].replace(replacements, inplace=True)
            logging.info("Applied sector name replacements")

            a = df['sector'].value_counts()[df['sector'].value_counts() >= 3]
            df = df[df['sector'].isin(a.index)]
            logging.info("Filtered sectors with at least 3 occurrences")

            df.loc[955, 'sector'] = 'sector 37'
            df.loc[2800, 'sector'] = 'sector 92'
            df.loc[2838, 'sector'] = 'sector 90'
            df.loc[2857, 'sector'] = 'sector 76'
            logging.info("Updated specific sector values")

            df.loc[[311, 1072, 1486, 3040, 3875], 'sector'] = 'sector 110'
            logging.info("Updated batch sector values")

            df.drop(columns=['property_name', 'address', 'description', 'rating'], inplace=True)
            logging.info("Dropped unnecessary columns")

            return df

        except Exception as e:
            logging.error("Error occurred in Processing data", exc_info=True)
            raise MyException(e, sys)
    
    def extract_sector_number(self, sector_name):
        try:
            match = re.search(r'\d+', sector_name)
            if match:
                return int(match.group())
            else:
                return float('inf')  # Return a large number for non-numbered sectors
        except Exception as e:
            logging.error("Error occurred in extract sector number data", exc_info=True)
            raise MyException(e, sys)

class DataPreProcessing(DataStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        logging.info("Initializing DataPreProcessing with given strategy")
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        logging.info("Handling data using the provided strategy")
        return self.strategy.handle_data(self.df)
