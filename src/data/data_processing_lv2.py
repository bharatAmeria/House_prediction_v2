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
            df = data
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
                'madanpuri': 'sector 7',
                'saraswati vihar':'sector 28',
                'arjun nagar':'sector 8',
                'ravi nagar':'sector 9',
                'vishnu garden':'sector 105',
                'bhondsi':'sector 11',
                'surya vihar':'sector 21',
                'devilal colony':'sector 9',
                'valley view estate':'gwal pahari',
                'mehrauli  road':'sector 14',
                'jyoti park':'sector 7',
                'ansal plaza':'sector 23',
                'dayanand colony':'sector 6',
                'sushant lok phase 2':'sector 55',
                'chakkarpur':'sector 28',
                'greenwood city':'sector 45',
                'subhash nagar':'sector 12',
                'sohna road road':'sohna road',
                'malibu town':'sector 47',
                'surat nagar 1':'sector 104',
                'new colony':'sector 7',
                'mianwali colony':'sector 12',
                'jacobpura':'sector 12',
                'rajiv nagar':'sector 13',
                'ashok vihar':'sector 3',
                'dlf phase 1':'sector 26',
                'nirvana country':'sector 50',
                'palam vihar':'sector 2',
                'dlf phase 2':'sector 25',
                'sushant lok phase 1':'sector 43',
                'laxman vihar':'sector 4',
                'dlf phase 4':'sector 28',
                'dlf phase 3':'sector 24',
                'sushant lok phase 3':'sector 57',
                'dlf phase 5':'sector 43',
                'rajendra park':'sector 105',
                'uppals southend':'sector 49',
                'sohna':'sohna road',
                'ashok vihar phase 3 extension':'sector 5',
                'south city 1':'sector 41',
                'ashok vihar phase 2':'sector 5',
                'sector 95a':'sector 95',
                'sector 23a':'sector 23',
                'sector 12a':'sector 12',
                'sector 3a':'sector 3',
                'sector 110 a':'sector 110',
                'patel nagar':'sector 15',
                'a block sector 43':'sector 43',
                'maruti kunj':'sector 12',
                'b block sector 43':'sector 43',
                'sector-33 sohna road':'sector 33',
                'sector 1 manesar':'manesar',
                'sector 4 phase 2':'sector 4',
                'sector 1a manesar':'manesar',
                'c block sector 43':'sector 43',
                'sector 89 a':'sector 89',
                'sector 2 extension':'sector 2',
                'sector 36 sohna road':'sector 36',
            }
            df['sector'].replace(replacements, inplace=True)
            logging.info("Applied sector name replacements")

            df[df['sector'] == 'new']
            df.loc[955, 'sector'] = 'sector 37'
            df.loc[2800, 'sector'] = 'sector 92'
            df.loc[2838, 'sector'] = 'sector 90'
            df.loc[2857, 'sector'] = 'sector 76'
            logging.info("Updated specific sector values")

            df[df['sector'] == 'new sector 2']
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
