import sys
import pandas as pd
import numpy as np

from typing import Union
from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from pathlib import Path

class MissingValueImputationStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_missing_values(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
   
class MissingValueStrategy(MissingValueImputationStrategy):
    """
    Fixing Missing Values
    """
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data

            all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]
            super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()
            carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()

            # both present built up null
            sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
            sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/super_to_built_up_ratio) + (sbc_df['carpet_area']/carpet_to_built_up_ratio))/2),inplace=True)
            df.update(sbc_df)

            # sb present c is null built up null
            sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
            sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/super_to_built_up_ratio),inplace=True)
            df.update(sb_df)

            # sb null c is present built up null
            c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
            c_df['built_up_area'].fillna(round(c_df['carpet_area']/carpet_to_built_up_ratio),inplace=True)
            df.update(c_df)

            anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]
            anamoly_df['built_up_area'] = anamoly_df['area']
            df.update(anamoly_df)

            df.drop(columns=['area',
                             'areaWithType',
                             'super_built_up_area',
                             'carpet_area'],inplace=True)
            
            # Floor Num
            df['floorNum'].fillna(2.0,inplace=True)
            df.drop(columns=['facing'],inplace=True)
            df.drop(index=[2536],inplace=True)

            # Age Possession
            df[df['agePossession'] == 'Undefined']
            df['agePossession'] = df.apply(lambda row: self.mode_based_imputation(row, df), axis=1)
            df['agePossession'] = df.apply(lambda row: self.mode_based_imputation2(row, df), axis=1)
            df['agePossession'] = df.apply(lambda row: self.mode_based_imputation3(row, df), axis=1)
            
            return

        except Exception as e:
              logging.error("Error occurred in Imputing missing values", exc_info=True)
              raise MyException(e, sys)
        
    def mode_based_imputation(self, row, df):
        try:
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
        except Exception as e:
            logging.error("Error occurred in mode based imputation function 1", exc_info=True)
            raise MyException(e, sys)

            
    def mode_based_imputation2(self, row, df):
        try:
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
        except Exception as e:
            logging.error("Error occurred in mode based imputation function 2", exc_info=True)
            raise MyException(e, sys)

       
    def mode_based_imputation3(self, row, df):
        try:
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
        except Exception as e:
            logging.error("Error occurred in mode based imputation function 3", exc_info=True)
            raise MyException(e, sys)
        
class RemovingMissingValues:

    def __init__(self, data: pd.DataFrame, strategy: MissingValueImputationStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_missing_values(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_missing_values(self.df)