import pandas as pd
from src.config.configuration import ConfigurationManager
from src.feature_engg.feature_selection import FeatureSelection, FeatureSelectionConfig
from src.logger import logging
from src.exception import MyException
from src.constants import FEATURE_SELECTION_STAGE
import sys

class FeatureSelectionPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config()

        cleaned_data = pd.read_csv(data_cleaning_config.cleaned_gurgaon_data)

        logging.info(">>>>>Feature Selection Started...<<<<<")
        feature_selection_strategy = FeatureSelection(data=cleaned_data, strategy=FeatureSelectionConfig(), config=data_cleaning_config)
        final_df = feature_selection_strategy.handle_FS()
        logging.info(">>>>>Feature Selection Completed<<<<<\n")

        return final_df
    
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} started <<<<<<")
        obj = FeatureSelectionPipeline()
        obj.main()
        logging.info(f">>>>>> stage {FEATURE_SELECTION_STAGE} completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
