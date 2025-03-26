import sys
import pandas as pd
from src.visualization.data_visualization import DataVisualization, Datavisualization
from src.config.configuration import ConfigurationManager
from src.logger import logging
from src.exception import MyException

class DataVizPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        config = ConfigurationManager()
        data_viz_config = config.get_data_visualization_config()
        data = pd.read_csv("artifacts/cleaned_data/missing_value_imputed.csv")

        logging.info(">>>>>Feature Selection Started...<<<<<")
        data_viz_strategy = DataVisualization(data=data, strategy=Datavisualization(), config=data_viz_config)

        final_df = data_viz_strategy.handle_viz()
        logging.info(">>>>>Feature Selection Completed<<<<<\n")

        return final_df
    
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage started <<<<<<")
        obj = DataVizPipeline()
        obj.main()
        logging.info(f">>>>>> stage completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
