import sys
import pandas as pd
from src.config.configuration import ConfigurationManager
from src.recommender_system import RecommenderSystem, RecommenderSystemConfig
from src.exception import MyException
from src.logger import logging

class RecoomendationSystemPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        # config = ConfigurationManager()
        # get_recommend_config = config.get_recommend_sys_config()
        data = pd.read_csv("artifacts/gurgaon_data/appartments.csv")

        logging.info(">>>>>Recommender System  Started...<<<<<")
        data_viz_strategy = RecommenderSystem(data=data, strategy=RecommenderSystemConfig())

        reommend = data_viz_strategy.handle_recommend()
        logging.info(">>>>>Recommender System Completed<<<<<\n")

        return reommend
    
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage started <<<<<<")
        obj = RecoomendationSystemPipeline()
        obj.main()
        logging.info(f">>>>>> stage completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
