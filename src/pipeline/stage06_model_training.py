import sys
import pandas as pd
from src.logger import logging
from src.exception import MyException
from src.model.model_selection import ModelTraining, ModelTrainingConfig
class ModelPipeline:
    def __init__(self):
        pass

    @staticmethod
    def main():
        
        df = pd.read_csv("gurgaon_properties_post_feature_selection_v2.csv")

        logging.info(">>>>>Model Training Started...<<<<<")
        model_training_strategy = ModelTraining(data=df, strategy=ModelTrainingConfig())

        training = model_training_strategy.handle_training()
        logging.info(">>>>>Model Training Completed<<<<<\n")

        return training
    
if __name__ == '__main__':
    try:
        logging.info(f"*******************")
        logging.info(f">>>>>> stage started <<<<<<")
        obj = ModelPipeline()
        obj.main()
        logging.info(f">>>>>> stage completed <<<<<<\nx==========x")
    except MyException as e:
            raise MyException(e, sys)
