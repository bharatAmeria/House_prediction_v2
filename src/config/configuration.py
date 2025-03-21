from src.entity.config_entity import DataIngestionConfig
from src.utils.common import read_yaml, create_directories
from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            gurgaon_data_path= config.gurgaon_houses_data_path,
            gurgaon_flats_data_path= config.gurgaon_flats_data_path,
            gurgaon_appartments_data_path = config.gurgaon_appartments_data_path,
            gurgaon_houses_data_path= config.gurgaon_houses_data_path
        )

        return data_ingestion_config

    # def get_data_transformation_config(self) -> DataTransformationConfig:
    #     config = self.config.data_transformation

    #     create_directories([config.root_dir])

    #     data_transformation_config = DataTransformationConfig(
    #         root_dir=config.root_dir,
    #         directory=config.directory,
    #     )



    # def get_data_divider(self) -> DataDividerConfig:
    #     config = self.config.data_divider

    #     create_directories([config.root_dir])
    #     data_divider = DataDividerConfig(
    #         root_dir=config.root_dir,
    #         train_dir=config.train_dir,
    #         test_dir=config.test_dir,
    #         params_test_size=self.params.TEST_SIZE,
    #         params_train_size=self.params.TRAIN_SIZE,
    #         params_random_state=self.params.RANDOM_STATE,
    #     )

    #     return data_divider

    # def get_evaluation_config(self) -> EvaluationConfig:
    #     eval_config = EvaluationConfig(
    #         path_of_model="artifacts/training/model.h5",
    #         training_data="artifacts/data_ingestion/gurgaon_properties/gurgaon_properties.csv",
    #         mlflow_uri=" ",
    #         all_params=self.params,
    #     )
    #     return eval_config

