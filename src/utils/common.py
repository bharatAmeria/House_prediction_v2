import json
import pickle
import pandas as pd 
import numpy as np
from box import ConfigBox
import yaml
from pathlib import Path
from src.logger import logging
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import logger
from ensure import ensure_annotations
import os
from box.exceptions import BoxValueError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
        :param path_to_directories:
        :param verbose:
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.exception(e)
        raise e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.exception(e)
        raise e


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.exception(e)
        raise e


def scorer(model_name, model, preprocessor=None, y_transformed=None, X=None):
    output = []
    output.append(model_name)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # K-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y_transformed, cv=kfold, scoring='r2')

    output.append(scores.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_pred = np.expm1(y_pred)

    output.append(mean_absolute_error(np.expm1(y_test), y_pred))

    return output


@ensure_annotations
def read_csv(path_to_csv: Path) -> pd.DataFrame:
    """Reads a CSV file and returns a DataFrame.

    Args:
        path_to_csv (Path): Path to the CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV file is empty.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the data.
    """
    try:
        if not path_to_csv.exists():
            raise FileNotFoundError(f"CSV file not found: {path_to_csv}")

        df = pd.read_csv(path_to_csv)

        if df.empty:
            raise ValueError("CSV file is empty")

        logging.info(f"CSV file loaded successfully from: {path_to_csv}")
        return df

    except Exception as e:
        logging.exception(f"Error reading CSV file: {path_to_csv}")
        raise e


@ensure_annotations
def save_csv(df: pd.DataFrame, path_to_csv: Path, index: bool = False):
    """Saves a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        path_to_csv (Path): Path to save the CSV file.
        index (bool, optional): Whether to include the index. Defaults to False.

    Raises:
        ValueError: If DataFrame is empty.
    """
    try:
        if df.empty:
            raise ValueError("DataFrame is empty, cannot save to CSV")

        dir_path = path_to_csv.parent
        os.makedirs(dir_path, exist_ok=True)

        df.to_csv(path_to_csv, index=index)
        logging.info(f"CSV file saved at: {path_to_csv}")

    except Exception as e:
        logging.exception(f"Error saving CSV file: {path_to_csv}")
        raise e
