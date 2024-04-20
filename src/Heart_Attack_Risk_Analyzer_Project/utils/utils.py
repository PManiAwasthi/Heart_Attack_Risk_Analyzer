import zipfile
import os, sys
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import yaml
import pandas as pd
import numpy as np
import dill
from src.Heart_Attack_Risk_Analyzer_Project.constant import DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY, DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY

def unzip_file(zip_file_path:str, extract_to:str) -> None:
    """
    This function extracts the zip file content.
    zip_file_path: path to the zip file to be extracted
    extract_to: path where the extracted content will be placed
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def read_yaml_file(config_file_path:str)->dict:
    """
    This function is used to read a yaml file and return its content
    config_file_path: path where the config file is placed in the project directory
    """
    try:
        with open(config_file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HeartRiskException(e, sys)

def load_data_helper(dataframe: pd.DataFrame, schema_type: list, data_type: str) -> pd.DataFrame:
    try:
        error_message = ""
        for column in schema_type:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(data_type)
            else:
                error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
        if len(error_message) > 0:
            raise Exception(error_message)
        return dataframe
    except Exception as e:
        raise HeartRiskException(e, sys)


def load_data(file_path:str, schema_file_path: str) -> pd.DataFrame:
    """
    this function loads the data as per the given schema and raises an exception in case the dataframe doesn't 
    match the given schema
    file_path: str location to load data from
    schema_file_path: str file to check the specified column type and other properties
    """
    try:
        dataset_schema = read_yaml_file(schema_file_path)

        schema_num_columns = dataset_schema[DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY]
        schema_cat_columns = dataset_schema[DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY]

        dataframe = pd.read_csv(file_path)

        error_message = ""

        dataframe = load_data_helper(dataframe=dataframe, schema_type=schema_num_columns, data_type="float")
        dataframe = load_data_helper(dataframe=dataframe, schema_type=schema_cat_columns, data_type="object")
        
        return dataframe
    except Exception as e:
        raise HeartRiskException(e, sys)

def save_numpy_array_data(file_path:str, array:np.array):
    """
    Save the supplied array data to file
    file_path: str location of the file to save
    array: np.array data to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HeartRiskException(e, sys)

def numpy_array_data(file_path: str) -> np.array:
    """
    loads the numpy array from a file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HeartRiskException(e, sys)

def save_object(file_path: str, obj):
    """
    save any kind of object to the specified file_path
    file_path: str location to save file at
    obj: any object that is to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise HeartRiskException(e, sys)