import zipfile
import os, sys
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import yaml

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