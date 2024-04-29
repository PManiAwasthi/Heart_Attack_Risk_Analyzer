import numpy as np
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from collections import namedtuple
import yaml
import os, sys

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELCTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = 'search_papram_grid'

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchBestModel = namedtuple("GridSearchBestModel", 
                                 ["model_serial_number", "model", "best_model", "best_parameters", "best_score"])

BestModel = namedtuple("BestModel", 
                       ["model_serial_number", "model", "best_model", "best_parameters", "best_score"])


class ModelFactory:
    def __init__(self, model_config_path: str = None):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = self.config[GRID_SEARCH_KEY][PARAM_KEY]

            self.models_initialization_config: dict = dict(self.config[MODEL_SELCTION_KEY])
            self.initialized_model_list = None
            self.grid_search_best_model_list = None
            
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path, 'r') as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise HeartRiskException(e, sys)