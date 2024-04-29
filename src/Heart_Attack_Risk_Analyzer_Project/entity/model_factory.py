import numpy as np
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from collections import namedtuple
import yaml
import os, sys
from typing import List
import importlib

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
    
    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            # load the module, will raise an Exception if the module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise an Exception if the class is not found
            class_ref = getattr(module_name, class_name)

            return class_ref
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    @staticmethod
    def update_property_of_class(instance_ref, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        this function is to read the model_config dictionary and return a list of model details.
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY][CLASS_KEY])
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class()
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        try:
            initialized_model_list = self.get_initialized_model_list()
        except Exception as e:
            raise HeartRiskException(e, sys)