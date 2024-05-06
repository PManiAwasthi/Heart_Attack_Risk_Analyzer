import numpy as np
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from collections import namedtuple
import yaml
import os, sys
from typing import List
import importlib
from sklearn.metrics import recall_score

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
        """
        this function is used to import the required class of the model and get the model reference to return to the
        get_initialized_model_list function which then makes a list of all such model references.
        """
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
        """
        This function updates the parameters of the model objects for the get_initialized_model_list function
        which then obtains the models with updated parameters to prepare the final model object list of type InitializedModelDetial
        """
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
        and return the list to the get_best_model function.
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY][CLASS_KEY])
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name)
                initialized_model_list.append(model_initialization_config)
            
            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def execute_grid_search_operation(self,
                                      initialized_model: InitializedModelDetail,
                                      input_feature,
                                      output_feature) -> GridSearchBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class
            message = "*" * 50, f"training {type(initialized_model.model).__name__}", "*" * 50
            logging.info(message)
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)

            grid_search_cv.fit(input_feature, output_feature)

            grid_search_best_model = GridSearchBestModel(
                model_serial_number=initialized_model.model_serial_number,
                model=initialized_model.model,
                best_model=grid_search_cv.best_estimator_,
                best_parameters=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )
            return grid_search_best_model
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def initiate_best_parameter_search_for_initialized_model(self,
                                                             initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchBestModel:
        """
        This purpose of this function is to call execute_grid_search_operation for the supplied InitializedModelDetail object.
        This is so because each InitializedModelDetail objects contains a model to be trained and the execute_grid_search_operation
        runs the grid_search and returns the result in a GridSearchBestModel object format.
        This function then returns these to the initiate_best_parameter_search_for_initialized_models function so that they can be 
        appended to the List[GridSearchBestModel]
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def initiate_best_parameter_search_for_initialized_models(self, 
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchBestModel]:
        """
        this function executes after the get_initialized_model_list function and is takes list of initialized models as input,
        its purpose is to iteratively call the initiate_best_parameter_search_for_initialized_model for all the models in the 
        input list. And make a list of all the models then returned by the aforementioned function and return them to get_best_model 
        function.
        """
        try:
            self.grid_searched_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_search_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_search_best_model_list.append(grid_search_best_model)
            return self.grid_search_best_model_list
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    @staticmethod
    def get_recall_grid_searched_best_model_list(bestModel: GridSearchBestModel, input_features, output_features) -> float:
        try:
            best_estimator = bestModel.best_model
            y_pred = best_estimator.predict(input_features)
            recall = recall_score(output_features, y_pred)
            return recall
        except Exception as e:
            raise HeartRiskException(e, sys)

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_serach_best_model_list: List[GridSearchBestModel], 
                                                          input_features, output_features, base_accuracy = 1) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_serach_best_model_list:
                recall = ModelFactory.get_recall_grid_searched_best_model_list(grid_searched_best_model, input_features=input_features,
                                                                               output_features=output_features)
                accuracy = grid_searched_best_model.best_model.best_score_
                score = abs(recall-accuracy)
                if base_accuracy > score:
                    logging.info(f"Acceptable Model Found: {grid_searched_best_model}")
                    base_accuracy = score
                    best_model = grid_searched_best_model
                if not best_model:
                    raise Exception(f"None of the Models are acceptable")
                logging.info(f"Best Model: {best_model}")
            return best_model
                
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized Model: {initialized_model_list}")
            grid_search_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list()

        except Exception as e:
            raise HeartRiskException(e, sys)