from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import os, sys
import pandas as pd
import json
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataValidationConfig
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import read_yaml_file
from src.Heart_Attack_Risk_Analyzer_Project.constant import *

class DataValidation:

    def __init__(self, data_validation_config:DataValidationConfig,
                 data_ingestion_artifiact:DataIngestionArtifact):
        try:
            logging.info(f"{'='*20} Data Validation log started. {'='*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifiact
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def is_train_test_file_exists(self) -> bool:
        try:
            logging.info("Checking if training and test file is available")
            is_train_file_exists = False
            is_test_file_exists = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exists = os.path.exists(train_file_path)
            is_test_file_exists = os.path.exists(test_file_path)

            is_available = is_train_file_exists and is_test_file_exists

            logging.info(f"Is train and test file exists? -> {is_available}")

            if not is_available:
                message=f"Training file: [ {train_file_path} ] or Testing File: [ {test_file_path} ] is not present"
                raise Exception(message)
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def validate_dataset_schema(self) -> bool:
        try:
            train_set, _ = self.get_train_and_test_df()
            
            #getting schema structure
            schema_structure = read_yaml_file(config_file_path=self.data_validation_config.schema_file_path)
            columns = schema_structure[DATA_VALIDATION_GET_ALL_COLUMNS_KEY]

            schema_feature_datatype = {}
            for schema_feature, schema_datatype in columns.items():
                schema_feature_datatype[schema_feature] = schema_datatype
            
            #getting dataset structure
            train_set_feature_datatype = {}
            for feature in train_set.columns:
                train_set_feature_datatype[feature] = str(train_set[feature].dtypes)
            
            flag = True
            for feature in schema_feature_datatype.keys():
                if train_set_feature_datatype[feature] != schema_feature_datatype[feature]:
                    flag = False
                    raise Exception(f"column {feature} or datatype {train_set_feature_datatype[feature]} didn't match the requirements.")
            
            return flag


        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_cat_num_feature_list(self):
        try:
            train_dataset, _ = self.get_train_and_test_df()

            categorical_feature_list = []
            numerical_feature_list = []
            for feature in train_dataset.columns:
                if train_dataset[feature].nunique() < 5:
                    categorical_feature_list.append(feature)
                else:
                    numerical_feature_list.append(feature)
            
            return categorical_feature_list, numerical_feature_list
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    
    def __del__(self):
        logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")