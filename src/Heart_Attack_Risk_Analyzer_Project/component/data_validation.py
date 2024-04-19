from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import os, sys
import pandas as pd
import json
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataValidationConfig
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import read_yaml_file
from src.Heart_Attack_Risk_Analyzer_Project.constant import *
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.metrics import *

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

            return is_available
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
    
    def get_and_save_data_drift_report(self):
        try:
            dataset_train, dataset_test = self.get_train_and_test_df()

            report = Report(metrics=[
                DataDriftPreset()
            ])
            report.run(reference_data=dataset_train, current_data=dataset_test)
            
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            report_json = json.loads(report.json())
            
            with open(report_file_path, 'w') as report_file:
                json.dump(report_json, report_file, indent = 6)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_file_dir= os.path.dirname(report_page_file_path)
            os.makedirs(report_page_file_dir, exist_ok=True)

            report.save_html(report_page_file_path)

            return report_json, report_file_path, report_page_file_path
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_and_save_data_test_report(self):
        try:
            dataset_train, dataset_test = self.get_train_and_test_df()
            tests = TestSuite(tests=[
                TestNumberOfColumnsWithMissingValues(),
                TestNumberOfRowsWithMissingValues(),
                TestNumberOfConstantColumns(),
                TestNumberOfDuplicatedRows(),
                TestNumberOfDuplicatedColumns(),
                TestColumnsType(),
                TestNumberOfDriftedColumns()
            ])
            tests.run(reference_data=dataset_train, current_data=dataset_test)

            test_json = json.loads(tests.json())
            test_file_path = self.data_validation_config.report_file_path
            test_file_dir = os.path.dirname(test_file_path)
            os.makedirs(test_file_dir, exist_ok= True)

            test_file_page_path = os.path.join(test_file_dir, "tests.html")
            tests.save_html(test_file_page_path)
            test_file_path = os.path.join(test_file_dir, "tests.json")
            with open(test_file_path, 'w') as test_file:
                json.dump(test_json, test_file, indent=6)

            return test_json, test_file_path, test_file_page_path
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def check_data_and_get_paths(self):
        try:
            #file existance check
            logging.info("Checking for the existance of files.")
            file_flag = self.is_train_test_file_exists()
            logging.info(f"Result of check [{file_flag}]")


            #schema check
            logging.info("Checking for correct schema.")
            schema_flag = self.validate_dataset_schema()
            logging.info(f"Schema check results are [{schema_flag}]")

            #data drift check
            logging.info("Checking for data drift.")
            report_json, report_file_path, report_file_page_path = self.get_and_save_data_drift_report()
            data_drift_test_result = report_json["metrics"][0]["result"]["dataset_drift"]
            logging.info(f"Is data drift found [{not data_drift_test_result}]")

            #other miscelleneous tests
            logging.info("Checking using different valid checks.")
            tests_json, tests_file_path, tests_file_page_path = self.get_and_save_data_test_report()
            tests_flag = True
            for test in tests_json["tests"]:
                if test['status'] != "SUCCESS":
                    tests_flag = False
            logging.info(f"Miscellaneous test results [{tests_flag}]")

            check_flag = False
            if file_flag and schema_flag and not data_drift_test_result and tests_flag:
                check_flag = True
            logging.info(f"Did the data pass all tests. [{check_flag}]")

            return check_flag, report_file_path, report_file_page_path, tests_file_path, tests_file_page_path
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            check_flag, report_file_path, report_file_page_path, tests_file_path, tests_file_page_path = self.check_data_and_get_paths()

            if check_flag:
                logging.info(f"The Dataset completed all the validation checks successfully.")
            else:
                logging.info("Dataset Failed the validation checks")
                raise Exception("Checks not successful")
            
            cat_features_list, num_features_list = self.get_cat_num_feature_list()

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=report_file_path,
                report_file_page_path=report_file_page_path,
                tests_file_path=tests_file_path,
                tests_file_page_path=tests_file_page_path,
                cat_features_list=cat_features_list,
                num_features_list=num_features_list
            )
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def __del__(self):
        logging.info(f"{'='*20}Data Validation log completed.{'='*20} \n\n")