from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import os
import sys
from src.Heart_Attack_Risk_Analyzer_Project.constant import *
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import read_yaml_file

class Config:
    def __init__(self, config_file_path:str = CONFIG_FILE_PATH,
                 current_time_stamp:str = CURRENT_TIME_STAMP) -> None:
        try:
            self.config_info = read_yaml_file(config_file_path=config_file_path)
            self.training_pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = current_time_stamp
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir
            data_ingestion_artifact_dir = os.path.join(artifact_dir,
                                                       DATA_INGESTION_ARTIFACT_DIR,
                                                       self.time_stamp)
            data_ingestion_info = self.config_info[DATA_INGESTION_CONFIG_KEY]

            dataset_url = data_ingestion_info[DATA_INGESTION_DOWNLOAD_URL_KEY]
            zip_data_dir = os.path.join(
                data_ingestion_artifact_dir,
                data_ingestion_info[DATA_INGESTION_ZIP_DATA_DIR_KEY]
            )
            raw_data_dir = os.path.join(data_ingestion_artifact_dir,
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ingested_data_dir = os.path.join(data_ingestion_artifact_dir,
                                             data_ingestion_info[DATA_INGESTION_INGESTED_DIR_NAME_KEY])
            
            ingested_train_dir = os.path.join(ingested_data_dir,
                                             data_ingestion_info[DATA_INGESTION_TRAIN_DIR_KEY])
            
            ingested_test_dir = os.path.join(ingested_data_dir,
                                             data_ingestion_info[DATA_INGESTION_TEST_DIR_KEY])
            
            zip_file_name = data_ingestion_info[DATA_INGESTSION_ZIP_DATA_FILE_NAME_KEY]
            
            data_ingestion_config = DataIngestionConfig(dataset_url=dataset_url,
                                                        zip_data_dir=zip_data_dir,
                                                        zip_file_name=zip_file_name,
                                                        raw_data_dir=raw_data_dir,
                                                        ingested_train_dir=ingested_train_dir,
                                                        ingested_test_dir=ingested_test_dir
                                                        )
            logging.info(f"Data Ingestion config: {data_ingestion_config}")
            return data_ingestion_config
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_validation_artifact_dir = os.path.join(artifact_dir,
                                                        DATA_VALIDATION_ARTIFACT_DIR_NAME,
                                                        self.time_stamp)
            data_validation_config = self.config_info[DATA_VALIDATION_CONFIG_KEY]

            schema_file_path = os.path.join(ROOT_DIR,
                                            data_validation_config[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                            data_validation_config[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            
            report_file_path = os.path.join(data_validation_artifact_dir,
                                            data_validation_config[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            
            report_page_file_path = os.path.join(data_validation_artifact_dir,
                                                 data_validation_config[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY])
            
            data_validation_config = DataValidationConfig(
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path
            )
            logging.info(f"Data Validation config : [{data_validation_config}]")
            return data_validation_config
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_data_transformation_config(self):
        try:
            artifact_dir = self.training_pipeline_config.artifact_dir

            data_transformation_artifact_dir = os.path.join(artifact_dir,
                                                            DATA_TRANSFORMATION_ARTIFACT_DIR,
                                                            self.time_stamp)
            
            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]

            convert_features_to_object = data_transformation_config_info[DATA_TRANSFORMATION_CONVERT_FEATURES_TO_OBJECTS]

            preprocessed_object_file_path = os.path.join(data_transformation_artifact_dir,
                                                         data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY],
                                                         data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_OBJECT_FILE_NAME_KEY])
            transformed_train_dir = os.path.join(data_transformation_artifact_dir,
                                                 data_transformation_config_info[DATA_TRANSFORMATION_DIR_KEY],
                                                 data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_KEY])
            
            transformed_test_dir = os.path.join(data_transformation_artifact_dir,
                                                data_transformation_config_info[DATA_TRANSFORMATION_DIR_KEY],
                                                data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_KEY])
            
            change_feature_male_to_gender = data_transformation_config_info[DATA_TRANSFORMATION_CHANGE_FEATURE_NAME_TO_GENDER_KEY]

            data_transformation_config = DataTransformationConfig(preprocessed_object_file_path=preprocessed_object_file_path,
                                                                  transformed_train_dir=transformed_train_dir,
                                                                  transformed_test_dir=transformed_test_dir,
                                                                  convert_features_to_object=convert_features_to_object,
                                                                  change_feature_male_to_gender=change_feature_male_to_gender)
            logging.info(f"Data Transformation Config: {data_transformation_config}")
            return data_transformation_config

        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(PROJECT_DIR_PATH,
                                        training_pipeline_config[TRAINING_PIPELINE_NAME_KEY],
                                        training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY])
            
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir)
            logging.info(f"Training Pipeline Config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise HeartRiskException(e, sys)

