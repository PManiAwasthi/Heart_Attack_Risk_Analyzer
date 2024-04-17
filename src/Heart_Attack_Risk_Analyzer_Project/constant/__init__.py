import os 
from datetime import datetime

ROOT_DIR = os.getcwd() #to fetch the current working directory

SOURCE_DIR = "src"
SOURCE_DIR_PATH = os.path.join(ROOT_DIR, "src")

PROJECT_DIR = "Heart_Attack_Risk_Analyzer_Project"
PROJECT_DIR_PATH = os.path.join(SOURCE_DIR_PATH, PROJECT_DIR)

CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

#training pipeline realted variables
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "pipeline_name"
TRAINING_PIPELINE_NAME_KEY = "artifact_dir"

#Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_ARTIFACT_DIR = "data_ingestion"
DATA_INGESTION_DOWNLOAD_URL_KEY = "dataset_url"
DATA_INGESTION_ZIP_DATA_DIR_KEY = "zip_data_dir"
DATA_INGESTSION_ZIP_DATA_FILE_NAME_KEY = "zip_file_name"
DATA_INGESTION_RAW_DATA_DIR_KEY = "raw_data_dir"
DATA_INGESTION_INGESTED_DIR_NAME_KEY = "ingested_dir"
DATA_INGESTION_TRAIN_DIR_KEY = "ingested_train_dir"
DATA_INGESTION_TEST_DIR_KEY = "ingested_test_dir"

EXPERIMENT_DIR_NAME = "experiment"

#Data Validation related variable
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_GET_ALL_COLUMNS_KEY = "columns"
DATA_VALIDATION_GET_TARGET_COLUMN_KEY = "target_column"
DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY = "categorical_columns"
DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY = "numerical_columns"