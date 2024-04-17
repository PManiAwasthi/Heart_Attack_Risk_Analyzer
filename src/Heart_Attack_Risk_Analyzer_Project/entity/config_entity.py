from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig", 
["dataset_url", "zip_data_dir", "zip_file_name", "raw_data_dir", "ingested_train_dir", "ingested_test_dir",])

TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ["schema_file_path", "report_file_path", "report_page_file_path"])