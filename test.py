from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.pipeline.pipeline import Pipeline
from src.Heart_Attack_Risk_Analyzer_Project.config.config import Config
from src.Heart_Attack_Risk_Analyzer_Project.component.data_ingestion import DataIngestion
from src.Heart_Attack_Risk_Analyzer_Project.component.data_validation import DataValidation
from src.Heart_Attack_Risk_Analyzer_Project.component.data_transformation import DataTransformation
from src.Heart_Attack_Risk_Analyzer_Project.constant import *
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact
import evidently
# testing the pipeline
def main():
    try:
        pipeline = Pipeline()
        pipeline.start()
        logging.info("main function execution completed.")
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__=="__main__":
    main()

# testing componenets
# config_1 = Config(config_file_path=CONFIG_FILE_PATH, current_time_stamp=CURRENT_TIME_STAMP)
# data_ingestion_obj = DataIngestion(data_ingestion_config=config_1.get_data_ingestion_config())
# data_ingestion_artifact = data_ingestion_obj.initiate_data_ingestion()
# data_validation_obj = DataValidation(data_validation_config=config_1.get_data_validation_config(), data_ingestion_artifiact=data_ingestion_artifact)
# data_validation_artifact = data_validation_obj.initiate_data_validation()
# data_transformation_obj = DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact,
#                                              data_transformation_config=config_1.get_data_transformation_config())
# print(data_transformation_obj.initiate_data_transformation())