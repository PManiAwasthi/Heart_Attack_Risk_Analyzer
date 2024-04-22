from src.Heart_Attack_Risk_Analyzer_Project.logger import logging, get_log_file_name
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import os, sys
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.Heart_Attack_Risk_Analyzer_Project.component.data_ingestion import DataIngestion
from src.Heart_Attack_Risk_Analyzer_Project.component.data_validation import DataValidation
from src.Heart_Attack_Risk_Analyzer_Project.component.data_transformation import DataTransformation
from collections import namedtuple
from threading import Thread
from src.Heart_Attack_Risk_Analyzer_Project.config.config import Config
import uuid
from src.Heart_Attack_Risk_Analyzer_Project.constant import EXPERIMENT_DIR_NAME
from datetime import datetime
import pandas as pd

Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "log_file_name",
                                       "running_status", "start_time", "stop_time", "execution_time",
                                       "message", "experiment_file_path"])

class Pipeline(Thread):

    experiment:Experiment=Experiment(*[None]*9)

    def __new__(cls, *args, **kwargs):
        if Pipeline.experiment.running_status:
            raise Exception("Pipeline is already running.")
        return super(Pipeline, cls).__new__(cls)
    
    def __init__(self, config: Config = Config()) -> None:
        try:
            super().__init__(daemon=False, name="pipeline")
            self.config = config
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise HeartRiskException(e, sys)

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(), data_ingestion_artifiact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact,
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_validation_artifact=data_validation_artifact,
                                                     data_transformation_config=self.config.get_data_transformation_config())
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise HeartRiskException(e, sys)

    def run_pipeline(self):
        try:
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            
            logging.info("Pipeline Starting.")

            experiment_id = str(uuid.uuid4())
            aritifact_dir = os.path.join(self.config.training_pipeline_config.artifact_dir, EXPERIMENT_DIR_NAME)
            os.makedirs(aritifact_dir, exist_ok=True)

            file_name = f"experiemnt-{experiment_id}.csv"
            experiment_file_path = os.path.join(aritifact_dir, file_name)

            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             log_file_name=get_log_file_name(self.config.time_stamp),
                                             running_status=True,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             experiment_file_path=experiment_file_path,
                                             message="Pipeline has been started.")
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()
            data_ingestion_artifact = self.start_data_ingestion()
            print(data_ingestion_artifact)
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            print(data_validation_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                          data_validation_artifact=data_validation_artifact)
            print(data_transformation_artifact)
            
            logging.info(f"Pipeline Completed.")
            stop_time = datetime.now()

            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
            initialization_timestamp=self.config.time_stamp,
            log_file_name=get_log_file_name(self.config.time_stamp),
            running_status=True,
            start_time=Pipeline.experiment.start_time,
            stop_time=stop_time,
            execution_time=stop_time-Pipeline.experiment.start_time,
            experiment_file_path=Pipeline.experiment.experiment_file_path,
            message="Pipeline has been completed."
            )
            logging.info(f"Pipeline Experiment: {Pipeline.experiment}")

            self.save_experiment()
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def run(self):
        try:
            self.run_pipeline()
        except Exception as e:
            raise e
    
    def save_experiment(self):
        try:
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_report = pd.DataFrame(zip(experiment._fields, experiment))
                experiment_report.to_csv(experiment.experiment_file_path, mode='w', index=False, header=False)
            else:
                print("First start the experiment.")
        except Exception as e:
            raise HeartRiskException(e, sys)