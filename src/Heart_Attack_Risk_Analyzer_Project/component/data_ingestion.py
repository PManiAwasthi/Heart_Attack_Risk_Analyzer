from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataIngestionConfig
import sys, os
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import unzip_file
from dotenv import load_dotenv
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact

load_dotenv()

#this is imported afterwards to import the necessary kaggle credentials into the environment that will be used for authentication on import
from kaggle.api.kaggle_api_extended import KaggleApi

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started...{'='*20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def download_heart_risk_dataset(self,) -> str:
        try:
            #dataset url to download the data using kaggle api
            dowload_url = self.data_ingestion_config.dataset_url

            #folder location to download file
            zip_data_dir = self.data_ingestion_config.zip_data_dir

            if os.path.exists(zip_data_dir):
                os.remove(zip_data_dir)
            
            os.makedirs(zip_data_dir, exist_ok=True)

            try:
                logging.info("Dataset source authentication in progress...")
                api = KaggleApi()
                api.authenticate()
                logging.info("Dataset source authentication completed.")
            except Exception as e:
                raise HeartRiskException(e, sys)
            
            try:
                logging.info(f"Downloading dataset from : [{dowload_url}] into : [{zip_data_dir}]")
                api.dataset_download_files(dowload_url, path=zip_data_dir)
                logging.info(f"Download completed successfully. File available at : [{zip_data_dir}]")
            except Exception as e:
                raise HeartRiskException(e, sys)
            return zip_data_dir

        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def extract_zip_file(self):
        try:
            zip_data_dir = self.data_ingestion_config.zip_data_dir
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)
            
            os.makedirs(raw_data_dir, exist_ok=True)

            zip_file_path = os.path.join(self.data_ingestion_config.zip_data_dir,
                                         self.data_ingestion_config.zip_file_name)
            try:
                logging.info(f"Extracting the dataset zip file from : [{zip_file_path}] into : [{self.data_ingestion_config.raw_data_dir}]")
                unzip_file(zip_file_path=zip_file_path, extract_to=self.data_ingestion_config.raw_data_dir)
                logging.info(f"Extraction complete. Extracted data available at : [{self.data_ingestion_config.raw_data_dir}]")
            except Exception as e:
                raise HeartRiskException(e, sys)
            
            return zip_file_path
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def split_dataset_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            dataset_file_path = os.path.join(raw_data_dir, file_name)

            heart_data_frame = pd.read_csv(dataset_file_path)

            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

            logging.info("Spliting the dataset into train and test.")
            for train_index, test_index in split.split(heart_data_frame.drop(['TenYearCHD'], axis=1), heart_data_frame['TenYearCHD']):
                strat_train_set = heart_data_frame.loc[train_index]
                strat_test_set = heart_data_frame.loc[test_index]
            
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,
                                           file_name)
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,
                                          file_name)
            
            logging.info(f"Splitting completed. Datasets are available at train : [{train_file_path}] and test : [{test_file_path}]")
            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                strat_train_set.to_csv(train_file_path, index=False)
            
            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                strat_test_set.to_csv(test_file_path, index=False)
            
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                            test_file_path=test_file_path,
                                                            is_ingested=True,
                                                            message="Data Ingestion completed successfully")
            logging.info(f"DataIngestionArtifact generated : [{data_ingestion_artifact}]")
            return data_ingestion_artifact
        except Exception as e:
            raise HeartRiskException(e, sys)
        
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        try:
            zip_data_dir = self.download_heart_risk_dataset()
            self.extract_zip_file()
            return self.split_dataset_as_train_test()
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n\n")