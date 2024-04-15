from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.entity.config_entity import DataIngestionConfig
import sys, os
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started...{'='*20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def download_heart_risk_dataset(self,) -> str:
        try:
            pass
        except Exception as e:
            raise HeartRiskException(e, sys)