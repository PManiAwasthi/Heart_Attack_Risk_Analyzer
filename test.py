from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
import os, sys
logging.info("testing exception")

try:
    raise Exception("This is a test Exception")
except Exception as e:
    raise HeartRiskException(e,sys) from e