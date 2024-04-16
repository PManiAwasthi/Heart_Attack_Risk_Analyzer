from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.pipeline.pipeline import Pipeline
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