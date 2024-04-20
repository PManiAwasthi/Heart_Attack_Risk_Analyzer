from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.Heart_Attack_Risk_Analyzer_Project.config.config import DataTransformationConfig
import os,sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import load_data, read_yaml_file, save_numpy_array_data, save_object
from src.Heart_Attack_Risk_Analyzer_Project.constant import DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY, DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY, DATA_VALIDATION_GET_TARGET_COLUMN_KEY


class ReSampling(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        try:
            smote_obj = SMOTE()
            adasyn_obj = ADASYN()

            X, y = smote_obj.fit_resample(X.drop(column=["TenYearCHD"], axis=1), X["TenYearCHD"])
            X, y = adasyn_obj.fit_resample(X, y)
            generated_data = np.c_[X, y]
            return generated_data
        except Exception as e:
            raise HeartRiskException(e, sys)

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            logging.info(f"{'='*20} Data Transformation started. {'='*20}")
            self.data_transformation_config= data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(config_file_path=schema_file_path)

            numerical_columns = dataset_schema[DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY]

            column_list = numerical_columns + categorical_columns

            # pipeline = Pipeline(steps=[
            #     ('imputer', KNNImputer(n_neighbors=5)),
            #     ('upsampling', SMOTE()),
            #     ('downsampling', ADASYN())
            # ])
            pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ('resampling', ReSampling())
            ])

            preprocessing = ColumnTransformer([
                ('complete_pipeline', pipeline, column_list)
            ])

            return preprocessing
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(config_file_path=schema_file_path)

            target_column_name = schema[DATA_VALIDATION_GET_TARGET_COLUMN_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe")
            input_feature_train_df=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df, np.array(target_feature_test_df)]

            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")


            transformed_train_file_path = os.path.join(self.data_transformation_config.transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(self.data_transformation_config.transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path, array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data Transformation successful.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path
                                                                      )
            logging.info(f"Data Transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise HeartRiskException(e, sys)

    def __del__(self):
        logging.info(f"{'='*20}Data Transformation log completed.{'='*20} \n\n")