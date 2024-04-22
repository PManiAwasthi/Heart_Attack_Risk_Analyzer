from src.Heart_Attack_Risk_Analyzer_Project.logger import logging
from src.Heart_Attack_Risk_Analyzer_Project.exception import HeartRiskException
from src.Heart_Attack_Risk_Analyzer_Project.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.Heart_Attack_Risk_Analyzer_Project.config.config import DataTransformationConfig
import os,sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.Heart_Attack_Risk_Analyzer_Project.utils.utils import load_data, read_yaml_file, save_numpy_array_data, save_object
from src.Heart_Attack_Risk_Analyzer_Project.constant import DATA_VALIDATION_GET_NUMERICAL_COLUMN_KEY, DATA_VALIDATION_GET_CATEGORICAL_COLUMN_KEY, DATA_VALIDATION_GET_TARGET_COLUMN_KEY


class ReSampling(BaseEstimator, TransformerMixin):
    def __init__(self, TenYearCHD=15):
        try:
            self.TenYearCHD = TenYearCHD
        except Exception as e:
            raise HeartRiskException(e, sys)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        try:
            smote_obj = SMOTE()
            adasyn_obj = ADASYN()

            y = X[:, self.TenYearCHD]
            X = np.delete(X, self.TenYearCHD, axis=1)

            X, y = smote_obj.fit_resample(X, y)
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
            column_list.append(dataset_schema[DATA_VALIDATION_GET_TARGET_COLUMN_KEY])
            # pipeline = Pipeline(steps=[
            #     ('imputer', KNNImputer(n_neighbors=5)),
            #     ('upsampling', SMOTE()),
            #     ('downsampling', ADASYN())
            # ])

            num_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder', OneHotEncoder())
            ])

            preprocessing = ColumnTransformer([
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ('num_pipeline', num_pipeline, numerical_columns)
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

            logging.info(f"Applying preprocessing object on traing dataframe and testing dataframe")
            input_feature_train_df=preprocessing_obj.fit_transform(train_df)
            input_feature_test_df=preprocessing_obj.fit_transform(test_df)

            logging.info(f"After preprocessing applying resampling of the data to avoid class imbalance.")
            re_sampling_obj = ReSampling()
            input_feature_train_df = re_sampling_obj.fit_transform(input_feature_train_df)
            input_feature_test_df = re_sampling_obj.fit_transform(input_feature_test_df)

            logging.info(f"Data set up and down sampling completed successfully.")

            train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")


            transformed_train_file_path = os.path.join(self.data_transformation_config.transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(self.data_transformation_config.transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path=transformed_train_file_path, array=input_feature_train_df)
            save_numpy_array_data(file_path=transformed_test_file_path, array=input_feature_test_df)

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