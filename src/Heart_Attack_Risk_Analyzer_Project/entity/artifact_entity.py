from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact", 
                                   ["train_file_path", "test_file_path", "is_ingested", "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
                                    ["schema_file_path", "report_file_path", "report_file_page_path",
                                     "tests_file_path", "tests_file_page_path", "cat_features_list",
                                     "num_features_list"])

DataTransformationArtifact = namedtuple("DataTransformationArtifact",
                                        ["is_transformed", "message", "transformed_train_file_path", "transformed_test_file_path",
                                         "preprocessed_object_file_path"])