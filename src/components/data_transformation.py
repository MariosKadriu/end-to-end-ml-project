import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.data_path = "artifacts/data.csv"
        self.target_column_name = "math_score"

    def get_numerical_categorical_columns(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[self.target_column_name], axis=1)

        self.numerical_columns = X.select_dtypes(exclude="object").columns
        self.categorical_columns = X.select_dtypes(include="object").columns

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''

        try:
            self.get_numerical_categorical_columns()

            logging.info(f"Numerical columns: {self.numerical_columns}")
            logging.info(f"Categorical columns: {self.categorical_columns}")

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, self.numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, self.categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object completed")

            input_features_train_df = train_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_features_test_df = test_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]
            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info(f"Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)