import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("src/notebook/data/stud.csv")
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(
                self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,
                      header=True, index=False)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, header=True, index=False)

            test_set.to_csv(self.ingestion_config.test_data_path,
                            header=True, index=False)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data, test_data)

    model_trainer = ModelTrainer()
    print(
        f"Best model: {model_trainer.initiate_model_trainer(train_array, test_array)}")
