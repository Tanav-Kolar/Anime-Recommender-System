import os
import sys
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml_file

logger = get_logger(__name__)


class DataIngestion: 
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_names = self.config["bucket_file_names"]

        os.makedirs(RAW_DIR, exist_ok=True)
        
        logger.info(f"DataIngestion initialized with bucket: {self.bucket_name} and files: {self.bucket_file_names}")

    def download_data_from_gcs(self):
        """Downloads data files from Google Cloud Storage to local raw data directory."""
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                destination_file_path = os.path.join(RAW_DIR, file_name)
                
                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(destination_file_path)

                    data = pd.read_csv(destination_file_path, nrows=10_000_000)
                    data.to_csv(destination_file_path, index=False)
                    logger.info(f"Downloaded and truncated {file_name} to {destination_file_path}, as large file detected.")
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(destination_file_path)
                    logger.info(f"Downloaded {file_name} to {destination_file_path}")


        except Exception as e:
            logger.error(f"Error downloading data from GCS: {e}")
            raise CustomException(e, sys)
        
    def run(self):
        """Executes the data ingestion process."""
        try:
            self.download_data_from_gcs()
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Custom Exception: {str(e)}")
            raise CustomException(e, sys)
        finally:
            logger.info("Data ingestion process finished.")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml_file(CONFIG_PATH))
    data_ingestion.run() 


