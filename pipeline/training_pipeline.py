from utils.common_functions import read_yaml_file
from config.path_config import *
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining


if __name__ == "__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run_pipeline()

    model_trainer = ModelTraining(data_path=PROCESSED_DIR)
    model_trainer.train_model()