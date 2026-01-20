import joblib
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from config.path_config import *

logger = get_logger(__name__)



class ModelTraining:
    def __init__ (self, data_path):
        self.data_path = data_path
        logger.info("ModelTraining initialized")

    def load_data(self):
        try:
            X_train = joblib.load(X_TRAIN_ARRAY_PATH)
            X_test = joblib.load(X_TEST_ARRAY_PATH)
            y_train = joblib.load(Y_TRAIN_PATH)
            y_test = joblib.load(Y_TEST_PATH)
            logger.info("Data loaded successfully for training and testing.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Error loading data for model training.")
            raise CustomException(e)
    
    def train_model(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            n_users = len(joblib.load(USER2USER_ENCODED))
            n_animes = len(joblib.load(ANIME2ANIME_ENCODED))

            base_model = BaseModel(CONFIG_PATH)
            model = base_model.RecommenderNet(n_users, n_animes)

            #set callabacks and hyperparameters
            start_lr = 0.00001
            min_lr = 0.0001
            max_lr = 0.00005
            batch_size = 10000

            ramup_epochs = 5
            sustain_epochs = 0
            exponential_decay = 0.8

            def lrfn(epoch):
                if epoch < ramup_epochs:
                    lr = (max_lr - start_lr) / ramup_epochs * epoch + start_lr
                elif epoch < ramup_epochs + sustain_epochs:
                    lr = max_lr
                else:
                    lr = (max_lr - min_lr) * exponential_decay**(epoch - ramup_epochs - sustain_epochs) + min_lr
                return lr

            lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose = 0)

            model_checkpoint = ModelCheckpoint(filepath=CHECKPOINT_FILE_PATH, save_weights_only=True, monitor = "val_loss", save_best_only=True)

            early_stopping = EarlyStopping(patience = 3, monitor="val_loss", restore_best_weights=True, mode = 'min')

            my_callbacks = [lr_callback, model_checkpoint, early_stopping]

            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            try: 
                history = model.fit(
                    x = X_train,
                    y = y_train,
                    batch_size = batch_size,
                    epochs = 20,
                    verbose = 1,
                    validation_data = (X_test, y_test),
                    callbacks = my_callbacks
                )
                
                model.load_weights(CHECKPOINT_FILE_PATH)
                logger.info("Model training completed successfully.")

            except Exception as e:
                raise CustomException("Model training failed.",e)
            
            self.save_model_weights(model)
        
        except Exception as e:
            logger.error("Error during model training.")
            raise CustomException(e)
        
    def extract_weights(self, layer_name, model):
        try: 
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0] 
            weights = weights/np.linalg.norm(weights, axis=1).reshape((-1,1))
            return weights
        except Exception as e:
            logger.error("Error during weight extraction.")
            raise CustomException(e)
        

    def save_model_weights(self, model):
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH} successfully.")

            user_weights = self.extract_weights("user_embedding", model)
            anime_weights = self.extract_weights("anime_embedding", model)

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            logger.info("User and Anime weights extracted and saved successfully.")

        except Exception as e:
            logger.error("Error during weight extraction.")
            raise CustomException(e)

if __name__ == "__main__":
    model_trainer = ModelTraining(data_path=PROCESSED_DIR)
    model_trainer.train_model()
