from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Activation, BatchNormalization
from utils.common_functions import read_yaml_file
from src.logger import get_logger
from src.custom_exception import CustomException


logger = get_logger(__name__)


#creating class
class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml_file(config_path)
            logger.info("Configuration file loaded successfully.")
        except Exception as e:
            logger.error("Error loading configuration file.")
            raise CustomException(e)
   
    def RecommenderNet(self, n_users, n_animes):
        try: 
            embedding_size = self.config['model']['embedding_size']

            user = Input(shape=[1], name='user')
            anime = Input(shape=[1], name='anime')

            user_embedding = Embedding(input_dim=n_users, output_dim=embedding_size, name='user_embedding')(user)
            anime_embedding = Embedding(input_dim=n_animes, output_dim=embedding_size, name='anime_embedding')(anime)

            x = Dot(name = 'dot_product', normalize=True, axes=2)([user_embedding, anime_embedding])
            x = Flatten()(x)

            x = Dense(1, kernel_initializer='he_normal', name='dense_1')(x)
            x = BatchNormalization(name='batch_normalization_1')(x)
            x = Activation('sigmoid', name='activation_1')(x)
            
            model = Model(inputs=[user, anime], outputs=x)
            model.compile(
                loss = self.config['model']['loss'],
                optimizer = self.config['model']['optimizer'], 
                metrics = self.config['model']['metrics']
                )
            logger.info("Model created successfully.")
            return model 
        
        except Exception as e:
            logger.error("Error creating the RecommenderNet model.")
            raise CustomException(e)
        
    