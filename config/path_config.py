import os

########################## DATA INGESTION CONFIGURATION ##########################

RAW_DIR = "artifacts/raw"
CONFIG_PATH = "config/config.yaml"


########################### DATA PROCESSING CONFIGURATION ##########################
PROCESSED_DIR = "artifacts/processed"
ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv")
ANIME_SYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv")

X_TRAIN_ARRAY_PATH = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY_PATH = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.pkl")

RATING_DF = os.path.join(PROCESSED_DIR, "rating_df.csv")

ANIME_DF_PATH = os.path.join(PROCESSED_DIR, "anime_df.csv")
SYNOPSIS_DF_PATH = os.path.join(PROCESSED_DIR, "synopsis_df.csv")


USER2USER_ENCODED = "artifacts/processed/user2user_encoded.pkl"
USER2USER_DECODED = "artifacts/processed/user2user_decoded.pkl"

ANIME2ANIME_ENCODED = "artifacts/processed/anime2anime_encoded.pkl"
ANIME2ANIME_DECODED = "artifacts/processed/anime2anime_decoded.pkl"

