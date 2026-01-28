from utils.helpers import *
from config.path_config import *



find_similar_users(2264,USER_WEIGHTS_PATH,USER2USER_ENCODED,USER2USER_ENCODED)
print(find_similar_users)
get_user_preferences = get_user_preferences(11880 , RATING_DF, ANIME_DF_PATH , plot=True)
print(get_user_preferences)