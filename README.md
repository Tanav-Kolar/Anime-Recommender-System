# Anime-Recommender-System
Recommends a selection of Anime to users.


How to run

1. Create a virtual environment. 
 uv venv 
 source .venv/bin/activate

2. Initialise project setup
uv pip install -e .

3. Since the animelist.csv file is too big, we perform selective data ingestion.


GCLOUD SETUP & Authentication


Running data_ingestion.py

1. Try " python3 src/data_ingestion.py" , if encountered error "module not found", then run "python3 -m src.data_ingestion"
