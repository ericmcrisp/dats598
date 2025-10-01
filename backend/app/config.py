import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# build relevant paths off base 
DATA_DIR = os.path.join(BASE_DIR, "data")
FEVER_DATA_PATH = os.path.join(DATA_DIR, "fever_subset.json")

# set default to dev but read the variable in 
env_mode = os.getenv("ENV", "development")

# get the env specific info 
dotenv_file = f".env.{env_mode}"
dotenv_path = os.path.join(BASE_DIR, dotenv_file)
load_dotenv(dotenv_path=dotenv_path)
ENV = env_mode
DEBUG = ENV == "development"

# get API keys 
CLAIMBUSTER_API_KEY = os.getenv("CLAIMBUSTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WIKIDATA_API_KEY = os.getenv("WIKIDATA_API_KEY")

# NLP and model config info
SPACY_MODEL = "en_core_web_sm"
