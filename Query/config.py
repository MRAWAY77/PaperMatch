# Base directories
NEWS_BASE_DIR = "/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/datasets/News_Clusters/"
ACAD_BASE_DIR = "/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/datasets/Academic_Clusters/"

# Defaults None will be overridden in llm.py
TOPIC = None
ACADEMIC_CLUSTER = None
NEWS_CLUSTER = None

# Paths None will be rebuilt in llm.py
ACADEMIC_PARENT_DIR = None
NEWS_PARENT_DIR = None
CSV_PATH = None

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# Input placeholders
query = ""
academic_papers = []
news_articles = []