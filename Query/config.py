# Base directories
NEWS_BASE_DIR = "/home/mraway/Desktop/src/NUS_ISS/Clustered_Articles/"
ACAD_BASE_DIR = "/home/mraway/Desktop/src/NUS_ISS/PaperMatch/utils/Clusters"

# Defaults – will be overridden in llm.py
TOPIC = "Misinformation_and_Fake_News"
ACADEMIC_CLUSTER = "cluster_-1"
NEWS_CLUSTER = "C6"

# Paths – will be rebuilt in llm.py
ACADEMIC_PARENT_DIR = None
NEWS_PARENT_DIR = None
CSV_PATH = None

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# Input placeholders
query = ""
academic_papers = []
news_articles = []