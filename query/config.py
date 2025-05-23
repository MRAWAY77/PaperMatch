# Base directories
NEWS_BASE_DIR = "/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/datasets/News_Clusters/"
ACAD_BASE_DIR = "/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/datasets/Academic_Clusters/"

BASE_DIR = "/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch"
ACADEMIC_EMBEDDING_PATH = f"{BASE_DIR}/graph_network/cluster_embeddings/academics/{{topic}}_cluster_embeddings.pt"
NEWS_EMBEDDING_PATH = f"{BASE_DIR}/graph_network/cluster_embeddings/news/{{topic}}_cluster_embeddings.pt"

TOPIC_THRESHOLD = 0.5
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

IMAGE_A = '/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/assets/MDDI_blend.jpg'

api_id = 24612008
api_hash = '563023b41e675ac0934415912c0f2fe7'
phone_number = '+6593873799'  
TARGET_CHANNEL = 'https://t.me/+n4ryVexqsAdhOGU9'