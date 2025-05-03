import os
import glob
import networkx as nx
import matplotlib.pyplot as plt
from pdfplumber import open as pdf_open
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from tqdm import tqdm
import logging
import easyocr
import spacy 
import pdfplumber
import string
from transformers import AutoTokenizer
logging.getLogger("pdfminer").setLevel(logging.ERROR)

reader = easyocr.Reader(['en'], gpu=True)
nlp = spacy.load("en_core_web_trf")

# Load the pre-trained SentenceTransformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
# Make sure you load tokenizer for your summarization model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn") 
# Load summarizer pipeline from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def read_pdf(file_path):
    """Reads and extracts text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    
    cleaned_text = clean_text(text)
    return cleaned_text

def read_txt(file_path):
    """Reads and cleans text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    cleaned = clean_text(raw_text)
    return cleaned

# Function to clean and lemmatize the text
def clean_text(text):
    # Process the text with spacy NLP pipeline
    doc = nlp(text)
    
    # Filter out non-English words, stop words, punctuation, and lemmatize
    cleaned_text = " ".join([
        token.lemma_ for token in doc 
        if token.is_alpha and not token.is_stop and token.text not in string.punctuation
    ])
    
    return cleaned_text

def split_text_safely(text, tokenizer, max_tokens=512):
    """Splits text safely based on tokenizer tokens."""
    words = text.split()
    chunks = []
    current_chunk = []

    current_length = 0
    for word in words:
        word_len = len(tokenizer.tokenize(word))
        if current_length + word_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, chunk_size=512):
    """Safely summarizes text without exceeding GPU/memory limits and auto-adjusts max_length."""
    if not text.strip():
        return ""

    chunks = split_text_safely(text, tokenizer, max_tokens=chunk_size)
    summaries = []

    for chunk in chunks:
        if not chunk.strip():
            continue

        # Calculate number of input tokens
        input_tokens = len(tokenizer.tokenize(chunk))

        # Set max and min summary lengths dynamically
        dynamic_max_length = max(30, int(0.5 * input_tokens))  # 50% of input tokens
        dynamic_min_length = max(10, int(0.2 * input_tokens))  # 20% of input tokens

        try:
            output = summarizer(
                chunk,
                max_length=dynamic_max_length,
                min_length=dynamic_min_length,
                do_sample=False
            )
            summaries.append(output[0]['summary_text'])
        except Exception as e:
            print(f"Warning: Summarization failed for a chunk. Error: {e}")
            continue

    return " ".join(summaries)

def generate_embeddings_from_folder(folder_path):
    """Generate embeddings from PDFs and TXT files in the given folder."""
    cluster_embeddings = {}

    # Go through all subdirectories (clusters) in the given folder
    for cluster_name in tqdm(os.listdir(folder_path), desc="Processing Clusters"):
        cluster_dir = os.path.join(folder_path, cluster_name)
        if os.path.isdir(cluster_dir):
            texts_to_embed = []
            embeddings = []

            # Process each file in the subdirectory (cluster)
            for file_path in tqdm(glob.glob(os.path.join(cluster_dir, '*')), desc=f"Processing Files in {cluster_name}"):
                # Check if the file is a PDF or TXT
                if file_path.endswith('.pdf'):
                    text = read_pdf(file_path)
                elif file_path.endswith('.txt'):
                    text = read_txt(file_path)
                else:
                    continue  # Skip non-pdf and non-txt files
                
                if text:
                    # Summarize the text and append it to the list
                    summary = summarize_text(text)
                    texts_to_embed.append(summary)

            # Generate embeddings for all summaries in batch
            if texts_to_embed:
                embeddings = embedding_model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=True)
                cluster_embeddings[cluster_name] = embeddings

    return cluster_embeddings

def calculate_top_similarities(dict1, dict2):
    """Compare every embedding from dict1 to every embedding from dict2, and save top match per key2."""
    similarity_dict = {}

    for key1, embeds1 in tqdm(dict1.items(), desc="Comparing Clusters Dict1 -> Dict2"):
        for key2, embeds2 in tqdm(dict2.items(), desc=f"Comparing with {key1}"):
            top_similarity = -1  # initialize with worst possible
            for emb1 in embeds1:
                for emb2 in embeds2:
                    sim = util.cos_sim(emb1.clone().detach(), emb2.clone().detach()).item()
                    if sim > top_similarity:
                        top_similarity = sim
            similarity_dict[f"{key1} -> {key2}"] = top_similarity
    print(similarity_dict)
    return similarity_dict

def plot_similarity_graph(similarity_dict, threshold=0.01, save_path='Cybercrime_and_Digital_Fraud.png'):
    """Plot similarity graph with edges above threshold."""
    G = nx.Graph()

    for pair, similarity in similarity_dict.items():
        if similarity >= threshold:
            key1, key2 = pair.split(' -> ')
            G.add_node(key1, color='green')
            G.add_node(key2, color='red')
            G.add_edge(key1, key2, weight=similarity)

    pos = nx.spring_layout(G, seed=42)

    colors = [G.nodes[node]['color'] for node in G.nodes]

    plt.figure(figsize=(15, 10))
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray', node_size=700, font_size=8)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Semantic Similarity Between Clusters")
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()

folder1_path = '/home/mraway/Desktop/src/NUS_ISS/PaperMatch/utils/Clusters/Cybercrime_and_Digital_Fraud'
folder2_path = '/home/mraway/Downloads/Clustered_Articles/01 - Cybercrime & Digital Fraud'

# Part 1 & 2
dict1 = generate_embeddings_from_folder(folder1_path)
dict2 = generate_embeddings_from_folder(folder2_path)

# Part 3
similarity_dict = calculate_top_similarities(dict1, dict2)

# Part 4
plot_similarity_graph(similarity_dict, threshold=0.01)