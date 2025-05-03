import os
import glob
from collections import Counter
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, util
import easyocr
import spacy 
import pdfplumber
import string
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

reader = easyocr.Reader(['en'], gpu=True)
nlp = spacy.load("en_core_web_trf")
# Load the pre-trained SentenceTransformer model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

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

def generate_embeddings_from_folder(folder_path, save_path):
    """Generate embeddings from PDFs and TXT files in the given folder and save them in .pt format."""
    cluster_embeddings = {}

    # Process clusters with tqdm to show progress
    for cluster_name in tqdm(os.listdir(folder_path), desc="Processing Clusters"):
        cluster_dir = os.path.join(folder_path, cluster_name)
        if os.path.isdir(cluster_dir):
            texts_to_embed = []
            paper_names = []

            # Process files in each cluster folder
            files_in_cluster = glob.glob(os.path.join(cluster_dir, '*'))
            for file_path in tqdm(files_in_cluster, desc=f"Processing Files in {cluster_name}", leave=False):
                if file_path.endswith('.pdf'):
                    cleaned_text = read_pdf(file_path)
                elif file_path.endswith('.txt'):
                    cleaned_text = read_txt(file_path)
                else:
                    continue  # Skip unsupported files

                if cleaned_text:
                    word_counts = Counter(cleaned_text.split())
                    top_words = word_counts.most_common(100)

                    # Form text string from top 100 words (only)
                    top_words_text = " ".join([word for word, freq in top_words])
                    
                    texts_to_embed.append(top_words_text)
                    paper_names.append(os.path.basename(file_path))

            # Generate embeddings for all papers in batch
            if texts_to_embed:
                embeddings = embedding_model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=True)

                # Store embeddings and paper names in the cluster_embeddings dictionary
                cluster_embeddings[cluster_name] = list(zip(paper_names, embeddings))

    # Save the embeddings to a .pt file
    torch.save(cluster_embeddings, save_path)
    print(f"Cluster embeddings saved to {save_path}")

    return cluster_embeddings

# Example usage
folder_path = '/home/mraway/Downloads/Clustered_Articles/05 - Organised Crime & Drug Trafficking'
name = folder_path.split("/")[-1]
save_dir = '/home/mraway/Desktop/src/NUS_ISS/PaperMatch/Graph_Network/cluster_embeddings/news'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
save_path = os.path.join(save_dir, f'{name}_cluster_embeddings.pt')  # Full path to save .pt file
generate_embeddings_from_folder(folder_path, save_path)