import os
import pdfplumber
import numpy as np
import pandas as pd
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import json
import logging
import easyocr
import spacy
import string
import torch

logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
print(f"Using device: {'GPU' if device != -1 else 'CPU'}")

reader = easyocr.Reader(['en'], gpu=True)
nlp = spacy.load("en_core_web_trf")

# Paths
base_path = "./papers_categories"
save_path = "./topic_model_results"
os.makedirs(save_path, exist_ok=True)

# Configure topic model
vectorizer_model = CountVectorizer(stop_words="english")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", verbose=True)

topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    language="english",
    min_topic_size=2,
    calculate_probabilities=True,
    verbose=True,
)

# Function to extract text with OCR for image-based PDFs
def extract_text_with_easyocr(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    print(f"⚠️ OCR-ing page {page.page_number} of {pdf_path}")
                    im = page.to_image(resolution=300).original.convert("RGB")
                    im_array = np.array(im)
                    result = reader.readtext(im_array, detail=0)
                    text += "\n".join(result) + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return text

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

# Process folders
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
print(f"📂 Processing Folders: {len(folders)} total")

for folder in tqdm(folders, desc="📂 Processing Folders"):
    folder_path = os.path.join(base_path, folder)
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    # Skip folders with fewer than 3 PDFs
    if len(pdf_files) < 3:
        print(f"🚫 Skipping folder {folder} because it contains fewer than 3 PDFs.")
        continue

    print(f"\n📁 Now working on: {folder}")

    docs, filenames = [], []
    for pdf_file in tqdm(pdf_files, desc=f"📄 Extracting Text ({folder})", leave=False):
        try:
            pdf_path = os.path.join(folder_path, pdf_file)
            text = extract_text_with_easyocr(pdf_path)
            
            if text.strip():  # Ensure document has content
                cleaned_text = clean_text(text)  # Clean and lemmatize text
                docs.append(cleaned_text)
                filenames.append(pdf_file)
        except Exception as e:
            print(f"⚠️ Error reading {pdf_file}: {e}")
            continue

    if not docs:
        print(f"🚫 No readable documents in folder: {folder}")
        continue

    # Check the types and ensure they are strings
    print(f"🔍 Checking document types and length:")
    for i, doc in enumerate(docs[:3]):  # Check the first 3 documents
        print(f"Document {i}: Type - {type(doc)}, Length - {len(doc)}")

    # Ensure that all documents are strings and not lists or any other structure
    docs = [str(doc) for doc in docs]
    print("\n🔍 Fitting topic model...")
    
    try:
        # Fit the model
        topics, _ = topic_model.fit_transform(docs)
    except Exception as e:
        print(f"⚠️ Error fitting the topic model: {e}")
        continue

    # Create DataFrame for summary
    df = pd.DataFrame({
        "Filename": filenames,
        "Topic": topics
    })

    summary = df.groupby("Topic").agg({
        "Filename": list,
        "Topic": "count"
    }).rename(columns={"Topic": "Count"}).reset_index()

    print("\n📊 Topics Summary:")
    print(summary)

    # Create dictionary to hold topic and papers info
    topic_data = {}

    for topic_id in summary["Topic"]:
        print(f"\n🧵 Topic {topic_id}:")
        top_words = topic_model.get_topic(topic_id)
        topic_papers = df[df["Topic"] == topic_id]["Filename"].tolist()

        # Prepare data for this topic
        topic_data[topic_id] = {
            "Top_Words": top_words,
            "Papers": topic_papers
        }

    # Save topic data to JSON file
    json_file_path = os.path.join(save_path, folder, "topic_papers.json")
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    with open(json_file_path, "w") as json_file:
        json.dump(topic_data, json_file, indent=4)

    print(f"✅ Topic and papers data saved to: {json_file_path}")

    # Save the model
    folder_save_path = os.path.join(save_path, folder)
    os.makedirs(folder_save_path, exist_ok=True)
    topic_model.save(os.path.join(folder_save_path, "topic_model.pkl"))

print("\n🏁 Done.")
