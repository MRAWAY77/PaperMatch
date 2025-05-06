from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
from datetime import datetime

# Initialize your embedding model (must match the one used for clusters!)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim

# Topics
TOPICS: List[str] = [
    "Cybercrime_and_Digital_Fraud",
    "Forensic_Science_and_Criminal_Investigation",
    "Misinformation_and_Fake_News",
    "Organised_Crime_and_Drug_Trafficking",
    "Medical_Fraud_and_Malpractice"
]

# Templates
def shape_query_with_template(query: str) -> str:
    template = (
        "You are categorizing a query to find the most suitable research topic.\n"
        "Query: \"{query}\"\n"
        "Provide a cleaned, neutral version suitable for topic classification."
    )
    return template.format(query=query)

def keyword_classifier(cleaned_query: str) -> str:
    keyword_classifier = {
    "Medical_Fraud_and_Malpractice": [
        "medical fraud", "healthcare fraud", "insurance fraud", "billing fraud", "medicare fraud",
        "medicaid fraud", "upcoding", "phantom billing", "patient brokering", "kickback scheme",
        "medical identity theft", "prescription fraud", "fraudulent diagnosis", "unnecessary procedure",
        "health insurance fraud", "pharmaceutical fraud",
        "malpractice", "unethical treatment", "experimental treatment", "informed consent violation",
        "patient exploitation", "medical negligence", "falsified credentials", "medical license fraud",
        "counterfeit medicine", "unlicensed practice", "medical data manipulation", "clinical trial fraud"
    ],
    "Misinformation_and_Fake_News": [
        "fake news", "disinformation", "misinformation", "propaganda", "conspiracy theory",
        "information warfare", "social media manipulation", "deepfake", "fact-checking", "media literacy",
        "information disorder", "filter bubble", "echo chamber", "algorithmic bias", "synthetic media",
        "information operations", "coordinated inauthentic behavior", "influence operation", "astroturfing",
        "computational propaganda", "source verification", "media bias", "journalistic integrity",
        "information source", "primary source", "secondary source", "citation needed", "unverified claim",
        "anonymous source", "information provenance", "source attribution", "fact versus opinion",
        "source criticism"
    ],
    "Organised_Crime_and_Drug_Trafficking": [
        "organized crime", "organised crime", "criminal syndicate", "mafia", "crime network",
        "criminal organization", "criminal organisation", "mob", "racketeering", "criminal enterprise",
        "illegal operation", "crime family", "criminal group", "underworld", "yakuza", "triads",
        "criminal clan", "criminal conspiracy", "crime syndicate",
        "drug trafficking", "drug trade", "narcotics trade", "illegal drug", "drug smuggling",
        "cocaine trade", "heroin distribution", "methamphetamine", "drug cartel", "drug syndicate",
        "drug network", "illicit drug", "drug smuggler", "controlled substance", "drug ring",
        "international drug trade", "drug bust", "narcotics trafficking", "illegal drug trade",
        "narcotic distribution", "heroin trafficking", "methamphetamine trade", "drug distribution",
        "illegal drug network", "drug interdiction", "drug supply chain"
    ],
    "Cybercrime_and_Digital_Fraud": [
        "cyber attack", "malware", "ransomware", "phishing", "data breach", "hacking", "cybercrime",
        "cyber security", "cyber criminal", "dark web", "cyber fraud", "identity theft", "cyber espionage",
        "botnet", "DDoS attack", "cyber warfare", "computer virus", "data theft", "online fraud",
        "cryptocurrency crime",
        "zero-day exploit", "social engineering", "encryption", "keylogger", "backdoor", "brute force attack",
        "SQL injection", "man-in-the-middle", "password cracking", "spyware", "trojan horse", "rootkit",
        "cryptojacking", "extortion"
    ],
    "Forensic_Science_and_Criminal_Investigation": [
        "DNA analysis", "fingerprint analysis", "ballistics", "toxicology", "forensic pathology",
        "crime scene investigation", "digital forensics", "blood pattern analysis", "forensic anthropology",
        "trace evidence", "forensic entomology", "forensic psychology", "autopsy", "serology",
        "chain of custody", "forensic odontology", "chromatography", "spectroscopy", "PCR amplification",
        "mass spectrometry", "microscopy", "luminol test", "substance identification", "comparative analysis",
        "facial reconstruction", "voice analysis", "handwriting analysis", "geographic profiling"
    ]
    }

    cleaned_query_lower = cleaned_query.lower()

    # First pass: keyword match
    for topic, keywords in keyword_classifier.items():
        for keyword in keywords:
            if keyword in cleaned_query_lower:
                return topic

    # Fallback: Semantic similarity
    topic_embeddings = embedding_model.encode(TOPICS, convert_to_tensor=True)
    query_embedding = embedding_model.encode(cleaned_query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, topic_embeddings)[0]
    best_topic_idx = torch.argmax(cos_scores).item()
    return TOPICS[best_topic_idx]

def load_cluster_embeddings(path: str) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Load the cluster .pt file, returning embeddings, their corresponding paper/article names, and their cluster labels.
    """
    # Load the .pt file
    data = torch.load(path)
    
    # Prepare lists to store the embeddings, paper names, and clusters
    embeddings = []
    paper_names = []
    cluster_labels = []

    # Iterate over the clusters in the data
    for cluster_key, cluster_data in data.items():
        # Iterate over each paper and embedding pair in the cluster
        for paper_name, embedding in cluster_data:
            embeddings.append(embedding)  # Add the embedding tensor (already a tensor)
            paper_names.append(paper_name)  # Add the paper name
            cluster_labels.append(cluster_key)  # Add the cluster label

    return embeddings, paper_names, cluster_labels

def find_top_matches(query_embedding: torch.Tensor, embeddings: List[torch.Tensor], names: List[str], clusters: List[str], top_k: int = 3) -> List[Tuple[str, str]]:
    """
    Find the top_k most similar names based on cosine similarity and return their cluster.
    """
    # Ensure query_embedding has the correct shape
    query_embedding = query_embedding.unsqueeze(0)  # Add a batch dimension if it's 1D
    
    # Zip embeddings, names, and clusters together
    embeddings_zipped = zip(embeddings, names, clusters)
    
    # Compute cosine similarity between query_embedding and each embedding in embeddings
    cos_scores = []
    for emb, name, cluster in embeddings_zipped:
        # Compute similarity between query_embedding and the current embedding
        cos_sim_score = util.cos_sim(query_embedding, emb.unsqueeze(0))[0].item()  # Convert to a scalar value
        cos_scores.append((cos_sim_score, name, cluster))
    
    # Sort the cosine scores in descending order and select the top_k results
    top_results = sorted(cos_scores, key=lambda x: x[0], reverse=True)[:top_k]
    
    # Return the names and clusters of the top_k matches
    return [(name, cluster) for _, name, cluster in top_results]

def process_query(query: str, result_callback=None):
    output_data = {
        "status": "new",
        "query": query,
        "shaped_query": "",
        "classified_topic": "",
        "academic_embeddings_path": "",
        "news_embeddings_path": "",
        "top_academic_papers": [],
        "top_news_articles": []
    }

    shaped_query = shape_query_with_template(query)
    output_data["shaped_query"] = shaped_query
    result_message = "\n--- Shaped Query Template ---\n" + shaped_query

    classified_topic = keyword_classifier(query)
    output_data["classified_topic"] = classified_topic
    result_message += "\n--- Classified Topic ---\n" + classified_topic

    if classified_topic == "Unknown":
        result_message += "\nSorry, your query does not match any known topic."
        output_data["error"] = "Unknown topic"
    else:
        academic_path = f"/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/Graph_Network/cluster_embeddings/academics/{classified_topic}_cluster_embeddings.pt"
        news_path = f"/home/lenovo3/Desktop/Alvin/NUS_ISS/PaperMatch/Graph_Network/cluster_embeddings/news/{classified_topic}_cluster_embeddings.pt"
        output_data["academic_embeddings_path"] = academic_path
        output_data["news_embeddings_path"] = news_path

        # result_message += f"\n\nLoading academic embeddings from: {academic_path}"
        academic_embeddings, academic_names, academic_cluster = load_cluster_embeddings(academic_path)

        # result_message += f"\nLoading news embeddings from: {news_path}"
        news_embeddings, news_names, news_cluster = load_cluster_embeddings(news_path)

        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        top_papers = find_top_matches(query_embedding, academic_embeddings, academic_names, academic_cluster)
        top_articles = find_top_matches(query_embedding, news_embeddings, news_names, news_cluster)

        result_message += "\n--- Top 3 Academic Papers ---"
        for paper, cluster in top_papers:
            result_message += f"\nPaper: {paper}, Cluster: {cluster}"
            output_data["top_academic_papers"].append({"paper": paper, "cluster": cluster})

        result_message += "\n--- Top 3 News Articles ---"
        for article, cluster in top_articles:
            result_message += f"\nArticle: {article}, Cluster: {cluster}"
            output_data["top_news_articles"].append({"article": article, "cluster": cluster})

    os.makedirs("eval_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_logs/query_result_{classified_topic or 'unknown'}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    result_message += f"\n\nQuery results saved to: {filename}"

    print(result_message)

    if result_callback:
        try:
            result_callback(result_message)
        except Exception as e:
            print(f"Error in result_callback: {e}")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    user_input = input("Enter your query: ")
    process_query(user_input)
    
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")