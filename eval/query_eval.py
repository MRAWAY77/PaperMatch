from typing import List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
from datetime import datetime
import pandas as pd

# Load CSV containing queries
input_csv = '/home/mraway/Downloads/evaluation.csv'
output_csv = '/home/mraway/Downloads/output.csv'

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

def process_query(query: str, results_list: list, query_idx: int):
    output_data = {
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
    print("\n--- Shaped Query Template ---")
    print(shaped_query)

    classified_topic = keyword_classifier(query)
    output_data["classified_topic"] = classified_topic
    print("\n--- Classified Topic ---")
    print(classified_topic)

    if classified_topic == "Unknown":
        print("Sorry, your query does not match any known topic.")
        output_data["error"] = "Unknown topic"

        results_list.append({
            "Query": query,
            "Top_Academic_1": "",
            "Top_Academic_2": "",
            "Top_Academic_3": "",
            "Top_Academic_Cluster_1": "",
            "Top_Academic_Cluster_2": "",
            "Top_Academic_Cluster_3": "",
            "Top_News_1": "",
            "Top_News_2": "",
            "Top_News_3": "",
            "Top_News_Cluster_1": "",
            "Top_News_Cluster_2": "",
            "Top_News_Cluster_3": "",
            "Topic": "Unknown"
        })

    else:
        academic_path = f"/home/mraway/Desktop/src/NUS_ISS/PaperMatch/Graph_Network/cluster_embeddings/academics/{classified_topic}_cluster_embeddings.pt"
        news_path = f"/home/mraway/Desktop/src/NUS_ISS/PaperMatch/Graph_Network/cluster_embeddings/news/{classified_topic}_cluster_embeddings.pt"
        output_data["academic_embeddings_path"] = academic_path
        output_data["news_embeddings_path"] = news_path

        print(f"\nLoading academic embeddings from: {academic_path}")
        academic_embeddings, academic_names, academic_cluster = load_cluster_embeddings(academic_path)

        print(f"Loading news embeddings from: {news_path}")
        news_embeddings, news_names, news_cluster = load_cluster_embeddings(news_path)

        query_embedding = embedding_model.encode(query, convert_to_tensor=True)

        top_papers = find_top_matches(query_embedding, academic_embeddings, academic_names, academic_cluster)
        top_articles = find_top_matches(query_embedding, news_embeddings, news_names, news_cluster)

        academic_titles = []
        academic_clusters = []
        print("\n--- Top 3 Academic Papers ---")
        for paper, cluster in top_papers:
            print(f"Paper: {paper}, Cluster: {cluster}")
            output_data["top_academic_papers"].append({"paper": paper, "cluster": cluster})
            academic_titles.append(paper)
            academic_clusters.append(cluster)

        news_titles = []
        news_clusters = []
        print("\n--- Top 3 News Articles ---")
        for article, cluster in top_articles:
            print(f"Article: {article}, Cluster: {cluster}")
            output_data["top_news_articles"].append({"article": article, "cluster": cluster})
            news_titles.append(article)
            news_clusters.append(cluster)

        results_list.append({
            "Query": query,
            "Top_Academic_1": academic_titles[0] if len(academic_titles) > 0 else "",
            "Top_Academic_2": academic_titles[1] if len(academic_titles) > 1 else "",
            "Top_Academic_3": academic_titles[2] if len(academic_titles) > 2 else "",
            "Top_Academic_Cluster_1": academic_clusters[0] if len(academic_clusters) > 0 else "",
            "Top_Academic_Cluster_2": academic_clusters[1] if len(academic_clusters) > 1 else "",
            "Top_Academic_Cluster_3": academic_clusters[2] if len(academic_clusters) > 2 else "",
            "Top_News_1": news_titles[0] if len(news_titles) > 0 else "",
            "Top_News_2": news_titles[1] if len(news_titles) > 1 else "",
            "Top_News_3": news_titles[2] if len(news_titles) > 2 else "",
            "Top_News_Cluster_1": news_clusters[0] if len(news_clusters) > 0 else "",
            "Top_News_Cluster_2": news_clusters[1] if len(news_clusters) > 1 else "",
            "Top_News_Cluster_3": news_clusters[2] if len(news_clusters) > 2 else "",
            "Topic": classified_topic
        })

    # Save JSON log as before
    os.makedirs("llama3.3_eval_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llama3.3_eval_logs/query{query_idx}_{classified_topic or 'unknown'}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    df = pd.read_csv(input_csv)
    print(df.columns.tolist())
    queries = df['Query']

    results_list = []
    
    for idx, question in enumerate(queries):
        process_query(question,results_list,idx)
    
    # Save all collected results to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
        
    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")