#!/usr/bin/env python
# coding: utf-8

# In[90]:


"""
News Article Clustering Program
-------------------------------
This program clusters news articles based on their textual similarity.
It processes articles with relevance scores of 3, 4, and 5 from multiple themes.

The program:
1. Allows user to select which theme to process
2. Reads masterlists from the current working directory
3. Extracts article content from the appropriate repository folder
4. Performs text preprocessing and clustering
5. Outputs a consolidated masterlist with cluster assignments
"""

# In[92]:

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK resource download failed. If needed, download them manually.")


# In[93]:


class NewsArticleClustering:
    def __init__(self, theme_id='01'):
        """
        Initialize the clustering class.
        
        Args:
            theme_id: ID of the theme to process ('01' to '05')
        """
        # Current working directory is "Project Delphi"
        self.CWD = os.getcwd()
        self.theme_id = theme_id
        
        # Define constants
        self.PUBLISHERS = ['BBC', 'CNA', 'CNN', 'ST']
        self.RELEVANCE_THRESHOLD = 3
        self.THEMES = {
            '01': 'Cybercrime & Digital Fraud',
            '02': 'Forensic Science & Criminal Investigations',
            '03': 'Medical Fraud & Malpractice',
            '04': 'Media Misinformation & Fake News',
            '05': 'Organised Crime & Drug Trafficking'
        }
        
        # Configure paths based on selected theme
        self._configure_theme_paths(theme_id)
        
        # Text processing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize data containers
        self.masterlist_data = None
        self.data = None
        self.vectorizer = None
        self.cluster_model = None
        self.optimal_clusters = None
        
    def _configure_theme_paths(self, theme_id):
        """Configure file paths based on the selected theme."""
        # Validate theme ID
        if theme_id not in self.THEMES:
            raise ValueError(f"Invalid theme ID: {theme_id}. Must be one of: {', '.join(self.THEMES.keys())}")
        
        # Get theme name
        theme_name = self.THEMES[theme_id]
        
        # Set path to articles repository
        self.ARTICLES_REPO_PATH = os.path.join(f"{theme_id} - {theme_name}", "02 - Articles")
        
        # Set masterlist file paths based on theme
        self.MASTERLIST_FILES = {}
        
        # Define masterlist file patterns for each theme
        if theme_id == '01':  # Cybercrime & Digital Fraud
            self.MASTERLIST_FILES = {
                'BBC': "01 - BBC_Cybercrime_Articles_ProcessedUpdated.xlsx",
                'CNN': "01 - CNN_Cybercrime_Articles_ProcessedUpdated.xlsx",
                'CNA': "01 - CNA_Cybercrime_Articles_ProcessedUpdated.xlsx",
                'ST': "01 - ST_Cybersecurity_Articles_ProcessedUpdated.xlsx"
            }
        elif theme_id == '02':  # Forensic Science
            self.MASTERLIST_FILES = {
                'BBC': "02 - BBC_Forensic_Science_Articles_ProcessedUpdated.xlsx",
                'CNN': "02 - CNN_Forensic_Science_Articles_ProcessedUpdated.xlsx",
                'CNA': "02 - CNA_Forensic_Science_Articles_ProcessedUpdated.xlsx",
                'ST': "02 - ST_Forensic_Science_Articles_ProcessedUpdated.xlsx"
            }
        elif theme_id == '03':  # Medical Fraud
            self.MASTERLIST_FILES = {
                'BBC': "03 - BBC_Medical_Fraud_Articles_ProcessedUpdated.xlsx",
                'CNN': "03 - CNN_Medical_Fraud_Articles_ProcessedUpdated.xlsx",
                'CNA': "03 - CNA_Medical_Fraud_Articles_ProcessedUpdated.xlsx",
                'ST': "03 - ST_Medical_Fraud_Articles_ProcessedUpdated.xlsx"
            }
        elif theme_id == '04':  # Media Misinformation
            self.MASTERLIST_FILES = {
                'BBC': "04 - BBC_Misinformation_Articles_ProcessedUpdated.xlsx",
                'CNN': "04 - CNN_Misinformation_Articles_ProcessedUpdated.xlsx",
                'CNA': "04 - CNA_Misinformation_Articles_ProcessedUpdated.xlsx",
                'ST': "04 - ST_Misinformation_Articles_ProcessedUpdated.xlsx"
            }
        elif theme_id == '05':  # Organised Crime
            self.MASTERLIST_FILES = {
                'BBC': "05 - BBC_Organised_Crime_Articles_ProcessedUpdated.xlsx",
                'CNN': "05 - CNN_Organised_Crime_Articles_ProcessedUpdated.xlsx",
                'CNA': "05 - CNA_Organised_Crime_Articles_ProcessedUpdated.xlsx",
                'ST': "05 - ST_Organised_Crime_Articles_ProcessedUpdated.xlsx"
            }
    
    def load_masterlists(self):
        """
        Load and combine relevant articles from all publisher masterlists.
        
        Returns:
            pandas.DataFrame: Combined dataframe of relevant articles
        """
        theme_name = self.THEMES.get(self.theme_id, 'Unknown Theme')
        print(f"Loading masterlist data for theme: {self.theme_id} - {theme_name}")
        all_dfs = []
        
        for publisher in self.PUBLISHERS:
            file_path = self.MASTERLIST_FILES.get(publisher)
            if not file_path:
                print(f"  No masterlist file defined for {publisher}, skipping.")
                continue
                
            try:
                # Load the masterlist
                print(f"  Loading masterlist: {file_path}")
                df = pd.read_excel(file_path)
                
                # Add publisher info if not present
                if 'Publisher' not in df.columns:
                    df['Publisher'] = publisher
                
                # Find relevance column
                relevance_col = None
                for col in df.columns:
                    if 'relevance' in str(col).lower():
                        relevance_col = col
                        break
                
                if relevance_col:
                    # Filter for high relevance articles (3-5)
                    df_filtered = df[df[relevance_col] >= self.RELEVANCE_THRESHOLD].copy()
                    print(f"    Found {len(df_filtered)} relevant articles for {publisher}")
                    all_dfs.append(df_filtered)
                else:
                    print(f"    Warning: No relevance column found in {file_path}")
                    
            except Exception as e:
                print(f"    Error loading {file_path}: {str(e)}")
        
        # Combine all dataframes
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Total articles in masterlists: {len(combined_df)}")
            print(f"Articles by publisher: {combined_df['Publisher'].value_counts().to_dict()}")
            
            # Find filename column
            filename_col = self._find_filename_column(combined_df)
            
            self.masterlist_data = combined_df
            self.filename_col = filename_col
            return combined_df
        else:
            print("No data loaded. Please check masterlist files.")
            return pd.DataFrame()
    
    def _find_filename_column(self, df):
        """Find the column containing filenames in the masterlist."""
        for col in df.columns:
            col_lower = str(col).lower()
            if 'file' in col_lower or 'name' in col_lower or 'path' in col_lower:
                return col
        
        # If no obvious filename column, use index as fallback
        print("Warning: No filename column identified. Using index as filename.")
        df['filename'] = df.index.astype(str)
        return 'filename'
    
    def load_article_content(self):
        """
        Load actual article content from text files in the repository.
        
        Returns:
            pandas.DataFrame: Dataframe with article content added
        """
        print(f"Loading article content from: {self.ARTICLES_REPO_PATH}")
        
        if self.masterlist_data is None:
            print("No masterlist data available. Run load_masterlists() first.")
            return pd.DataFrame()
        
        # Create copy of masterlist data
        df = self.masterlist_data.copy()
        
        # Add columns for article content
        df['article_title'] = ''
        df['article_content'] = ''
        
        # Track progress
        articles_found = 0
        articles_not_found = 0
        
        # Process each article
        for idx, row in df.iterrows():
            publisher = row['Publisher']
            filename = str(row[self.filename_col])
            
            # Add .txt extension if needed
            if not filename.endswith('.txt'):
                filename = f"{filename}.txt"
            
            # Construct path to article file
            article_path = os.path.join(self.ARTICLES_REPO_PATH, filename)
            
            try:
                # Read article file
                with open(article_path, 'r', encoding='utf-8', errors='replace') as file:
                    lines = file.readlines()
                
                if lines:
                    # Extract title (first line)
                    title_line = lines[0].strip()
                    if title_line.startswith('Title:'):
                        title = title_line[6:].strip()
                    else:
                        title = title_line
                    
                    # Skip rows 2-9, use content after row 10
                    content_lines = lines[10:] if len(lines) > 10 else []
                    content = ''.join(content_lines).strip()
                    
                    # Store in dataframe
                    df.at[idx, 'article_title'] = title
                    df.at[idx, 'article_content'] = content
                    
                    articles_found += 1
                    if articles_found % 20 == 0:
                        print(f"  Processed {articles_found} articles...")
                
            except Exception as e:
                articles_not_found += 1
                if articles_not_found < 10:  # Limit error messages
                    print(f"  Could not read article: {filename} - {str(e)}")
                elif articles_not_found == 10:
                    print("  Additional file errors suppressed...")
        
        print(f"Article loading complete. Found {articles_found} articles, {articles_not_found} not found.")
        
        # Remove articles with no content
        df = df[df['article_content'] != '']
        print(f"Retained {len(df)} articles with content for analysis.")
        
        self.data = df
        return df
    
    def preprocess_text(self):
        """
        Preprocess article text for analysis, combining title and content.
        
        Returns:
            pandas.DataFrame: Dataframe with added 'processed_text' column
        """
        print("Preprocessing article text...")
        
        if self.data is None or len(self.data) == 0:
            print("No article data available.")
            return pd.DataFrame()
        
        df = self.data.copy()
        
        # Define preprocessing function
        def preprocess(title, content):
            # Combine title and content (giving title more weight)
            combined_text = f"{title} {title} {content}"
            
            if not isinstance(combined_text, str):
                return ""
            
            # Convert to lowercase
            text = combined_text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters and numbers
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                     if word not in self.stop_words and len(word) > 2]
            
            # Join back into a string
            return ' '.join(tokens)
        
        # Apply preprocessing to title and content
        df['processed_text'] = df.apply(
            lambda row: preprocess(row['article_title'], row['article_content']), 
            axis=1
        )
        
        # Remove articles with empty processed text
        df = df[df['processed_text'].str.strip() != '']
        
        self.data = df
        print(f"Preprocessing complete. {len(df)} articles retained.")
        
        return df
    
    def vectorize_text(self):
        """
        Convert processed text to TF-IDF vectors.
        
        Returns:
            scipy.sparse.csr.csr_matrix: TF-IDF feature matrix
        """
        print("Vectorizing text using TF-IDF...")
        
        if 'processed_text' not in self.data.columns:
            print("Text not preprocessed. Run preprocess_text() first.")
            return None
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,       # Limit features
            min_df=2,                # Ignore terms in fewer than 2 documents
            max_df=0.85,             # Ignore terms in more than 85% of documents
            ngram_range=(1, 2)       # Include unigrams and bigrams
        )
        
        # Fit and transform text
        X = self.vectorizer.fit_transform(self.data['processed_text'])
        
        print(f"Vectorization complete. Matrix shape: {X.shape}")
        return X
    
    def find_optimal_clusters(self, X, max_clusters=20):
        """
        Find optimal number of clusters using silhouette scores.
        
        Args:
            X: TF-IDF feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            int: Optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        # Limit range based on dataset size
        min_clusters = 2
        max_clusters = min(max_clusters, len(self.data) // 5)
        
        silhouette_scores = []
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            # Initialize KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                max_iter=300,
                n_init=10,
                random_state=42
            )
            
            # Fit and evaluate
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"  {n_clusters} clusters: silhouette score = {silhouette_avg:.3f}")
        
        # Find best number of clusters
        optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        
        print(f"Optimal number of clusters: {optimal_clusters}")
        self.optimal_clusters = optimal_clusters
        
        return optimal_clusters
    
    def cluster_articles(self, X, n_clusters=None):
        """
        Perform K-means clustering on vectorized text.
        
        Args:
            X: TF-IDF feature matrix
            n_clusters: Number of clusters (if None, use optimal clusters or default to 5)
            
        Returns:
            pandas.DataFrame: Dataframe with added cluster labels
        """
        print("Clustering articles...")
        
        if n_clusters is None:
            n_clusters = self.optimal_clusters if self.optimal_clusters else 5
            
        # Initialize KMeans
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=300,
            n_init=10,
            random_state=42
        )
        
        # Fit model and assign clusters
        self.data['cluster'] = self.cluster_model.fit_predict(X)
        
        prefix = chr(ord('A') + int(self.theme_id) - 1)  # A for 01, B for 02, etc. according to theme selected
        self.data['cluster_name'] = prefix + (self.data['cluster'] + 1).astype(str)
        
        # Count articles per cluster
        cluster_counts = self.data['cluster_name'].value_counts().sort_index()
        print("Articles per cluster:")
        for cluster, count in cluster_counts.items():
            print(f"  {cluster}: {count} articles")
        
        return self.data
    
    def get_cluster_keywords(self, top_n=10):
        """
        Extract top keywords for each cluster.
        
        Args:
            top_n: Number of top keywords to extract per cluster
            
        Returns:
            dict: Dictionary with cluster names as keys and lists of keywords as values
        """
        print("Extracting top keywords for each cluster...")
        
        if self.cluster_model is None or self.vectorizer is None:
            print("Clustering model or vectorizer not initialized.")
            return {}
        
        # Get feature names and cluster centers
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        cluster_centers = self.cluster_model.cluster_centers_
        
        # Dictionary to store cluster keywords
        cluster_keywords = {}
        
        for cluster_id in range(len(cluster_centers)):
            # Get indices of top features for this cluster
            order_centroids = cluster_centers[cluster_id].argsort()[::-1]
            
            # Get corresponding feature names (keywords)
            keywords = feature_names[order_centroids[:top_n]].tolist()
            
            # Store with 'A' prefix (starting from 1, not 0)
            prefix = chr(ord('A') + int(self.theme_id) - 1)  # A for 01, B for 02, etc. According to selected theme
            cluster_name = f"{prefix}{cluster_id + 1}"
            cluster_keywords[cluster_name] = keywords
            print(f"  {cluster_name}: {', '.join(keywords)}")
        
        return cluster_keywords
    
    def create_consolidated_masterlist(self):
        """
        Create consolidated masterlist with cluster information.
        
        Returns:
            pandas.DataFrame: Consolidated masterlist
        """
        theme_name = self.THEMES.get(self.theme_id, 'Unknown')
        output_path = f"{self.theme_id}_consolidated_masterlist.xlsx"
        
        print(f"Creating consolidated masterlist for theme {self.theme_id}...")
        
        if 'cluster_name' not in self.data.columns:
            print("No cluster labels found. Run cluster_articles() first.")
            return None
        
        # Create copy of the data
        masterlist = self.data.copy()

        prefix = chr(ord('A') + int(self.theme_id) - 1)  # A for 01, B for 02, etc. According to selected theme
        
        # Get cluster keywords
        cluster_keywords = self.get_cluster_keywords(top_n=5)
        
        # Add cluster descriptions
        masterlist['cluster_keywords'] = masterlist['cluster_name'].apply(
            lambda x: ', '.join(cluster_keywords.get(x, []))
        )
        
        # Add cluster label with keywords
        masterlist['cluster_label'] = masterlist.apply(
            lambda row: f"{row['cluster_name']} ({row['cluster_keywords']})",
            axis=1
        )
        
        # Save to Excel
        masterlist.to_excel(output_path, index=False)
        print(f"Consolidated masterlist saved to '{output_path}'")
        
        return masterlist
    
    def create_cluster_summary(self):
        """
        Create summary of clusters with key information.
        
        Returns:
            pandas.DataFrame: Cluster summary dataframe
        """
        theme_name = self.THEMES.get(self.theme_id, 'Unknown')
        output_path = f"{self.theme_id}_cluster_summary.xlsx"
        
        print(f"Creating cluster summary for theme {self.theme_id}...")
        
        if 'cluster_name' not in self.data.columns:
            print("No cluster labels found. Run cluster_articles() first.")
            return None

        prefix = chr(ord('A') + int(self.theme_id) - 1)  # A for 01, B for 02, etc. According to selected theme
        
        # Get cluster keywords
        cluster_keywords = self.get_cluster_keywords(top_n=10)
        
        # Create summary data
        summary_data = []
        
        for cluster_id in sorted(self.data['cluster'].unique()):
            prefix = chr(ord('A') + int(self.theme_id) - 1)  # A for 01, B for 02, etc. According to theme selected
            cluster_name = f"{prefix}{cluster_id + 1}"
            
            # Get articles in this cluster
            cluster_articles = self.data[self.data['cluster'] == cluster_id]
            
            # Calculate statistics
            article_count = len(cluster_articles)
            publishers = cluster_articles['Publisher'].value_counts().to_dict()
            
            # Get keywords
            keywords = cluster_keywords.get(cluster_name, [])
            
            # Get sample titles
            sample_titles = cluster_articles['article_title'].head(5).tolist()
            
            # Add to summary data
            summary_data.append({
                'Cluster ID': cluster_name,
                'Article Count': article_count,
                'Keywords': ', '.join(keywords),
                'Publisher Distribution': str(publishers),
                'Sample Articles': '\n'.join(sample_titles) if sample_titles else 'No titles found'
            })
        
        # Create dataframe and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(output_path, index=False)
        print(f"Cluster summary saved to '{output_path}'")
        
        return summary_df
    
    def run_full_pipeline(self):
        """Run the complete clustering pipeline from data loading to output creation."""
        theme_name = self.THEMES.get(self.theme_id, 'Unknown Theme')
        print(f"\nProcessing theme {self.theme_id}: {theme_name}")
        print("=" * 60)
        
        try:
            # Step 1: Load masterlists
            masterlist_df = self.load_masterlists()
            
            if len(masterlist_df) > 0:
                # Step 2: Load article content
                articles_df = self.load_article_content()
                
                if len(articles_df) > 0:
                    # Step 3: Preprocess text
                    self.preprocess_text()
                    
                    # Step 4: Vectorize text
                    X = self.vectorize_text()
                    
                    if X is not None:
                        try:
                            # Step 5: Find optimal clusters
                            optimal_clusters = self.find_optimal_clusters(X)
                            
                            # Step 6: Cluster articles
                            self.cluster_articles(X, n_clusters=optimal_clusters)
                            
                            # Step 7: Get cluster keywords
                            self.get_cluster_keywords()
                            
                            # Step 8: Create outputs
                            masterlist = self.create_consolidated_masterlist()
                            summary = self.create_cluster_summary()
                            
                            # Print final statistics
                            publisher_distribution = masterlist['Publisher'].value_counts()
                            print("\nFinal distribution of articles by publisher:")
                            for publisher, count in publisher_distribution.items():
                                print(f"  {publisher}: {count} articles")
                            
                            print("\nClustering complete!")
                            print(f"Results saved to:")
                            print(f"  - '{self.theme_id}_consolidated_masterlist.xlsx'")
                            print(f"  - '{self.theme_id}_cluster_summary.xlsx'")
                            return True
                            
                        except Exception as e:
                            print(f"\nError during clustering: {str(e)}")
                            print("Trying simplified approach with 5 clusters...")
                            
                            # Fall back to simple approach
                            self.cluster_articles(X, n_clusters=5)
                            self.get_cluster_keywords()
                            self.create_consolidated_masterlist()
                            self.create_cluster_summary()
                            
                            print("\nClustering completed with simplified approach.")
                            return True
                    else:
                        print("Vectorization failed. Cannot proceed.")
                        return False
                else:
                    print("No article content loaded. Cannot proceed.")
                    return False
            else:
                print("No masterlist data loaded. Cannot proceed.")
                return False
                
        except Exception as e:
            print(f"\nCritical error: {str(e)}")
            print("Please check file paths and data format.")
            return False

def get_user_theme_choice():
    """Get the user's theme choice for clustering."""
    themes = {
        '01': 'Cybercrime & Digital Fraud',
        '02': 'Forensic Science & Criminal Investigations',
        '03': 'Medical Fraud & Malpractice',
        '04': 'Media Misinformation & Fake News',
        '05': 'Organised Crime & Drug Trafficking'
    }
    
    print("Available themes for clustering:")
    for theme_id, theme_name in themes.items():
        print(f"  {theme_id}: {theme_name}")
    
    while True:
        choice = input("\nEnter theme number (01-05) or 'all' to process all themes: ").strip()
        
        if choice.lower() == 'all':
            return list(themes.keys())
        elif choice in themes:
            return [choice]
        else:
            print("Invalid choice. Please enter a valid theme number (01-05) or 'all'.")


# In[94]:


def main():
    """Main function allowing user to select theme(s) to process."""
    print("=" * 60)
    print("NEWS ARTICLE CLUSTERING PROGRAM")
    print("=" * 60)
    
    # Get user's theme choice
    theme_choices = get_user_theme_choice()
    
    # Process selected themes
    successful = 0
    failed = 0
    
    for theme_id in theme_choices:
        try:
            # Create clustering object for this theme
            clustering = NewsArticleClustering(theme_id=theme_id)
            
            # Run full pipeline
            result = clustering.run_full_pipeline()
            
            if result:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            print(f"\nError processing theme {theme_id}: {str(e)}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print(f"Themes successfully processed: {successful}")
    print(f"Themes that failed: {failed}")
    print("=" * 60)


# In[95]:


if __name__ == "__main__":
    main()

