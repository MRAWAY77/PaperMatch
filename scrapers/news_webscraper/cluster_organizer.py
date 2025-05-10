#!/usr/bin/env python
# coding: utf-8

"""
News Article Cluster Organizer
------------------------------
This program organizes news articles into folders based on their cluster assignments.
It reads the consolidated masterlists created by MetaCluster-I.py and:
1. Creates a folder structure with cluster names as folder names
2. Copies all articles belonging to each cluster to their respective folders
3. Preserves article content for easy analysis

Dependecies:
    pip install pandas
    pip install tqdm
    pip install openpyxl  # Required for pandas to read Excel files

Usage:
    python cluster_organizer.py

"""

import os
import shutil
import pandas as pd
import sys
from tqdm import tqdm  # For progress bars (install with pip if needed)

class ClusterOrganizer:
    def __init__(self):
        """Initialize the cluster organizer."""
        # Current working directory
        self.CWD = os.getcwd()
        
        # Define theme information
        self.THEMES = {
            '01': 'Cybercrime & Digital Fraud',
            '02': 'Forensic Science & Criminal Investigations',
            '03': 'Medical Fraud & Malpractice',
            '04': 'Media Misinformation & Fake News',
            '05': 'Organised Crime & Drug Trafficking'
        }
        
        # Masterlist paths
        self.MASTERLIST_FILES = [
            "01_consolidated_masterlist.xlsx",
            "02_consolidated_masterlist.xlsx",
            "03_consolidated_masterlist.xlsx",
            "04_consolidated_masterlist.xlsx",
            "05_consolidated_masterlist.xlsx"
        ]
        
        # Output directory
        self.OUTPUT_DIR = "Clustered_Articles"
        
        # Track statistics
        self.stats = {
            'themes_processed': 0,
            'clusters_created': 0,
            'articles_copied': 0,
            'articles_not_found': 0
        }
    
    def find_filename_column(self, df):
        """Find the column containing filenames in the masterlist."""
        # Check common filename column names
        candidate_cols = ['filename', 'file_name', 'file', 'path', 'article_file']
        
        # Look for exact matches first
        for col in candidate_cols:
            if col in df.columns:
                return col
        
        # Then look for partial matches
        for col in df.columns:
            col_lower = str(col).lower()
            if any(candidate in col_lower for candidate in ['file', 'name', 'path']):
                return col
        
        # If no obvious filename column, use the first column as fallback
        print(f"Warning: No filename column identified. Using first column: {df.columns[0]}")
        return df.columns[0]
    
    def create_output_directories(self):
        """Create the base output directory structure."""
        # Create main output directory
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            print(f"Created output directory: {self.OUTPUT_DIR}")
    
    def process_masterlist(self, masterlist_path):
        """
        Process a single consolidated masterlist.
        
        Args:
            masterlist_path: Path to the consolidated masterlist Excel file
        
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Extract theme ID from filename
            theme_id = os.path.basename(masterlist_path).split('_')[0]
            theme_name = self.THEMES.get(theme_id, 'Unknown Theme')
            
            print(f"\nProcessing theme {theme_id}: {theme_name}")
            print("=" * 60)
            
            # Load the masterlist
            df = pd.read_excel(masterlist_path)
            if len(df) == 0:
                print(f"Warning: Empty masterlist {masterlist_path}")
                return False
            
            print(f"Loaded {len(df)} articles from {masterlist_path}")
            
            # Find the filename column
            filename_col = self.find_filename_column(df)
            print(f"Using '{filename_col}' as the filename column")
            
            # Check for cluster column
            if 'cluster_name' not in df.columns:
                print(f"Error: 'cluster_name' column not found in {masterlist_path}")
                return False
            
            # Set path to articles repository
            articles_repo_path = os.path.join(f"{theme_id} - {theme_name}", "02 - Articles")
            if not os.path.exists(articles_repo_path):
                print(f"Warning: Article repository not found at {articles_repo_path}")
                # Try alternate format without spaces
                alternate_path = os.path.join(f"{theme_id}-{theme_name}", "02-Articles")
                if os.path.exists(alternate_path):
                    articles_repo_path = alternate_path
                    print(f"Using alternate path: {articles_repo_path}")
                else:
                    print(f"Error: Could not find article repository for theme {theme_id}")
                    return False
            
            # Get unique clusters
            clusters = df['cluster_name'].unique()
            print(f"Found {len(clusters)} clusters: {', '.join(sorted(clusters))}")
            
            # Create theme directory in output folder
            theme_output_dir = os.path.join(self.OUTPUT_DIR, f"{theme_id} - {theme_name}")
            if not os.path.exists(theme_output_dir):
                os.makedirs(theme_output_dir)
            
            # Process each cluster
            for cluster in sorted(clusters):
                # Create cluster directory
                cluster_dir = os.path.join(theme_output_dir, cluster)
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                    self.stats['clusters_created'] += 1
                
                # Get articles in this cluster
                cluster_articles = df[df['cluster_name'] == cluster]
                print(f"  Processing {len(cluster_articles)} articles in cluster {cluster}...")
                
                # Copy articles to cluster directory
                articles_found = 0
                articles_not_found = 0
                
                for _, article in tqdm(cluster_articles.iterrows(), total=len(cluster_articles), 
                                      desc=f"Cluster {cluster}", ncols=80):
                    # Get filename
                    filename = str(article[filename_col])
                    
                    # Add .txt extension if needed
                    if not filename.endswith('.txt'):
                        filename = f"{filename}.txt"
                    
                    # Source path
                    source_path = os.path.join(articles_repo_path, filename)
                    
                    # Destination path
                    dest_path = os.path.join(cluster_dir, filename)
                    
                    # Copy the file if it exists
                    if os.path.exists(source_path):
                        try:
                            shutil.copy2(source_path, dest_path)
                            articles_found += 1
                            self.stats['articles_copied'] += 1
                        except Exception as e:
                            print(f"    Error copying {filename}: {str(e)}")
                            articles_not_found += 1
                            self.stats['articles_not_found'] += 1
                    else:
                        # Try searching for the file (filenames might differ slightly)
                        found = False
                        base_name = os.path.splitext(filename)[0]
                        
                        if os.path.exists(articles_repo_path):
                            for file in os.listdir(articles_repo_path):
                                if base_name in file:
                                    try:
                                        source_path = os.path.join(articles_repo_path, file)
                                        shutil.copy2(source_path, dest_path)
                                        articles_found += 1
                                        self.stats['articles_copied'] += 1
                                        found = True
                                        break
                                    except Exception as e:
                                        print(f"    Error copying {file}: {str(e)}")
                        
                        if not found:
                            articles_not_found += 1
                            self.stats['articles_not_found'] += 1
                
                print(f"    Copied {articles_found} articles, {articles_not_found} not found")
            
            self.stats['themes_processed'] += 1
            return True
        
        except Exception as e:
            print(f"Error processing {masterlist_path}: {str(e)}")
            return False
    
    def create_readme(self):
        """Create a README file with information about the clustered articles."""
        readme_path = os.path.join(self.OUTPUT_DIR, "README.txt")
        
        with open(readme_path, 'w') as f:
            f.write("CLUSTERED NEWS ARTICLES\n")
            f.write("======================\n\n")
            f.write("This folder contains news articles organized by theme and cluster.\n\n")
            
            f.write("Themes:\n")
            for theme_id, theme_name in self.THEMES.items():
                f.write(f"  {theme_id}: {theme_name}\n")
            
            f.write("\nFolder Structure:\n")
            f.write("  Clustered_Articles/\n")
            f.write("  ├── [Theme ID] - [Theme Name]/\n")
            f.write("  │   ├── [Cluster Name 1]/\n")
            f.write("  │   │   ├── article1.txt\n")
            f.write("  │   │   ├── article2.txt\n")
            f.write("  │   │   └── ...\n")
            f.write("  │   ├── [Cluster Name 2]/\n")
            f.write("  │   └── ...\n")
            f.write("  └── README.txt\n\n")
            
            f.write("Statistics:\n")
            f.write(f"  Themes processed: {self.stats['themes_processed']}\n")
            f.write(f"  Clusters created: {self.stats['clusters_created']}\n")
            f.write(f"  Articles copied: {self.stats['articles_copied']}\n")
            f.write(f"  Articles not found: {self.stats['articles_not_found']}\n\n")
            
            f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Created README file at {readme_path}")
    
    def run(self):
        """Run the complete organization process."""
        print("=" * 60)
        print("NEWS ARTICLE CLUSTER ORGANIZER")
        print("=" * 60)
        
        # Create output directories
        self.create_output_directories()
        
        # Process each masterlist
        processed_count = 0
        for masterlist_file in self.MASTERLIST_FILES:
            # Check if the file exists
            if os.path.exists(masterlist_file):
                result = self.process_masterlist(masterlist_file)
                if result:
                    processed_count += 1
            else:
                print(f"Warning: Masterlist file not found: {masterlist_file}")
        
        # Create README file
        self.create_readme()
        
        # Print summary
        print("\n" + "=" * 60)
        print("ORGANIZATION SUMMARY")
        print(f"Themes processed: {self.stats['themes_processed']} of {len(self.THEMES)}")
        print(f"Clusters created: {self.stats['clusters_created']}")
        print(f"Articles copied: {self.stats['articles_copied']}")
        print(f"Articles not found: {self.stats['articles_not_found']}")
        print("=" * 60)
        
        print(f"\nArticles have been organized in the '{self.OUTPUT_DIR}' directory.")
        print("You can now compress this directory into a ZIP file.")
        print("\nTo create a ZIP file, you can use:")
        print(f"- File Explorer: Right-click on the '{self.OUTPUT_DIR}' folder and select 'Send to > Compressed (zipped) folder'")
        print("- Command line: Run the following command:")
        if os.name == 'nt':  # Windows
            print(f"  powershell Compress-Archive -Path '{self.OUTPUT_DIR}' -DestinationPath '{self.OUTPUT_DIR}.zip'")
        else:  # Linux/Mac
            print(f"  zip -r '{self.OUTPUT_DIR}.zip' '{self.OUTPUT_DIR}'")


if __name__ == "__main__":
    organizer = ClusterOrganizer()
    organizer.run()
