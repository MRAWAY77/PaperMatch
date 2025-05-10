#!/usr/bin/env python3
"""
Document Manager Module - For Summary Evaluation System

This module handles document access and content retrieval from index files.
"""

import os
import re
import glob
import json
import pandas as pd
import PyPDF2


class DocumentManager:
    """Manages document access using Excel index files with absolute paths"""
    
    def __init__(self, ap_index_file="file_index-ap.xlsx", na_index_file="file_index-na.xlsx", 
                 debug_mode=False, debug_log_file="debug_log.txt"):
        self.debug_mode = debug_mode
        self.debug_log_file = debug_log_file
        self.ap_index_file = ap_index_file
        self.na_index_file = na_index_file
        
        # Clear debug log file if in debug mode
        if debug_mode:
            with open(debug_log_file, 'w', encoding='utf-8') as f:
                f.write("--- Debug Log ---\n\n")
        
        # Load Excel index files
        self.ap_index = self._load_excel_index(ap_index_file, "academic papers")
        self.na_index = self._load_excel_index(na_index_file, "news articles")
        
        # Print column names for diagnostics
        self._print_columns()
        
        # Counters for tracking progress
        self.paths_found = 0
        self.texts_extracted = 0
    
    def debug_log(self, message):
        """Write a message to the debug log file if debug mode is enabled"""
        if self.debug_mode:
            with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
    
    def _load_excel_index(self, file_path, description):
        """Load an Excel index file"""
        try:
            if os.path.exists(file_path):
                index_df = pd.read_excel(file_path)
                print(f"Loaded {len(index_df)} {description} from {file_path}")
                
                # Log a sample in debug mode
                if self.debug_mode:
                    self.debug_log(f"\n--- First 3 rows of {file_path} ---")
                    self.debug_log(index_df.head(3).to_string())
                
                return index_df
            else:
                print(f"Warning: {description} index file not found at {file_path}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading {description} index: {e}")
            return pd.DataFrame()
    
    def _print_columns(self):
        """Print column names from index files to help diagnose issues"""
        if not self.ap_index.empty:
            print(f"Academic papers index columns: {list(self.ap_index.columns)}")
        if not self.na_index.empty:
            print(f"News articles index columns: {list(self.na_index.columns)}")
    
    def get_document_path(self, doc_id, doc_type):
        """Get the file path for a document based on its ID and type"""
        # Select appropriate index
        if doc_type == 'academic':
            index_df = self.ap_index
        else:  # doc_type == 'news'
            index_df = self.na_index
        
        if index_df.empty:
            return None
        
        try:
            # Look for the document ID in the 'filename' column
            if 'filename' in index_df.columns:
                matching_rows = index_df[index_df['filename'] == doc_id]
                
                if not matching_rows.empty:
                    # Get the path from the 'path' column
                    if 'path' in index_df.columns:
                        file_path = matching_rows.iloc[0]['path']
                        
                        # The path is an absolute path, check if it exists
                        if os.path.exists(file_path):
                            self.paths_found += 1
                            if self.debug_mode:
                                self.debug_log(f"Found path for {doc_id}: {file_path}")
                            return file_path
                        else:
                            if self.debug_mode:
                                self.debug_log(f"Path in index doesn't exist: {file_path}")
                            return None
                    else:
                        print(f"Error: 'path' column not found in {doc_type} index")
                else:
                    if self.debug_mode:
                        self.debug_log(f"No matching entry for {doc_id} in {doc_type} index")
                        # Check for similar entries for debugging
                        prefix = doc_id.split('_')[0] if '_' in doc_id else doc_id[:5]
                        similar = index_df[index_df['filename'].str.contains(prefix, na=False)]
                        if not similar.empty:
                            self.debug_log(f"Similar entries found: {similar['filename'].tolist()[:3]}")
            else:
                print(f"Error: 'filename' column not found in {doc_type} index")
                
        except Exception as e:
            print(f"Error finding path for {doc_id}: {str(e)}")
        
        return None
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    text += page_text + "\n\n"
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = text.strip()
            
            if not text:
                return f"[PDF content extraction failed - no text found in {os.path.basename(pdf_path)}]"
            
            self.texts_extracted += 1
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return f"[PDF content extraction error: {str(e)}]"
    
    def get_document_content(self, doc_id, doc_type):
        """Get the content of a document"""
        file_path = self.get_document_path(doc_id, doc_type)
        
        if not file_path:
            return None
        
        try:
            # Check file extension to determine how to read it
            _, ext = os.path.splitext(file_path)
            
            # If PDF, extract text from it
            if ext.lower() == '.pdf':
                return self.extract_text_from_pdf(file_path)
            
            # For text files, read the content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self.texts_extracted += 1
                return content
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None


# Utility function to find query directory
def find_query_dir():
    """Try to find a query directory"""
    # Look for directories that might contain queries
    possible_dirs = ["queries", "query", "eval", "logs", "XXX - Eval Logs", 
                     "C:\\Users\\Bertrand Tan\\Documents\\Python Scripts\\Web_Scraping\\Project Delphi\\0x - Evaluation\\XXX - Eval Logs (Gemma3)"]
    
    for dirname in possible_dirs:
        if os.path.isdir(dirname) and glob.glob(os.path.join(dirname, "query*.json")):
            return dirname
    
    # Check current directory
    if glob.glob("query*.json"):
        return "."
    
    return None


# Function to load queries from JSON files
def load_queries(query_dir, num_queries=None):
    """Load query data from JSON files"""
    query_files = sorted(glob.glob(os.path.join(query_dir, "query*.json")))
    
    if num_queries is not None:
        query_files = query_files[:num_queries]
    
    queries = []
    query_info = {}
    
    for query_file in query_files:
        try:
            with open(query_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            query_id = os.path.basename(query_file).replace('.json', '')
            query_data = {
                'query_id': query_id,
                'query': data.get('query', ''),
                'topic': data.get('classified_topic', ''),
                'academic_papers': data.get('top_academic_papers', []),
                'news_articles': data.get('top_news_articles', [])
            }
            
            # Store in our query_info dictionary for reference
            query_info[query_id] = query_data
            queries.append(query_data)
            
        except Exception as e:
            print(f"Error loading {query_file}: {e}")
    
    return queries, query_info


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    doc_manager = DocumentManager(debug_mode=True)
    
    # Find query directory
    query_dir = find_query_dir()
    if query_dir:
        print(f"Found query directory: {query_dir}")
        queries, query_info = load_queries(query_dir, num_queries=5)
        print(f"Loaded {len(queries)} queries")
        
        # Test document retrieval for first query
        if queries:
            first_query = queries[0]
            print(f"Testing document retrieval for query: {first_query['query']}")
            
            # Try to get an academic paper
            if first_query['academic_papers']:
                paper_id = first_query['academic_papers'][0].get('paper', '')
                print(f"Testing retrieval of academic paper: {paper_id}")
                content = doc_manager.get_document_content(paper_id, 'academic')
                if content:
                    print(f"Successfully retrieved content ({len(content)} characters)")
                else:
                    print("Failed to retrieve content")
            
            # Try to get a news article
            if first_query['news_articles']:
                article_id = first_query['news_articles'][0].get('article', '')
                print(f"Testing retrieval of news article: {article_id}")
                content = doc_manager.get_document_content(article_id, 'news')
                if content:
                    print(f"Successfully retrieved content ({len(content)} characters)")
                else:
                    print("Failed to retrieve content")
    else:
        print("No query directory found.")
