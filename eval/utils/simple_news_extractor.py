#!/usr/bin/env python3

# pip install pandas tqdm PyPDF2 openpyxl

"""
Simple News Article Summary Extractor

This script extracts ALL news article summaries and their metadata 
from PDF files without attempting to match them to source documents.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
import glob

# Try to import required modules
try:
    from pdf_extractor import PDFSummaryExtractor, find_summary_pdfs_dir
except ImportError:
    print("Error: Required modules not found. Make sure pdf_extractor.py is in the same directory.")
    print(f"Current directory: {os.getcwd()}")
    print("Files in directory:")
    for filename in os.listdir():
        print(f"  - {filename}")
    sys.exit(1)

def extract_news_summaries(output_file="news_article_summaries_simple.xlsx"):
    """
    Extract all news article summaries and metadata and save to Excel.
    
    Args:
        output_file (str): Path to save the output Excel file.
    """
    print("Simple News Article Summary Extractor")
    print("=====================================")
    
    # Step 1: Find PDF directory
    print("\nLocating PDF directory...")
    pdf_dir = find_summary_pdfs_dir()
    if not pdf_dir:
        print("Error: Could not find directory containing summary PDFs.")
        print("Please specify the directory containing summary PDFs.")
        return
    print(f"Found PDF directory: {pdf_dir}")
    
    # Step 2: Find all PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    print(f"Found {len(pdf_files)} PDF files")
    
    # Step 3: Initialize the extractor
    extractor = PDFSummaryExtractor(debug_mode=True)
    
    # Step 4: Process each PDF to extract summaries
    all_summaries = {}
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        try:
            pdf_name = os.path.basename(pdf_file)
            summaries = extractor.extract_summaries_from_pdf(pdf_file)
            if summaries:
                # Add the PDF source filename to each summary
                for title, data in summaries.items():
                    data['source_pdf'] = pdf_name
                all_summaries.update(summaries)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    print(f"Extracted {len(all_summaries)} total summaries")
    
    # Step 5: Filter for news articles
    news_summaries = {title: data for title, data in all_summaries.items() 
                     if data.get('type', '').lower() == 'news'}
    
    print(f"Found {len(news_summaries)} news article summaries")
    
    # Step 6: Prepare data for Excel
    news_data = []
    
    for title, data in tqdm(news_summaries.items(), desc="Preparing data"):
        # Calculate summary length
        summary_text = data.get('summary', '')
        summary_length = len(summary_text.split()) if summary_text else 0
        
        # Prepare row data
        article_data = {
            'title': title,
            'url': data.get('url', ''),
            'query': data.get('query', ''),
            'summary': sanitize_for_excel(summary_text),
            'summary_length': summary_length,
            'source_pdf': data.get('source_pdf', '')
        }
        
        news_data.append(article_data)
    
    # Create DataFrame
    if not news_data:
        print("No news article summaries found.")
        return
    
    df = pd.DataFrame(news_data)
    
    # Step 7: Save to Excel
    print(f"\nSaving {len(df)} news article summaries to {output_file}...")
    try:
        df.to_excel(output_file, index=False)
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        # Try saving as CSV if Excel fails
        csv_file = output_file.replace('.xlsx', '.csv')
        print(f"Attempting to save as CSV to {csv_file}...")
        df.to_csv(csv_file, index=False)
        print(f"Saved as CSV file: {csv_file}")
    
    print("\nExtraction complete!")
    return df

def sanitize_for_excel(text, max_length=5000):
    """Sanitize text to make it safe for Excel"""
    if not text:
        return ""
        
    # Truncate text if it's too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Remove control characters and other problematic characters
    import re
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract news article summaries and metadata to Excel")
    parser.add_argument('--output', '-o', type=str, default="news_article_summaries_simple.xlsx",
                       help="Output Excel file path")
    parser.add_argument('--pdf_dir', type=str, default=None,
                       help="Directory containing summary PDF files (optional)")
    
    args = parser.parse_args()
    
    # Override PDF directory if provided
    if args.pdf_dir:
        original_find_pdf_dir = find_summary_pdfs_dir
        find_summary_pdfs_dir = lambda: args.pdf_dir
    
    extract_news_summaries(args.output)
