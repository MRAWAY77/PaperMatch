#!/usr/bin/env python3
"""
PDF Extractor Module - For Summary Evaluation System (Gemma 3 Version)

This module handles PDF processing, extracting summaries from PDF files with improved pattern matching.
"""

import os
import re
import json
import glob
from tqdm import tqdm
import PyPDF2


class PDFSummaryExtractor:
    """Extracts summaries from PDF files containing AI-generated summaries"""
    
    def __init__(self, debug_mode=False, debug_log_file="debug_log.txt"):
        self.debug_mode = debug_mode
        self.debug_log_file = debug_log_file
    
    def debug_log(self, message):
        """Write a message to the debug log file if debug mode is enabled"""
        if self.debug_mode:
            with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
    
    def extract_summaries_from_pdf(self, pdf_path):
        """Extract summaries from a PDF file containing AI-generated summaries"""
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found at {pdf_path}")
            return {}
        
        summaries = {}
        try:
            # Extract all text from the PDF
            full_text = self._extract_text_from_pdf(pdf_path)
            
            if self.debug_mode:
                self.debug_log(f"Extracted {len(full_text)} characters of text from PDF {os.path.basename(pdf_path)}")
                
            # Find the query first
            query_match = re.search(r'Query:\s*(.*?)(?=\n|$)', full_text)
            query = query_match.group(1).strip() if query_match else "Unknown Query"
            
            if self.debug_mode:
                self.debug_log(f"Found query: {query}")
            
            # First try the structured format (Type, Title, URL, Summary)
            summaries = self._extract_structured_format(full_text, query)
            
            # If no summaries found, try the original format
            if not summaries:
                summaries = self._extract_original_format(full_text, query)
            
            # If still no summaries found, try to find JSON
            if not summaries:
                summaries = self._extract_from_json(full_text, query)
            
            print(f"Extracted {len(summaries)} summaries from PDF file {os.path.basename(pdf_path)}")
            
            if self.debug_mode and summaries:
                self.debug_log(f"First summary titles: {list(summaries.keys())[:3]}")
            
        except Exception as e:
            print(f"Error extracting summaries from PDF: {e}")
        
        return summaries
    
    def _extract_structured_format(self, full_text, query):
        """Extract summaries using the structured format with Type/Title/URL/Summary markers"""
        summaries = {}
        
        # Pattern to look for the structured format with clear Type, Title, URL, Summary sections
        # This handles both formats: with and without Type: prefix
        pattern = r'(?:Type:?\s*(\w+)\s*\n)?Title:?\s*([^\n]+)\s*\nURL:?\s*(https?://[^\s]+)\s*\nSummary:?\s*((?:.+?\n)+?)(?=(?:Type|Title):|$)'
        matches = re.finditer(pattern, full_text, re.DOTALL | re.IGNORECASE)
        
        count = 0
        for match in matches:
            doc_type = match.group(1).lower() if match.group(1) else "unknown"
            title = match.group(2).strip()
            url = match.group(3).strip()
            summary = match.group(4).strip()
            
            # Skip entries with empty critical fields
            if not title or not summary:
                continue
            
            summaries[title] = {
                'type': doc_type,
                'title': title,
                'url': url,
                'summary': summary,
                'query': query
            }
            count += 1
        
        if self.debug_mode:
            self.debug_log(f"Extracted {count} summaries using structured format")
        
        return summaries
    
    def _extract_original_format(self, full_text, query):
        """Extract summaries using the original pattern format"""
        summaries = {}
        
        # Original pattern looking for Type, Title, URL, and Summary blocks
        pattern = r'(academic|news)\s+(.*?)\s+(https?://[^\s]+)\s+((?:This|The|Here).+?)(?=(academic|news)\s+|\[|Raw JSON Output|\Z)'
        matches = re.finditer(pattern, full_text, re.DOTALL | re.IGNORECASE)
        
        count = 0
        for match in matches:
            doc_type = match.group(1).lower()
            title = match.group(2).strip()
            url = match.group(3).strip()
            summary = match.group(4).strip()
            
            summaries[title] = {
                'type': doc_type,
                'title': title,
                'url': url,
                'summary': summary,
                'query': query
            }
            count += 1
        
        if self.debug_mode:
            self.debug_log(f"Extracted {count} summaries using original format")
        
        return summaries
    
    def _extract_from_json(self, full_text, query):
        """Extract summaries from the Raw JSON Output section if present"""
        summaries = {}
        
        # Look for sections in the Raw JSON Output
        json_match = re.search(r'Raw JSON Output:?\s*(\[.+?\])', full_text, re.DOTALL)
        if not json_match:
            # Try alternate format with just a JSON array
            json_match = re.search(r'(\[\s*\{\s*"type":.+?\}\s*\])', full_text, re.DOTALL)
        
        if json_match:
            try:
                json_text = json_match.group(1).strip()
                # Clean up the JSON text (it might have linebreaks or other issues)
                json_text = re.sub(r'\n\s*', ' ', json_text)
                # Handle potential truncated JSON
                if json_text.endswith(','):
                    json_text = json_text[:-1] + ']'
                json_data = json.loads(json_text)
                
                for item in json_data:
                    doc_type = item.get('type', '').lower()
                    title = item.get('title', '')
                    url = item.get('url', '')
                    summary = item.get('summary', '')
                    
                    if title and summary:  # Skip entries with empty critical fields
                        summaries[title] = {
                            'type': doc_type,
                            'title': title,
                            'url': url,
                            'summary': summary,
                            'query': query
                        }
            except Exception as e:
                if self.debug_mode:
                    self.debug_log(f"Error parsing JSON from PDF: {e}")
        
        if self.debug_mode:
            self.debug_log(f"Extracted {len(summaries)} summaries from JSON")
        
        return summaries
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract all text content from a PDF file"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def extract_summaries_from_directory(self, pdf_dir):
        """Extract summaries from all PDFs in a directory"""
        if not pdf_dir or not os.path.exists(pdf_dir):
            print(f"Warning: PDF directory not found at {pdf_dir}")
            return {}
        
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return {}
        
        print(f"Found {len(pdf_files)} PDF files in directory")
        
        # Initialize combined summaries dictionary
        all_summaries = {}
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
            try:
                new_summaries = self.extract_summaries_from_pdf(pdf_file)
                
                # Handle duplicate titles by appending a unique identifier
                for title, summary_data in new_summaries.items():
                    unique_title = title
                    counter = 1
                    while unique_title in all_summaries:
                        unique_title = f"{title} ({counter})"
                        counter += 1
                    
                    all_summaries[unique_title] = summary_data
                
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        
        print(f"Loaded {len(all_summaries)} total summaries from {len(pdf_files)} PDF files")
        return all_summaries


# Utility function to find summary PDFs directory
def find_summary_pdfs_dir():
    """Try to find directory containing summary PDFs"""
    possible_dirs = ["summaries", "summary", "eval", "XXX - Eval Logs",
                     "C:\\Users\\Bertrand Tan\\Documents\\Python Scripts\\Web_Scraping\\Project Delphi\\0x - Evaluation\\XXX - Eval Logs (Gemma3)"]
    
    for dirname in possible_dirs:
        if os.path.isdir(dirname) and glob.glob(os.path.join(dirname, "*.pdf")):
            return dirname
    
    return None


# Example usage
if __name__ == "__main__":
    # This will run if the script is executed directly
    extractor = PDFSummaryExtractor(debug_mode=True)
    
    # Find PDF directory
    pdf_dir = find_summary_pdfs_dir()
    if pdf_dir:
        print(f"Found PDF directory: {pdf_dir}")
        summaries = extractor.extract_summaries_from_directory(pdf_dir)
        print(f"Extracted {len(summaries)} summaries")
    else:
        print("No PDF directory found.")
