#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import time
import random
from tqdm import tqdm
import logging
from pathlib import Path


# In[56]:


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# In[57]:


class NewsArticleExtractor:
    def __init__(self, excel_file, output_dir="extracted_articles"):
        """
        Initialize the extractor with the Excel file path and output directory.
        
        Args:
            excel_file (str): Path to the Excel file with article metadata
            output_dir (str): Root directory to save extracted articles
        """
        self.excel_file = excel_file
        self.output_dir = output_dir
        
        # Extract source and topic from filename
        filename = os.path.basename(excel_file)
        parts = filename.replace('.xlsx', '').split('_')
        self.source = parts[0]  # e.g., ST, BBC, CNN
        
        # Extract topic - handle different naming patterns
        if 'Articles' in parts:
            articles_index = parts.index('Articles')
            self.topic = '_'.join(parts[1:articles_index])
        elif 'Processed' in parts:
            processed_index = parts.index('Processed')
            self.topic = '_'.join(parts[1:processed_index])
        else:
            self.topic = '_'.join(parts[1:3])  # Fallback
            
        # Create output directories
        self.create_directories()
        
        # Headers for requests to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    def create_directories(self):
        """Create the directory structure for the extracted articles"""
        # Main directories
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # By source, topic, and relevance
        Path(f"{self.output_dir}/by_source/{self.source}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/by_topic/{self.topic}").mkdir(parents=True, exist_ok=True)
        
        # Relevance directories
        for r in range(1, 6):
            Path(f"{self.output_dir}/by_relevance/R{r}_{'Most_' if r == 5 else 'Highly_' if r == 4 else 'Somewhat_' if r == 2 else 'Least_' if r == 1 else ''}Relevant").mkdir(parents=True, exist_ok=True)
    
    def clean_and_format_content(self, content_list):
        """
        Clean and format the extracted content, handling sub-headers and paragraphs
        Args:
            content_list (list): List of extracted content elements
        Returns:
            str: Cleaned and formatted content
        Note to Claude: DO NOT AMEND THIS DEF / FUNCTION
        """
        # Cleaned and formatted content elements
        formatted_content = []
        
        # Previous element tracking to manage formatting
        last_type = None
        
        for item in content_list:
            # Clean the item (remove extra whitespace)
            cleaned_item = re.sub(r'\s+', ' ', item.strip())
            
            # Skip completely empty items
            if not cleaned_item:
                continue
            
            # Identify item type
            if cleaned_item.startswith('##'):
                # Sub-header handling
                sub_header = cleaned_item.replace('##', '').strip()
                
                # Add extra newlines around sub-headers for clear separation
                formatted_content.append(f"\n\n## {sub_header}\n\n")
                last_type = 'subheader'
            else:
                # Regular paragraph handling
                if last_type == 'paragraph':
                    # Add newline between consecutive paragraphs
                    formatted_content.append(f"\n{cleaned_item}")
                else:
                    formatted_content.append(cleaned_item)
                last_type = 'paragraph'
        
        # Join content with careful formatting
        return "\n".join(formatted_content).strip()
                          
    def extract_article_text(self, soup, url):
        """
        Extract the article text from the soup object with hierarchical preservation and list handling
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            url (str): URL of the article
        
        Returns:
            str: Formatted article content
        """
        # List to collect structured content
        structured_content = []
        site = urlparse(url).netloc
        
        # Check if it's an archive buttons URL
        if 'archivebuttons.com' in site:
            # Detailed logging for debugging
            logger.debug(f"Extracting content from Archive Buttons URL: {url}")
            
            # Try multiple methods to find article content
            content_areas = [
                soup.select_one('div.text'),  # Specific class I noticed in the screenshot
                soup.select_one('div[style*="color:rgb(0, 0, 0)"]'),  # Selector based on text color
                soup.find('div', class_=lambda x: x and 'content' in x.lower()),
                soup.select_one('body'),  # Fallback to entire body
            ]
        
            # Process the first non-None content area
            for area in content_areas:
                if area:
                    # Track current header hierarchy
                    current_headers = {
                        'h1': None,
                        'h2': None,
                        'h3': None,
                        'h4': None,
                        'h5': None,
                        'h6': None
                    }
                    
                    # Track list nesting
                    current_list_level = 0
                    
                    # Find all elements maintaining order
                    elements = area.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol'])
                    
                    for element in elements:
                        # Process headers
                        if element.name.startswith('h'):
                            header_text = element.get_text().strip()
                            header_level = int(element.name[1])
                            
                            # Reset lower-level headers
                            for level in current_headers:
                                if int(level[1]) > header_level:
                                    current_headers[level] = None
                            
                            # Store current header
                            current_headers[element.name] = header_text
                            
                            # Construct header hierarchy
                            header_levels = [current_headers[f'h{i}'] for i in range(1, header_level + 1) if current_headers[f'h{i}']]
                            header_string = " > ".join(filter(None, header_levels))
                            
                            # Add header with full context
                            structured_content.append(f"## {header_string}")
                            
                            # Reset list level when a new header is encountered
                            current_list_level = 0
                        
                        # Process lists
                        elif element.name in ['ul', 'ol']:
                            current_list_level += 1
                            list_items = element.find_all('li', recursive=False)
                            
                            for item in list_items:
                                # Indent based on list nesting level
                                indent = '  ' * (current_list_level - 1)
                                list_item_text = item.get_text().strip()
                                
                                # Check if the list item contains nested lists
                                nested_lists = item.find_all(['ul', 'ol'], recursive=False)
                                
                                # Add the list item
                                if list_item_text:
                                    structured_content.append(f"{indent}• {list_item_text}")
                            
                                # Process nested lists recursively
                                if nested_lists:
                                    for nested_list in nested_lists:
                                        nested_items = nested_list.find_all('li', recursive=False)
                                        for nested_item in nested_items:
                                            nested_item_text = nested_item.get_text().strip()
                                            if nested_item_text:
                                                structured_content.append(f"{indent}  • {nested_item_text}")
                        
                        # Process paragraphs
                        elif element.name == 'p':
                            para_text = element.get_text().strip()
                            if para_text:
                                structured_content.append(para_text)
                    
                    # If content was found, break the loop
                    if structured_content:
                        break
            
            logger.debug(f"Extracted content from Archive Buttons: {structured_content}")
        
        # Original Straits Times website handling
        elif 'straitstimes.com' in site:
            # Headline and Subheadline extraction
            headlines = []
            h1_elements = soup.select('h1.headline, h1.font-secondary-header')
            if h1_elements:
                headlines.append(h1_elements[0].get_text().strip())
            
            h2_elements = soup.select('h2.font-secondary-header')
            if h2_elements:
                headlines.append(h2_elements[0].get_text().strip())
            
            # Add headlines/subheadlines to content
            structured_content.extend(headlines)
            
            # Article content extraction
            article_content = soup.select('section.article-content')
            if article_content:
                for section in article_content:
                    # Track current header level
                    current_headers = {
                        'h1': None,
                        'h2': None,
                        'h3': None,
                        'h4': None,
                        'h5': None,
                        'h6': None
                    }
                
                    # Track list nesting
                    current_list_level = 0
                    
                    # Collect all elements in order
                    elements = section.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol'])
                    
                    for element in elements:
                        # Process headers
                        if element.name.startswith('h'):
                            header_text = element.get_text().strip()
                            header_level = int(element.name[1])
                            
                            # Reset lower-level headers
                            for level in current_headers:
                                if int(level[1]) > header_level:
                                    current_headers[level] = None
                            
                            # Store current header
                            current_headers[element.name] = header_text
                            
                            # Construct header hierarchy
                            header_levels = [current_headers[f'h{i}'] for i in range(1, header_level + 1) if current_headers[f'h{i}']]
                            header_string = " > ".join(filter(None, header_levels))
                            
                            # Add header with full context
                            structured_content.append(f"## {header_string}")
                            
                            # Reset list level when a new header is encountered
                            current_list_level = 0
                        
                        # Process lists
                        elif element.name in ['ul', 'ol']:
                            current_list_level += 1
                            list_items = element.find_all('li', recursive=False)
                        
                            for item in list_items:
                                # Indent based on list nesting level
                                indent = '  ' * (current_list_level - 1)
                                list_item_text = item.get_text().strip()
                                
                                # Check if the list item contains nested lists
                                nested_lists = item.find_all(['ul', 'ol'], recursive=False)
                                
                                # Add the list item
                                if list_item_text:
                                    structured_content.append(f"{indent}• {list_item_text}")
                                
                                # Process nested lists recursively
                                if nested_lists:
                                    for nested_list in nested_lists:
                                        nested_items = nested_list.find_all('li', recursive=False)
                                        for nested_item in nested_items:
                                            nested_item_text = nested_item.get_text().strip()
                                            if nested_item_text:
                                                structured_content.append(f"{indent}  • {nested_item_text}")
                    
                        # Process paragraphs
                        elif element.name == 'p':
                            # Skip empty or ad paragraphs
                            parent_id = element.parent.get('id', '')
                            p_text = element.get_text().strip()
                            if p_text and not (parent_id.startswith('dfp-ad') or parent_id.startswith('sph_')):
                                # Check for paragraph-specific selectors used in Straits Times
                                if element.get('class') and 'paragraph-base' in element.get('class'):
                                    structured_content.append(p_text)
                                elif not element.get('class'):
                                    structured_content.append(p_text)

########$###################################################################        
        # BBC website handling
        elif 'bbc.com' in site or 'bbc.co.uk' in site:
            # Headline extraction - only extract once
            headlines = []
            h1_elements = soup.select('h1.sc-18f6de06-0, h1.ssrcss-1q0x1qg-Paragraph')
            if h1_elements:
                headlines.append(h1_elements[0].get_text().strip())
            
            # Add headlines to content
            structured_content.extend(headlines)
            
            # Track current header hierarchy
            current_headers = {
                'h1': headlines[0] if headlines else None,
                'h2': None,
                'h3': None,
                'h4': None,
                'h5': None,
                'h6': None
            }

            # Track list nesting
            current_list_level = 0
            
            # Important fix: Choose only ONE content area to extract from
            # Previously multiple areas could contain duplicated content
            content_area = None
            
            # Try to find the main content area in order of preference
            potential_areas = [
                soup.select_one('main#main-content'),
                soup.select_one('div.article-body'),
                soup.select_one('div[data-component="text-block"]').parent if soup.select_one('div[data-component="text-block"]') else None,
                soup.select_one('div[data-component="rich-text-block"]').parent if soup.select_one('div[data-component="rich-text-block"]') else None
            ]
            
            # Use the first valid content area
            for area in potential_areas:
                if area:
                    content_area = area
                    break
            
            if content_area:
                # Collect all elements maintaining order
                elements = content_area.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'b'])
        
                for element in elements:
                    # Process headers
                    if element.name.startswith('h') and element.name != 'h1':
                        header_text = element.get_text().strip()
                        header_level = int(element.name[1])
                        
                        # Reset lower-level headers
                        for level in current_headers:
                            if int(level[1]) > header_level:
                                current_headers[level] = None
                        
                        # Store current header
                        current_headers[element.name] = header_text
                        
                        # Construct header hierarchy
                        header_levels = [current_headers[f'h{i}'] for i in range(1, header_level + 1) if current_headers[f'h{i}']]
                        header_string = " > ".join(filter(None, header_levels))
                        
                        # Add header with full context
                        structured_content.append(f"## {header_string}")
                        
                        # Reset list level when a new header is encountered
                        current_list_level = 0
            
                    # Process paragraphs - handle multiple BBC formats
                    elif element.name == 'p':
                        # Get class attribute as a string or empty list if no class
                        classes = element.get('class') or []
                        class_str = ' '.join(classes) if classes else ''
                        
                        # Check for any of the known BBC paragraph patterns:
                        # 1. Classes that start with 'sc-' (older pattern)
                        # 2. Classes that contain '-Paragraph' (newer pattern)
                        # 3. Classes that start with 'ssrcss-' (newer pattern)
                        # 4. No class at all (fallback)
                        has_sc_class = any(isinstance(cls, str) and cls.startswith('sc-') for cls in classes)
                        has_paragraph_class = 'Paragraph' in class_str
                        has_ssrcss_class = any(isinstance(cls, str) and cls.startswith('ssrcss-') for cls in classes)
                        
                        if has_sc_class or has_paragraph_class or has_ssrcss_class or not classes:
                            p_text = element.get_text().strip()
                            if p_text:
                                structured_content.append(p_text)
            
                    # Process bold text (sometimes contains important content)
                    elif element.name == 'b':
                        b_text = element.get_text().strip()
                        if b_text and len(b_text) > 20:  # Only include substantial bold text
                            structured_content.append(b_text)
                    
                    # Process lists
                    elif element.name in ['ul', 'ol']:
                        current_list_level += 1
                        list_items = element.find_all('li', recursive=False)
                        
                        for item in list_items:
                            # Indent based on list nesting level
                            indent = '  ' * (current_list_level - 1)
                            list_item_text = item.get_text().strip()
                            
                            # Check if the list item contains nested lists
                            nested_lists = item.find_all(['ul', 'ol'], recursive=False)
                            
                            # Add the list item
                            if list_item_text:
                                structured_content.append(f"{indent}• {list_item_text}")
                    
                            # Process nested lists recursively
                            if nested_lists:
                                for nested_list in nested_lists:
                                    nested_items = nested_list.find_all('li', recursive=False)
                                    for nested_item in nested_items:
                                        nested_item_text = nested_item.get_text().strip()
                                        if nested_item_text:
                                            structured_content.append(f"{indent}  • {nested_item_text}")
            
            # Additional fix: Deduplicate content
            # Sometimes the same content might still be extracted twice from different elements
            deduplicated_content = []
            seen_content = set()
            
            for item in structured_content:
                # Skip this item if it's identical to the previous one
                if item in seen_content:
                    continue
                    
                deduplicated_content.append(item)
                seen_content.add(item)
            
            # Replace the original list with deduplicated content
            structured_content = deduplicated_content
            
            # Logging
            logger.debug(f"BBC Article Extraction Debug:")
            logger.debug(f"Total content items after deduplication: {len(structured_content)}")
            
########$###################################################################
        # Optimized Channel News Asia (CNA) website handling with excluded sections
        elif 'channelnewsasia.com' in site or 'cnalifestyle.channelnewsasia.com' in site:
            # Headline extraction
            headlines = []
            h1_elements = soup.select('h1.layout__title, h1.h1--page-title, div.article-header h1')
            if h1_elements:
                headlines.append(h1_elements[0].get_text().strip())
            
            # Add headlines to content
            structured_content.extend(headlines)
            
            # Define termination markers and AI disclaimers once
            termination_markers = [
                "Subscribe to our Chief Editor's Week in Review",
                "Get our pick of top stories and thought-provoking articles in your inbox"
            ]
            ai_disclaimers = ["This audio is generated by an AI tool"]
            
            # Define sections to exclude
            excluded_sections = [
                'div.referenced-card',    # "Related:" section with links to other articles
                '.elementor-grid-item',   # Other potential "related content" containers
                '.media-object',          # Media object containers that often contain related articles
                '.teaser',                # Article teasers
                'div.recommended-articles', # Recommended articles section
                '.read-more',             # "Read more" sections
                '.article-tags',          # Article tags
                '.article-social-share'   # Social share buttons
            ]
    
            # Check if an element is inside an excluded section
            def is_in_excluded_section(element):
                # Check if the element has any of the excluded classes
                for cls in excluded_sections:
                    # Strip the CSS selector prefix if present
                    class_name = cls.replace('.', '')
                    if class_name in (element.get('class') or []):
                        return True
                        
                # Check if any parent is in an excluded section
                parent = element.parent
                while parent and parent.name != 'body':
                    if parent.get('class'):
                        for cls in excluded_sections:
                            class_name = cls.replace('.', '')
                            if class_name in parent.get('class'):
                                return True
                    parent = parent.parent
                    
                return False
            
            # Function to check if text should be skipped or signals termination
            def should_skip_or_terminate(text):
                if any(marker in text for marker in termination_markers):
                    logger.info("Found newsletter subscription marker - terminating extraction")
                    return "terminate"
                elif any(disclaimer in text for disclaimer in ai_disclaimers):
                    return "skip"
                return False
    
            # Find all potential content containers in one go
            content_containers = soup.select(
                'section[data-title="Content"], '
                'div.layout_region, '
                'div.article-body, '
                'div.article-content, '
                'div.content-wrapper, '
                'div.text-long, '
                'div.text'
            )
            
            # Extract text content from all containers
            all_content = []
            terminate_extraction = False
            
            for container in content_containers:
                if terminate_extraction:
                    break
                    
                # Process all text elements (paragraphs and spans)
                text_elements = container.select('p, span')
                
                for element in text_elements:
                    # Skip elements that are likely not actual content
                    if element.parent.name == 'p' and element.name == 'span':
                        # Skip spans inside paragraphs as we'll get their text from the paragraph
                        continue
                        
                    # Skip elements in excluded sections
                    if is_in_excluded_section(element):
                        continue
                
                    text = element.get_text().strip()
                    
                    # Skip empty elements
                    if not text or len(text) <= 1:
                        continue
                        
                    # Check if we should skip or terminate
                    check_result = should_skip_or_terminate(text)
                    if check_result == "terminate":
                        terminate_extraction = True
                        break
                    elif check_result == "skip":
                        continue
                        
                    # Add content if it's not a duplicate of the last item
                    if not all_content or text != all_content[-1]:
                        all_content.append(text)
                
                # Process headers in the container but exclude those in excluded sections
                if not terminate_extraction:
                    headers = container.select('h2, h3, h4, h5, h6')
                    for header in headers:
                        if is_in_excluded_section(header):
                            continue
                    
                        header_text = header.get_text().strip()
                        if header_text:
                            structured_content.append(f"## {header_text}")
                    
                    # Process lists in the container but exclude those in excluded sections
                    lists = container.select('ul, ol')
                    for list_elem in lists:
                        if is_in_excluded_section(list_elem):
                            continue
                            
                        list_items = list_elem.select('li')
                        for item in list_items:
                            item_text = item.get_text().strip()
                            if item_text:
                                structured_content.append(f"• {item_text}")
            
            # Add the collected content to structured_content
            structured_content.extend(all_content)
            
            # Deduplicate content more efficiently
            deduplicated_content = []
            seen_content = set()
            
            for item in structured_content:
                # Normalize whitespace for comparison
                item_normalized = re.sub(r'\s+', ' ', item).strip()
                
                # Skip if we've seen this exact content
                if item_normalized in seen_content:
                    continue
            
                # Skip if this content is part of another item we've already included
                if any(item_normalized in seen and len(seen) > len(item_normalized) * 1.2 for seen in seen_content):
                    continue
                
                deduplicated_content.append(item)
                seen_content.add(item_normalized)
            
            # Replace the original list with deduplicated content
            structured_content = deduplicated_content
            
            # Log extraction results
            logger.debug(f"CNA Article Extraction: Found {len(structured_content)} content items")      

########$###################################################################
        # CNN website handling
        elif 'cnn.com' in site or 'edition.cnn.com' in site:
            # Headline extraction
            headlines = []
            h1_selectors = [
                'h1.headline__text',
                'h1.inline-placeholder',
                'h1[data-editable="headlineText"]',
                'h1.headline'
            ]
            
            for selector in h1_selectors:
                h1_elements = soup.select(selector)
                if h1_elements:
                    headlines.append(h1_elements[0].get_text().strip())
                    break
            
            # Add headlines to content
            structured_content = headlines.copy() if headlines else []
    
            # Define sections to exclude
            excluded_sections = [
                # "Up Next" sections
                'div[data-title="Up next"]',
                'section.layout_end',
                'div.container_list-headlines-with-read-times',
                'div[data-collapsed-text="Up next"]',
                # "Most Read" sections
                'div[data-title="Most read"]',
                'div.container_list-headlines-ranked',
                'div[data-collapsed-text="Most read"]',
                # Other sections to exclude
                'div.ad-slot-dynamic',
                'div.ad-feedback',
                'div.zoneAds',
                'div.related-content',
                'div.related-articles',
                '.newsletter-container'
            ]
    
            # Function to check if element should be excluded
            def is_excluded_element(element):
                # Check element itself
                if element.name == 'div' and element.get('data-title') in ["Up next", "Most read"]:
                    return True
                    
                # Check for collapsed text markers
                if element.get('data-collapsed-text') in ["Up next", "Most read"]:
                    return True
                    
                # Check if the element is within an excluded section
                parent = element.parent
                while parent:
                    # Check if parent has data-title attribute that matches excluded sections
                    if parent.get('data-title') in ["Up next", "Most read"]:
                        return True
                        
                    # Check if parent has data-collapsed-text attribute that matches excluded sections
                    if parent.get('data-collapsed-text') in ["Up next", "Most read"]:
                        return True
                        
                    # Check if parent has any of the excluded classes
                    if parent.get('class'):
                        parent_class = ' '.join(parent.get('class'))
                        if any(x in parent_class for x in ['container_list-headlines-with-read-times', 
                                                          'container_list-headlines-ranked',
                                                          'ad-slot-dynamic']):
                            return True
            
                    parent = parent.parent
                
                return False
            
            # Find main content container
            main_content = None
            for selector in [
                'section[data-tabcontent="content"]',
                'main.article__main',
                'div.article__content',
                'div.article_content',
                'div.l-container',
                'div.body-text'
            ]:
                container = soup.select_one(selector)
                if container:
                    main_content = container
                    break
            
            # Process elements in the main content
            if main_content:
                # Get all text elements in the right order
                article_elements = main_content.find_all(['h2', 'h3', 'h4', 'p', 'ul', 'ol'])
                
                for element in article_elements:
                    # Skip elements in excluded sections
                    if is_excluded_element(element):
                        continue
            
                    # Process headers
                    if element.name.startswith('h'):
                        header_text = element.get_text().strip()
                        if header_text and len(header_text) > 1:
                            structured_content.append(f"## {header_text}")
                    
                    # Process paragraphs
                    elif element.name == 'p':
                        p_text = element.get_text().strip()
                        
                        # Skip empty paragraphs
                        if not p_text or len(p_text) <= 1:
                            continue
                            
                        # Skip paragraphs with "This is a developing story" text
                        if "This is a developing story" in p_text:
                            continue
                            
                        structured_content.append(p_text)
                    
                    # Process lists
                    elif element.name in ['ul', 'ol']:
                        list_items = element.find_all('li')
                        for item in list_items:
                            item_text = item.get_text().strip()
                            if item_text:
                                structured_content.append(f"• {item_text}")
    
            else:
                # Fallback to simple paragraph extraction if main container not found
                paragraphs = soup.find_all('p', class_=lambda x: x and ('paragraph' in x or 'article' in x))
                
                for p in paragraphs:
                    # Skip elements in excluded sections
                    if is_excluded_element(p):
                        continue
                        
                    p_text = p.get_text().strip()
                    if p_text and len(p_text) > 1:
                        structured_content.append(p_text)
            
            # Deduplicate content
            deduplicated_content = []
            seen_content = set()
            
            for item in structured_content:
                # Normalize whitespace for comparison
                item_normalized = re.sub(r'\s+', ' ', item).strip()
                
                # Skip if we've seen this exact content
                if item_normalized in seen_content:
                    continue
                    
                deduplicated_content.append(item)
                seen_content.add(item_normalized)
            
            # Replace the original list with deduplicated content
            structured_content = deduplicated_content
            
            # Log extraction results
            logger.debug(f"CNN Article Extraction: Found {len(structured_content)} content items")
        
########$###################################################################

        # Fallback if no content found
        if not structured_content:
            logger.warning(f"No content extracted from URL: {url}")
            paragraphs = soup.find_all('p')
            structured_content = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        
        # Clean and format the extracted content
        formatted_content = self.clean_and_format_content(structured_content)
        
        return formatted_content

    # Enhance the journalist extraction function to better capture BBC journalist names
    # This should be added to the extract_journalist method in the NewsArticleExtractor class
    
    def extract_journalist(self, soup, url):
        """
        Extract the journalist name from the article
        Args:
            soup (BeautifulSoup): Parsed HTML content
            url (str): URL of the article
        Returns:
            str or None: Extracted journalist name
        """
        site = urlparse(url).netloc
    
        if 'bbc.com' in site or 'bbc.co.uk' in site:
            # Method 1: Look for byline block section with journalist names
            byline_block = soup.select_one('div[data-component="byline-block"]')
            if byline_block:
                # Find all journalist name spans within the byline block
                # These often have classes like "sc-XXXXXX-7 kItaYD" or similar naming patterns
                journalist_spans = byline_block.select('span[class*="kItaYD"], span[class*="byline-name"]')
                
                if journalist_spans:
                    # Collect all journalist names
                    journalists = [span.get_text().strip() for span in journalist_spans if span.get_text().strip()]
                    if journalists:
                        # Join multiple journalists with commas
                        return ", ".join(journalists)
        
            # Method 2: Look for specific byline containers with contributor details
            byline_containers = soup.select('div[data-testid="byline-new"], div[data-testid="byline-new-contributors"]')
            if byline_containers:
                journalists = []
                for container in byline_containers:
                    # Look for name spans within the container
                    # BBC often uses class names with patterns like "sc-XXXXXX-7" for journalist names
                    name_spans = container.select('span[class*="-7"]')
                    for span in name_spans:
                        name = span.get_text().strip()
                        if name and len(name) > 2:  # Basic validation to ensure it looks like a name
                            journalists.append(name)
                
                if journalists:
                    return ", ".join(journalists)
                
            # Method 3: Look for byline sections with specific class patterns
            byline_sections = soup.select('.byline, .byline-name, .author-name, div[class*="byline"]')
            for section in byline_sections:
                text = section.get_text().strip()
                if text and "By " in text:
                    return text.replace("By ", "").strip()
                elif text and len(text) > 2:
                    return text


        elif 'straitstimes.com' in site:
            # Original Straits Times logic
            byline_name_links = soup.select('a.text-blue-hyperlink.byline-name')
            if byline_name_links:
                for link in byline_name_links:
                    journalist_name = link.get_text().strip()
                    if journalist_name:
                        return journalist_name
        
            # Look for photo captions with news agency or journalist name
            photo_captions = soup.select('.photo-caption, figcaption')
            for caption in photo_captions:
                text = caption.get_text().strip()
                
                # Improved regex to match full names
                name_matches = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b(?=\s*/)', text)
                if name_matches:
                    return name_matches[0]
                
                # Check for news agencies
                news_agencies = ['REUTERS', 'AP', 'AFP', 'ASSOCIATED PRESS']
                for agency in news_agencies:
                    if agency in text.upper():
                        return agency
    
        # Final paragraph check for news agencies
        paragraphs = soup.select('p.paragraph-base, p')
        if paragraphs:
            last_paragraph = paragraphs[-1].get_text().strip()
            news_agencies = ['REUTERS', 'AP', 'AFP', 'ASSOCIATED PRESS', 'BBC News']
            for agency in news_agencies:
                if agency in last_paragraph:
                    return agency

########$#######################################################################################
        # Optimized Channel News Asia (CNA) extraction with proper heading placement
        elif 'channelnewsasia.com' in site or 'cnalifestyle.channelnewsasia.com' in site:
            # Headline extraction
            headlines = []
            h1_elements = soup.select('h1.layout__title, h1.h1--page-title, div.article-header h1')
            if h1_elements:
                headlines.append(h1_elements[0].get_text().strip())
            
            # Define termination markers and AI disclaimers
            termination_markers = [
                "Subscribe to our Chief Editor's Week in Review",
                "Get our pick of top stories and thought-provoking articles in your inbox"
            ]
            ai_disclaimers = ["This audio is generated by an AI tool"]
            
            # Define sections to exclude
            excluded_sections = [
                'div.referenced-card',    # "Related:" section with links to other articles
                '.elementor-grid-item',   # Grid items for other articles
                '.media-object',          # Media object containers (often related articles)
                '.teaser',                # Article teasers
                'div.recommended-articles', # Recommended articles section
                '.read-more',             # "Read more" sections
                '.article-tags',          # Article tags
                '.article-social-share',  # Social share buttons
                '[data-title="Related"]'  # Related content sections
            ]
    
            # Helper function to check if element should be excluded
            def should_exclude(element):
                # Check if the element itself has an excluded class
                if element.get('class'):
                    element_classes = ' '.join(element.get('class'))
                    for selector in excluded_sections:
                        class_name = selector.replace('.', '')
                        if class_name in element_classes:
                            return True
                
                # Check parent elements
                parent = element.parent
                while parent and parent.name != 'body':
                    if parent.get('class'):
                        parent_classes = ' '.join(parent.get('class'))
                        for selector in excluded_sections:
                            class_name = selector.replace('.', '')
                            if class_name in parent_classes:
                                return True
                    
                    # Check for data attributes
                    if parent.get('data-title') == 'Related':
                        return True
                        
                    parent = parent.parent
                    
                return False
    
            # Function to check if text should be skipped or signals termination
            def should_skip_or_terminate(text):
                if any(marker in text for marker in termination_markers):
                    logger.info("Found newsletter subscription marker - terminating extraction")
                    return "terminate"
                elif any(disclaimer in text for disclaimer in ai_disclaimers):
                    return "skip"
                return False
            
            # Add the main headline first
            structured_content = headlines.copy()
            
            # Find the main content container - more likely to have the correct element ordering
            main_content = None
            content_selectors = [
                'section[data-title="Content"]',
                'div.layout_region',
                'div.article-body',
                'div.article-content',
                'div.content-wrapper'
            ]
            
            for selector in content_selectors:
                container = soup.select_one(selector)
                if container:
                    main_content = container
                    break
    
            # If we found the main content, process all elements in document order
            if main_content:
                # Get ALL elements in their natural document order
                all_elements = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'div'])
                
                # Process elements in the order they appear in the document
                terminate_extraction = False
                
                for element in all_elements:
                    if terminate_extraction:
                        break
                        
                    # Skip elements in excluded sections
                    if should_exclude(element):
                        continue
                    
                    # Process headers
                    if element.name.startswith('h') and element.name != 'h1':
                        header_text = element.get_text().strip()
                        if header_text and len(header_text) > 1:
                            # Add header as a markdown section
                            structured_content.append(f"## {header_text}")
                    
                    # Process paragraphs
                    elif element.name == 'p':
                        p_text = element.get_text().strip()
                        
                        # Skip empty paragraphs or very short ones
                        if not p_text or len(p_text) <= 1:
                            continue
                    
                        # Check for termination markers
                        check_result = should_skip_or_terminate(p_text)
                        if check_result == "terminate":
                            terminate_extraction = True
                            break
                        elif check_result == "skip":
                            continue
                        
                        # Add the paragraph text
                        structured_content.append(p_text)
                    
                    # Process lists
                    elif element.name in ['ul', 'ol']:
                        list_items = element.find_all('li')
                        for item in list_items:
                            item_text = item.get_text().strip()
                            if item_text:
                                structured_content.append(f"• {item_text}")
                    
                    # Process divs that might contain text (but only if they have text-long class)
                    elif element.name == 'div' and element.get('class') and 'text-long' in element.get('class'):
                        # Only process if direct text (not through child elements we'll process separately)
                        direct_text = ''.join([text for text in element.contents if isinstance(text, str)]).strip()
                        if direct_text and len(direct_text) > 1:
                            structured_content.append(direct_text)
            else:
                # Fallback to old approach if main content container not found
                logger.warning("Main content container not found, using fallback approach")
                
                # This is a simplified version of the old approach
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    if should_exclude(p):
                        continue
                        
                    p_text = p.get_text().strip()
                    if p_text and len(p_text) > 1:
                        structured_content.append(p_text)
            
            # Deduplicate content
            deduplicated_content = []
            seen_content = set()
            
            for item in structured_content:
                # Normalize whitespace for comparison
                item_normalized = re.sub(r'\s+', ' ', item).strip()
                
                # Skip if we've seen this exact content
                if item_normalized in seen_content:
                    continue
                
                deduplicated_content.append(item)
                seen_content.add(item_normalized)
            
            # Replace the original list with deduplicated content
            structured_content = deduplicated_content
    
            # Log extraction results
            logger.debug(f"CNA Article Extraction: Found {len(structured_content)} content items")

########$#######################################################################################        
        # Add this to your extract_journalist method
        
        # CNN journalist extraction
        elif 'cnn.com' in site or 'edition.cnn.com' in site:
            # Method 1: Look for byline section containing journalist names
            byline_selectors = [
                'div[data-component-name="byline"]',
                'div.byline',
                'div.byline_names',
                'div.headline__byline',
                'div[class*="byline"]'
            ]
            
            for selector in byline_selectors:
                byline_elements = soup.select(selector)
                if byline_elements:
                    for element in byline_elements:
                        # Extract all text from byline
                        byline_text = element.get_text().strip()
                        
                        # Clean up byline text
                        byline_text = byline_text.replace("By ", "").strip()
                        if byline_text and len(byline_text) > 3:
                            return byline_text
    
            # Method 2: Look for specific byline elements (like those in screenshots)
            byline_name_selectors = [
                'span.byline__name',
                'span.byline_name',
                'a[class*="byline_link"]',
                'div[class*="byline-sub-text"]'
            ]
            
            all_names = []
            for selector in byline_name_selectors:
                name_elements = soup.select(selector)
                for element in name_elements:
                    name = element.get_text().strip()
                    if name and len(name) > 3 and name not in all_names:
                        all_names.append(name)
            
            if all_names:
                return ", ".join(all_names)
            
            # Method 3: Look for common CNN byline formats
            # Pattern: "By [Author Name], CNN"
            byline_patterns = [
                r'By\s+([\w\s]+),\s+CNN',
                r'By\s+([\w\s]+)\s+and\s+([\w\s]+),\s+CNN',
                r'By\s+([\w\s]+),\s+([\w\s]+)\s+and\s+([\w\s]+),\s+CNN'
            ]
            
            page_text = soup.get_text()
            for pattern in byline_patterns:
                match = re.search(pattern, page_text)
                if match:
                    # Join all captured groups
                    return ", ".join(match.groups())
    
            # Method 4: Extract from byline text in paragraphs
            for p in soup.find_all('p'):
                text = p.get_text().strip()
                if text.startswith('By ') and ', CNN' in text and len(text) < 100:
                    return text.replace('By ', '').replace(', CNN', '').strip()
                    
            # Method 5: Check for CNN staff/service attribution
            for p in soup.find_all('p')[-3:]:  # Check last few paragraphs
                text = p.get_text().strip()
                if 'CNN' in text and any(x in text.lower() for x in ['contributed', 'reporting', 'service']):
                    # Extract just the CNN attribution
                    return "CNN"
                    
            # Default to CNN if we found nothing else
            return "CNN"

########$#######################################################################################

    def generate_filename(self, article):
        """
        Generate a filename for an article based on the naming convention
        
        Args:
            article (dict): Article metadata dictionary
        
        Returns:
            str: Generated filename
        """
        # Get date in YYYYMMDD format
        try:
            date_obj = pd.to_datetime(article.get('Date'))
            date_str = date_obj.strftime('%Y%m%d')
        except:
            date_str = '00000000'
        
        # Get source 
        source = article.get('Media', self.source)
        source = str(source).replace('The ', '').strip().split()[0]
        
        # Get topic
        topic = article.get('Theme', self.topic)
        if isinstance(topic, str):
            # Standardize topic names
            topic_mapping = {
                'Cybercrime': 'Cybercrime',
                'Forensic': 'Forensic',
                'Misinformation': 'Misinformation',
                'Medical Fraud': 'MedicalFraud',
                'Organised Crime': 'OrganisedCrime'
            }
            topic = next((mapped for key, mapped in topic_mapping.items() if key in topic), topic)
        
        # Get relevance score
        relevance = article.get('Relevance', 0)
        
        # Clean title for filename
        title = self.clean_title_for_filename(article.get('Title', ''))
        
        # Generate filename
        filename = f"{date_str}_{source}{topic}R{relevance}_{title}.txt"
        
        return filename
    
    def clean_title_for_filename(self, title):
        """
        Clean a title for use in a filename
        
        Args:
            title (str): Original title
        
        Returns:
            str: Cleaned title suitable for filename
        """
        if not title:
            return "no-title"
        
        # Convert to lowercase
        title = title.lower()
        # Replace spaces with hyphens
        title = title.replace(' ', '-')
        # Remove special characters
        title = re.sub(r'[^a-z0-9\-]', '-', title)
        # Replace multiple hyphens with single hyphen
        title = re.sub(r'\-+', '-', title)
        # Truncate to 50 characters
        title = title[:50].rstrip('-')
        
        return title
    
    def clean_date_format(self, date_value):
        """
        Clean date format to remove time component and ensure DD/MM/YYYY format
        
        Args:
            date_value: Input date value
        
        Returns:
            str: Formatted date string
        """
        if pd.isna(date_value):
            return date_value
        
        try:
            # If it's already a datetime object
            if isinstance(date_value, datetime):
                return date_value.strftime('%d/%m/%Y')
            
            # If it's a string
            if isinstance(date_value, str):
                # Try to extract date part before any time indicators
                date_part = date_value
                if 'T' in date_part:
                    date_part = date_part.split('T')[0]
                elif ' ' in date_part:
                    date_part = date_part.split(' ')[0]
                
                # Try to parse and reformat
                try:
                    date_obj = pd.to_datetime(date_part)
                    return date_obj.strftime('%d/%m/%Y')
                except:
                    return date_part
            
            # Return as is if we can't process it
            return date_value
        
        except Exception as e:
            logger.warning(f"Error cleaning date format: {str(e)}")
            return date_value

    def extract_and_save_article(self, article):
        """
        Extract the article from the URL and save it to the appropriate directories
        """
        url = article.get('URL', '')
        is_paywall = article.get('Paywall', 'N') == 'Y'
        
        if not url:
            logger.warning(f"No URL for article with title: {article.get('Title')}")
            return False, None, None
        
        try:
            # Add random delay to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            # For paywalled articles, always use archive buttons URL
            if is_paywall:
                archive_url = f"https://www.archivebuttons.com/articles?article={url}"
                logger.info(f"Processing paywalled article via archive: {archive_url}")
                
                # Additional wait for archive buttons to load
                time.sleep(random.uniform(3, 5))
                
                # Use archive buttons URL for request
                current_url = archive_url
            else:
                # Use original URL for non-paywalled articles
                current_url = url
            
            # Make the request
            response = requests.get(current_url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"Failed to retrieve URL: HTTP {response.status_code}")
                return False, None, None
            
            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract journalist name
            journalist = article.get('Journalist')
            if not journalist or pd.isna(journalist) or str(journalist).strip() == '':
                journalist = self.extract_journalist(soup, url)
            
            # Extract article text
            article_text = self.extract_article_text(soup, url)
            
            if not article_text:
                logger.error(f"Could not extract article text from {current_url}")
                return False, journalist, None
            
            # Generate filename
            filename = self.generate_filename(article)
            
            # Format the date
            formatted_date = self.clean_date_format(article.get('Date', ''))
            
            # Create a header with metadata
            header = f"""Title: {article.get('Title', 'No Title')}
Date: {formatted_date}
Source: {article.get('Media', self.source)}
Topic: {article.get('Theme', self.topic)}
Category: {article.get('Category', '')}
Relevance: {article.get('Relevance', '')}
URL: {url}
Paywall: {'Y' if is_paywall else 'N'}
"""
            if journalist:
                header += f"Journalist: {journalist}\n"
        
            # Combine header and article text
            content = f"{header}\n\n{article_text}"
            
            # Save to main file by source
            source_path = f"{self.output_dir}/by_source/{self.source}/{filename}"
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save to topic directory
            topic_path = f"{self.output_dir}/by_topic/{self.topic}/{filename}"
            with open(topic_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save to relevance directory
            relevance = article.get('Relevance', 0)
            relevance_suffix = {
                5: 'Most_',
                4: 'Highly_',
                3: '',
                2: 'Somewhat_',
                1: 'Least_'
            }.get(relevance, '')
            
            relevance_path = f"{self.output_dir}/by_relevance/R{relevance}_{relevance_suffix}Relevant/{filename}"
            with open(relevance_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Successfully saved article: {filename}")
            return True, journalist, filename
            
        except Exception as e:
            logger.error(f"Error extracting article from {url}: {str(e)}", exc_info=True)
            return False, None, None
    
    def check_paywall(self, soup):
        """
        Detect if the article is paywalled        
        Args:
            soup (BeautifulSoup): Parsed HTML content        
        Returns:
            bool: True if article is paywalled, False otherwise
        """
        # Look for the specific div with "FOR SUBSCRIBERS" text
        paywall_div = soup.find('div', class_=lambda x: x and 'border-t-5' in x and 'border-solid' in x and 'uppercase' in x)
        
        if paywall_div:
            paywall_text = paywall_div.get_text(strip=True).upper()
            return paywall_text == 'FOR SUBSCRIBERS'
        
        return False    

    def process_excel_file(self):
        try:
            # Read the Excel file
            logger.info(f"Reading Excel file: {self.excel_file}")
            df = pd.read_excel(self.excel_file)
        
            # Add Paywall column if it doesn't exist
            if 'Paywall' not in df.columns:
                df['Paywall'] = 'N'
            
            # Clean up date format in the DataFrame
            if 'Date' in df.columns:
                logger.info("Cleaning date format in Excel file")
                df['Date'] = df['Date'].apply(self.clean_date_format)
            
            # Add filename column if it doesn't exist
            if 'Filename' not in df.columns:
                df['Filename'] = None
            
            # Create a copy to track updates
            updated_df = df.copy()
            
            # Process each article
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting articles"):
                # Convert row to dict for easier handling
                article = row.to_dict()
                
                # Skip if already processed and successful
                if not pd.isna(updated_df.at[index, 'Filename']):
                    logger.info(f"Skipping already processed article: {article.get('Title')}")
                    continue
                
                # Original URL
                original_url = article.get('URL', '')
                    
                try:
                    # Fetch the page to check paywall status
                    response = requests.get(original_url, headers=self.headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Check for "FOR SUBSCRIBERS" text
                    paywall_div = soup.find('div', class_=lambda x: x and 'border-t-5' in x and 'border-solid' in x and 'uppercase' in x)
                    is_paywall = (paywall_div and 'FOR SUBSCRIBERS' in paywall_div.get_text(strip=True).upper())
                    
                    # Update Paywall column
                    updated_df.at[index, 'Paywall'] = 'Y' if is_paywall else 'N'
                    
                    # Update the article dictionary with paywall status
                    article['Paywall'] = 'Y' if is_paywall else 'N'
                    
                    # Extract and save the article
                    success, journalist, filename = self.extract_and_save_article(article)
                    
                    # Update the DataFrame with filename
                    if success:
                        updated_df.at[index, 'Filename'] = filename
                        
                        # If Journalist column exists and is empty, update it with extracted journalist
                        if 'Journalist' in updated_df.columns and (
                            pd.isna(updated_df.at[index, 'Journalist']) or 
                            updated_df.at[index, 'Journalist'] is None or 
                            str(updated_df.at[index, 'Journalist']).strip() == ''
                        ):
                            if journalist:
                                updated_df.at[index, 'Journalist'] = journalist
            
                except Exception as e:
                    logger.error(f"Error processing article {original_url}: {e}")
                    continue
            
            # Save the updated Excel file
            output_excel = self.excel_file.replace('.xlsx', 'Updated.xlsx')
            logger.info(f"Saving updated Excel file: {output_excel}")
            
            # Convert dates to string format before saving
            if 'Date' in updated_df.columns:
                updated_df['Date'] = updated_df['Date'].apply(
                    lambda x: self.clean_date_format(x) if not pd.isna(x) else x
                )
            
            # Save the updated Excel file
            updated_df.to_excel(output_excel, index=False)
            logger.info(f"Successfully saved to {output_excel}")
            
            return updated_df
        
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise

    def run(self):
        """
        Run the entire extraction process
        Returns:
            bool: True if extraction was successful, False otherwise
        """
        logger.info(f"Starting extraction from {self.excel_file}")
        logger.info(f"Source: {self.source}, Topic: {self.topic}")
        
        try:
            updated_df = self.process_excel_file()
            
            # Print summary
            total = len(updated_df)
            extracted = len(updated_df[~updated_df['Filename'].isna()])
            
            logger.info(f"Extraction complete: {extracted}/{total} articles extracted")
            logger.info(f"Articles saved to {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            return False


# In[58]:


def main():
    """
    Main function to run the article extraction process
    """
    import sys
    
    input_file = "CNN_Organised_Crime_Articles_Processed.xlsx"
    
    try:
        # Create and run the extractor
        extractor = NewsArticleExtractor(input_file)
        success = extractor.run()
        
        # Exit with appropriate status code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


# In[59]:


if __name__ == "__main__":
    main()


# In[ ]:


pwd


# In[ ]:


ls

