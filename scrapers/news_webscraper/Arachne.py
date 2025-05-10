# --- START OF FILE Arachne.py ---

# Required Libraries: selenium pandas webdriver-manager openpyxl requests beautifulsoup4 lxml
# Installation: python -m pip install selenium pandas webdriver-manager openpyxl requests beautifulsoup4 lxml
# Usage: python Arachne.py

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import pandas as pd
import time
import logging
import threading
import queue
import os
import re
from datetime import datetime
from urllib.parse import urljoin, quote_plus, urlparse # Added urlparse
import webbrowser
import sys
import subprocess
import random

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

# --- Import the Weaver module ---
try:
    import Weaver
except ImportError:
    messagebox.showerror("Import Error", "Could not find the required 'Weaver.py' file.\n\nPlease ensure Weaver.py is in the same directory as Arachne.py.")
    sys.exit(1)

# --- Configuration Constants ---
OUTPUT_DIR = "scraped_articles_excel"
DEFAULT_DELAY = 1.5

# --- Site-Specific Configurations (Unchanged) ---
SITE_CONFIG = {
    "Straits Times": {
        "search_url_template": "https://www.straitstimes.com/search?searchkey={query}",
        "base_url": "https://www.straitstimes.com",
        "container_selectors": [".queryly_item_row", ".queryly_item_container", ".card-body", "div.card"],
        "title_selectors": [".queryly_item_title", "a > div[style*='margin-top']", ".card-title a", "h5.card-title a"],
        "url_selectors": ["a"],
        "date_selectors": [".queryly_item_description", ".queryly_item_meta_info", ".card-text small", ".timestamp"],
        "extract_date_from_description": True
    },
     "CNA": {
        "search_url_template": "https://www.channelnewsasia.com/search?q={query}&type%5B0%5D=article",
        "base_url": "https://www.channelnewsasia.com",
        "container_selectors": [".list-object", ".ais-InfiniteHits-item", ".search-results-list .item", "div.teaser"],
        "title_selectors": [".list-object__heading-link", ".list-object__heading > a", "h6 > a", "h3.teaser__title a"],
        "url_selectors": [],
        "date_selectors": [".list-object__datetime-duration", ".list-object__timestamp", ".hit-date", ".teaser__datetime"],
    },
    "BBC": {
        "search_url_template": "https://www.bbc.com/search?q={query}&page=1",
        "base_url": "https://www.bbc.com",
        "container_selectors": ["div[data-testid='newport-card']", "li.ssrcss-1kxn2ku-Promo", "div[data-result-type='article']", "article"],
        "title_selectors": ["h2[data-testid='card-headline']", "p.ssrcss-6arcww-PromoHeadline", "h3.ssrcss-1k6fmdl-PromoHeadline", "a[data-testid='internal-link'] p", "h2.sc-9d830f2a-3", "div[role='text'] > span"],
        "url_selectors": ["a[data-testid='internal-link']", "a.ssrcss-17behz7-PromoLink", "a.sc-8a623a54-0", "a"],
        "date_selectors": ["span[data-testid='card-metadata-lastUpdated']", "time.ssrcss-1if1g9v-MetadataTimestamp", "span.ssrcss-1if1g9v-MetadataTimestamp", "span.sc-ac6bc755-1", "time"],
        "split_or_keywords": True
    },
    "CNN": {
        "search_url_template": "https://edition.cnn.com/search?q={query}&from=0&size=20&sort=relevance&types=article",
        "base_url": "https://edition.cnn.com",
        "container_selectors": [".container__item--type-article", ".container__item", ".card", "article.search__result--article", "div.result-card"],
        "title_selectors": [".container__headline-text", ".card__headline a", "h3.search__result-headline a", "span[data-editable='headline']"],
        "url_selectors": [],
        "date_selectors": [".container__date", ".card__timestamp", ".timestamp", ".search__result-publish-date", "div.result-card__meta > span"],
    }
}
# --- Common Date Selectors (Unchanged) ---
COMMON_DATE_SELECTORS = [
    "time", "[itemprop='datePublished']", "[itemprop='dateCreated']",
    ".date", ".timestamp", ".byline__datetime", ".entry-date", ".published",
    ".post-date", ".article-date", ".dateline"
]

# --- Theme Definitions & Mapping ---
KEYWORD_THEMES = [
    ("Cybercrime", '"Cybersecurity" OR "Cybercrime" OR "Digital Fraud" OR "Scams"'),
    ("Forensics", '"Forensic Science" OR "Criminal Investigations"'),
    ("Misinformation", '"Misinformation" OR "Fake News" OR "Propaganda"'),
    ("Disinformation", '"Disinformation" OR "Psyops"'), # Simplified Disinfo
    ("Medical Fraud", '"Medical Fraud" OR "Malpractice"'),
    ("Organised Crime", '"Organised Crime" OR "Drug Trafficking"') # Combined OC/DT for search button
]
# Create a reverse map for easily finding the theme name from the exact keyword string
KEYWORD_TO_THEME_MAP = {kw_string: theme_name for theme_name, kw_string in KEYWORD_THEMES}

# --- Logging Setup (Unchanged) ---
log_filename = f"scraper_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log_queue = queue.Queue()

# --- Helper Functions (Selenium related - Unchanged) ---
# normalize_url, setup_webdriver, DATE_REGEX, extract_element_text, extract_element_href, generate_search_url
# scrape_search_page, scrape_article_page
def normalize_url(base_url, link):
    """Joins a relative link with a base URL, handling protocol-relative URLs and path joining."""
    if not link: return "N/A" # Return "N/A" for empty links
    try:
        link = link.strip() # Remove leading/trailing whitespace
        # Handle protocol-relative URLs (e.g., "//example.com/page")
        if link.startswith("//"):
            base_scheme = urlparse(base_url).scheme # Get scheme (http/https) from the base URL
            link = f"{base_scheme}:{link}" # Prepend the scheme
        # Use urljoin for robust handling of relative paths (e.g., ../, /page.html)
        return urljoin(base_url, link)
    except ValueError:
        # Log errors if urljoin fails (e.g., due to malformed base URL or link)
        logging.warning(f"Malformed URL fragment encountered during normalization: {link}")
        return "N/A"

def setup_webdriver():
    """Initializes and returns a Selenium WebDriver instance with configured options to avoid detection."""
    logging.info("Setting up Selenium WebDriver...")
    try:
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-running-insecure-content')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36')
        # chrome_options.add_argument('--headless=new')
        os.environ['WDM_LOG_LEVEL'] = '0'
        logging.getLogger('WDM').setLevel(logging.WARNING)
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(60)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        logging.info("Selenium WebDriver successfully initialized.")
        return driver
    except WebDriverException as e:
         err_msg = f"WebDriver setup failed: {e}"
         if "net::ERR_" in str(e): err_msg = f"WebDriver setup failed: Network error ({e}). Check internet connection, proxy settings, or firewall."
         elif "session not created" in str(e): err_msg = f"WebDriver setup failed: Session creation error ({e}). Check Chrome/ChromeDriver compatibility, ensure Chrome is installed correctly, or check for conflicting Chrome processes."
         elif "path to the driver" in str(e).lower(): err_msg = f"WebDriver setup failed: Driver executable issue ({e}). Ensure ChromeDriver is installed and accessible."
         logging.error(err_msg)
         log_queue.put((f"FATAL: {err_msg}", "FATAL"))
         raise
    except Exception as e:
         err_msg = f"An unexpected error occurred during WebDriver initialization: {e}"
         logging.error(err_msg, exc_info=True)
         log_queue.put((f"FATAL: {err_msg}", "FATAL"))
         raise

DATE_REGEX = re.compile(
    r"""
    (?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})
    |
    (?:\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+\d{4})
    """,
    re.IGNORECASE | re.VERBOSE
)

def extract_element_text(element, selectors, config=None):
    """Extracts text content using a prioritized list of CSS selectors within a given Selenium element."""
    if not selectors: return "N/A"
    for i, selector in enumerate(selectors):
        try:
            found = element.find_element(By.CSS_SELECTOR, selector)
            text = found.text.strip()
            dt_attr = found.get_attribute('datetime')
            is_special_date_field = config and config.get("extract_date_from_description", False) and selector == config.get("date_selectors", [""])[0]
            if is_special_date_field:
                 if text:
                     match = DATE_REGEX.search(text)
                     if match: return match.group(0).strip().replace(',', '')
                     else: continue
                 else: continue
            if text: return text
            if dt_attr: return dt_attr.strip()
        except NoSuchElementException: continue
        except Exception as e: logging.warning(f"Error extracting text with selector '{selector}': {e}"); continue
    return "N/A"

def extract_element_href(element, selectors, base_url):
    """Extracts a URL (href attribute) using a prioritized list of CSS selectors within a given Selenium element."""
    effective_selectors = selectors if selectors else ['a']
    for selector in effective_selectors:
        try:
            targets = element.find_elements(By.CSS_SELECTOR, selector)
            for target in targets:
                href = None
                if target.tag_name.lower() == 'a': href = target.get_attribute('href')
                else:
                    try: link_element = target.find_element(By.XPATH, ".//a[@href]"); href = link_element.get_attribute('href')
                    except NoSuchElementException: href = None
                if href:
                    normalized_url = normalize_url(base_url, href)
                    if normalized_url != "N/A" and urlparse(normalized_url).scheme in ['http', 'https']: return normalized_url
                    elif normalized_url != "N/A": logging.debug(f"Normalized URL skipped due to non-HTTP/S scheme: {normalized_url}")
        except NoSuchElementException: continue
        except Exception as e: logging.warning(f"Error extracting href with selector '{selector}': {e}"); continue
    try:
        all_links_in_container = element.find_elements(By.XPATH, ".//a[@href]")
        if all_links_in_container:
            href = all_links_in_container[0].get_attribute('href')
            if href:
                normalized_url = normalize_url(base_url, href)
                if normalized_url != "N/A" and urlparse(normalized_url).scheme in ['http', 'https']: return normalized_url
    except Exception as e: logging.warning(f"Fallback href extraction (finding any 'a' tag) failed: {e}")
    return "N/A"

def generate_search_url(site_name, keywords):
    """Generates the full search URL for a given site and keywords using the SITE_CONFIG."""
    config = SITE_CONFIG.get(site_name)
    if not config: logging.error(f"Configuration error: Settings not found for site '{site_name}'."); return None
    template = config.get('search_url_template')
    if not template: logging.error(f"Configuration error: Search URL template missing for site '{site_name}'."); return None
    try:
        if site_name == "BBC" and config.get("split_or_keywords", False) and " OR " in keywords: query = keywords
        else: query = quote_plus(keywords)
        search_url = template.format(query=query)
        return search_url
    except KeyError: logging.error(f"URL template error: The search URL template for '{site_name}' might be missing the '{{query}}' placeholder."); return None
    except Exception as e: logging.error(f"Error generating search URL for '{site_name}' with keywords '{keywords}': {e}"); return None

def scrape_search_page(driver, url, config):
    """Scrapes article titles, URLs, and initial date strings from a search results page."""
    page_results = []
    base_url = config['base_url']
    site_name = next((k for k, v in SITE_CONFIG.items() if v == config), "Unknown Site")
    log_queue.put((f"Navigating to search page ({site_name}): {url}", "INFO"))
    try:
        driver.get(url)
        primary_wait_selector = config['container_selectors'][0]
        used_wait_selector = primary_wait_selector
        try: WebDriverWait(driver, 25).until(EC.presence_of_element_located((By.CSS_SELECTOR, primary_wait_selector)))
        except TimeoutException:
             secondary_selectors = config['container_selectors'][1:]
             found_alternative_wait = False
             for alt_selector in secondary_selectors:
                  try: WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.CSS_SELECTOR, alt_selector))); used_wait_selector = alt_selector; found_alternative_wait = True; log_queue.put((f"Primary wait selector timed out, used alternative '{alt_selector}'.", "WARNING")); break
                  except TimeoutException: continue
             if not found_alternative_wait: log_queue.put((f"Timeout waiting for any specified result container element ({site_name}, Waited on: '{used_wait_selector}'). URL: {url}", "ERROR")); return []
        time.sleep(random.uniform(DEFAULT_DELAY * 0.5, DEFAULT_DELAY * 1.0))
        try: WebDriverWait(driver, 15).until(lambda d: d.execute_script('return document.readyState') == 'complete')
        except TimeoutException: log_queue.put((f"Page readyState check timed out, scraping anyway: {url}", "WARNING"))
        containers, used_container_selector = [], "N/A"
        for selector in config['container_selectors']:
            try:
                found_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if found_elements: containers = found_elements; used_container_selector = selector; log_queue.put((f"Found {len(containers)} result containers using selector: '{selector}'.", "INFO")); break
            except Exception as e: logging.warning(f"Error finding containers with selector '{selector}': {e}"); continue
        if not containers: log_queue.put((f"No result containers found ({site_name}): {url}", "WARNING")); return []
        log_queue.put((f"Processing {len(containers)} containers found with '{used_container_selector}'...", "INFO"))
        count_extracted = 0
        processed_urls_on_page = set()
        for i, container_element in enumerate(containers):
            try:
                title = extract_element_text(container_element, config['title_selectors'], config)
                url_selectors_to_use = config.get('url_selectors', []) or config['title_selectors']
                article_url = extract_element_href(container_element, url_selectors_to_use, base_url)
                article_date = extract_element_text(container_element, config['date_selectors'], config)
                is_valid_title = title != "N/A" and len(title) > 3
                is_valid_url = article_url != "N/A" and article_url.startswith('http')
                is_new_url = article_url not in processed_urls_on_page
                if is_valid_title and is_valid_url and is_new_url:
                    page_results.append({'Title': title, 'URL': article_url, 'Date': article_date})
                    processed_urls_on_page.add(article_url)
                    count_extracted += 1
                    logging.debug(f"Initial scrape OK [{i+1}/{len(containers)}]: T='{title[:50]}...'")
                elif is_valid_url and not is_new_url: logging.debug(f"Skipping duplicate URL on page: {article_url}")
                else:
                    container_text = container_element.text.strip()[:100].replace('\n', ' ')
                    if container_text: logging.warning(f"Skipped container [{i+1}/{len(containers)}] invalid data (T='{title}', U='{article_url}'). Text: '{container_text}...'. Selector: '{used_container_selector}'")
            except Exception as e: logging.warning(f"Error processing container {i+1} on {url}: {e}", exc_info=False)
        log_queue.put((f"Initial scrape of {url} complete. Extracted {count_extracted} unique articles.", "INFO"))
    except TimeoutException: log_queue.put((f"Timeout loading search results page: {url}", "ERROR"))
    except WebDriverException as e: log_queue.put((f"WebDriver error during search page scrape ({site_name}) {url}: {e}", "ERROR"))
    except Exception as e: log_queue.put((f"Critical error scraping search page {url}: {e}", "ERROR")); logging.exception(f"Traceback for critical error on search page {url}:")
    return page_results

def scrape_article_page(driver, url):
    """Performs a minimal scrape of a direct article URL to get Title, URL, and Date (for Mode 2 initial pass)."""
    log_queue.put((f"Mode 2: Initial scrape attempt for direct URL: {url}", "INFO"))
    title, article_date = "N/A", "N/A"
    try:
        driver.get(url)
        WebDriverWait(driver, 25).until(lambda d: d.execute_script('return document.readyState') == 'complete')
        time.sleep(random.uniform(0.8, 1.8))
        page_title = "N/A"
        try:
             WebDriverWait(driver, 7).until(EC.presence_of_element_located((By.TAG_NAME, 'title')))
             page_title_raw = driver.title.strip()
             if page_title_raw:
                 seps = ['|', ' - ', ' :: ']; matched = False
                 for sep in seps:
                     if sep in page_title_raw:
                         parts = page_title_raw.split(sep)
                         first_part = parts[0].strip()
                         page_title = first_part if len(first_part) > 10 else page_title_raw
                         matched = True
                         break
                 if not matched: page_title = page_title_raw
        except TimeoutException:
            log_queue.put((f"Title tag not found/timed out for {url}, trying H1.", "WARNING"))
            page_title = "N/A"
        except Exception as e_title:
             log_queue.put((f"Error getting <title> tag for {url}: {e_title}", "WARNING"))
             page_title = "N/A"

        if page_title == "N/A":
            try:
                h1_tags = driver.find_elements(By.TAG_NAME, 'h1')
                if h1_tags:
                    # --- CORRECTED H1 FALLBACK BLOCK (Indented) ---
                    for h1 in h1_tags:
                        h1_text = h1.text.strip()
                        if h1_text:
                            page_title = h1_text
                            log_queue.put((f"Initial scrape using H1 tag for title: '{page_title[:60]}...'", "INFO"))
                            break # Exit loop after finding the first non-empty H1
                    # --- END OF CORRECTION ---
            except Exception as e_h1:
                log_queue.put((f"Error during initial H1 fallback: {e_h1}", "WARNING"))

        title = page_title if page_title != "N/A" else "Title Not Found"
        if title == "Title Not Found": log_queue.put((f"Could not find Title or H1 for {url}", "WARNING"))
        extracted_date = "N/A"
        try:
            body_element = driver.find_element(By.TAG_NAME, 'body')
            extracted_date = extract_element_text(body_element, COMMON_DATE_SELECTORS, None)
            if extracted_date == "N/A":
                 meta_date_selectors = ['meta[property="article:published_time"]', 'meta[name="pubdate"]', 'meta[name="publishdate"]', 'meta[itemprop="datePublished"]', 'meta[name="cXenseParse:recs:publishtime"]']
                 for selector in meta_date_selectors:
                     try:
                         meta_tag = driver.find_element(By.CSS_SELECTOR, selector)
                         content = meta_tag.get_attribute('content')
                         if content:
                             extracted_date = content.strip()
                             log_queue.put((f"Found date in meta ({selector}): {extracted_date}", "DEBUG"))
                             break
                     except NoSuchElementException: continue
                     except Exception as e_meta: log_queue.put((f"Error checking meta date ({selector}): {e_meta}", "WARNING"))
        except NoSuchElementException: log_queue.put((f"Body tag not found for date extraction on {url}", "WARNING"))
        except Exception as e_date: log_queue.put((f"Error during initial date extraction for {url}: {e_date}", "WARNING"))
        article_date = extracted_date
        log_queue.put((f"Initial scrape for direct URL completed. T='{title[:60]}...', D='{article_date}'", "INFO"))
        return {'Title': title, 'URL': url, 'Date': article_date}
    except TimeoutException: log_queue.put((f"Timeout loading direct article page: {url}", "ERROR")); return {'Title': 'Timeout Loading Page', 'URL': url, 'Date': 'N/A'}
    except WebDriverException as e: log_queue.put((f"WebDriver error loading/processing direct URL {url}: {e}", "ERROR")); return {'Title': 'WebDriver Error on Load', 'URL': url, 'Date': 'N/A'}
    except Exception as e: log_queue.put((f"Unexpected error scraping direct URL {url}: {e}", "ERROR")); logging.exception(f"Traceback for unexpected error on direct URL {url}:"); return {'Title': 'Error During Initial Scrape', 'URL': url, 'Date': 'N/A'}

# --- Scraper Thread Class (Unchanged) ---
class ScraperThread(threading.Thread):
    """Thread responsible for running Selenium web scraping tasks."""
    def __init__(self, mode, site, keywords, urls, completion_callback):
        super().__init__()
        self.mode = mode
        self.site = site
        self.keywords = keywords
        self.urls_input = urls
        self.completion_callback = completion_callback
        self.driver = None
        self.results_holder = []
        self.daemon = True
        self.name = "ScraperThread"

    def run(self):
        log_queue.put((f"Scraper thread started. Mode: {self.mode}", "INFO"))
        self.results_holder = []
        try:
            self.driver = setup_webdriver()
            if not self.driver: self.completion_callback([]); return
            if self.mode == 1:
                config = SITE_CONFIG.get(self.site)
                if not config: log_queue.put((f"Configuration error: Settings not found for site '{self.site}'.", "ERROR")); self.completion_callback([]); return
                if self.site == "BBC" and config.get("split_or_keywords", False) and " OR " in self.keywords:
                    search_terms = [t.strip().strip('"') for t in self.keywords.split(" OR ") if t.strip()]
                    log_queue.put((f"BBC Mode: Splitting into {len(search_terms)} searches.", "INFO"))
                    combined_results, seen_urls = [], set()
                    for i, term in enumerate(search_terms):
                        log_queue.put((f"BBC searching term {i+1}/{len(search_terms)}: '{term}'", "INFO"))
                        term_search_url = generate_search_url(self.site, term)
                        if term_search_url:
                            results_for_term = scrape_search_page(self.driver, term_search_url, config)
                            added_count = 0
                            for r in results_for_term:
                                r_url = r.get('URL')
                                if r_url and r_url != "N/A" and r_url.startswith('http') and r_url not in seen_urls:
                                    combined_results.append(r); seen_urls.add(r_url); added_count += 1
                            log_queue.put((f"BBC term '{term}': Found {len(results_for_term)}, added {added_count} unique.", "INFO"))
                            time.sleep(max(1.0, DEFAULT_DELAY * 0.75))
                        else: log_queue.put((f"BBC: Skipping term '{term}', failed URL generation.", "WARNING"))
                    self.results_holder = combined_results
                else:
                    search_url = generate_search_url(self.site, self.keywords)
                    if search_url: self.results_holder.extend(scrape_search_page(self.driver, search_url, config))
                    else: log_queue.put(("Could not generate search URL.", "ERROR"))
            elif self.mode == 2:
                if not self.urls_input: log_queue.put(("No URLs provided for Mode 2.", "WARNING"))
                else:
                    num_urls = len(self.urls_input)
                    log_queue.put((f"Mode 2: Starting initial scrape for {num_urls} URLs.", "INFO"))
                    for i, url_in in enumerate(self.urls_input):
                        original_url = url_in.strip()
                        url_to_scrape = original_url
                        if not url_to_scrape: log_queue.put((f"Skipping empty line at index {i}.", "INFO")); continue
                        if not re.match(r'^[a-zA-Z]+://', url_to_scrape): url_to_scrape = 'https://' + url_to_scrape; log_queue.put((f"Prepended 'https://': {url_to_scrape}", "INFO"))
                        log_queue.put((f"Initial scrape URL {i+1}/{num_urls}: {url_to_scrape}", "INFO"))
                        try:
                            result = scrape_article_page(self.driver, url_to_scrape)
                            if isinstance(result, dict): result['OriginalInputURL'] = original_url; self.results_holder.append(result)
                            else: log_queue.put((f"Unexpected non-dict result from scrape_article_page for {url_to_scrape}. Creating placeholder.", "ERROR")); self.results_holder.append({'Title': "Internal Scrape Function Error", 'URL': url_to_scrape, 'Date': 'N/A', 'OriginalInputURL': original_url})
                            time.sleep(max(0.8, DEFAULT_DELAY * 0.6))
                        except Exception as e_direct_scrape:
                            log_queue.put((f"Error during initial scrape of {url_to_scrape}: {e_direct_scrape}. Creating placeholder.", "ERROR"))
                            self.results_holder.append({'Title': "Error During Initial Scrape", 'URL': url_to_scrape, 'Date': 'N/A', 'OriginalInputURL': original_url})
                            time.sleep(max(0.5, DEFAULT_DELAY * 0.5))
            self.completion_callback(self.results_holder)
        except WebDriverException as e_wd_thread: log_queue.put((f"Unhandled WebDriver error in ScraperThread: {e_wd_thread}", "FATAL")); logging.error(f"Unhandled WebDriver error in ScraperThread: {e_wd_thread}", exc_info=True); self.completion_callback([])
        except Exception as e_thread: log_queue.put((f"Unexpected critical error in ScraperThread: {e_thread}", "FATAL")); logging.exception("ScraperThread encountered unexpected critical error:"); self.completion_callback([])
        finally:
            if self.driver:
                try: log_queue.put(("Closing WebDriver instance...", "INFO")); self.driver.quit(); log_queue.put(("WebDriver instance closed.", "INFO"))
                except Exception as e_quit: log_queue.put((f"Error closing WebDriver: {e_quit}", "ERROR"))
            log_queue.put(("Scraper thread finished.", "INFO"))


# --- GUI Application Class ---
class ScraperApp:
    """Main class for the Tkinter GUI application."""

    def __init__(self, root):
        """Initialize the Scraper Application GUI."""
        self.root = root
        self.root.title("Arachne Scraper V3.3") # Updated version
        self.root.geometry("1100x800")

        # --- Style Configuration (Unchanged) ---
        self.style = ttk.Style()
        try:
            themes = self.style.theme_names(); theme = 'clam' if 'clam' in themes else ('vista' if 'vista' in themes else themes[0])
            self.style.theme_use(theme)
            logging.info(f"Using Tkinter theme: {theme}")
        except tk.TclError: logging.warning("Failed to set preferred Tkinter theme."); pass
        self.style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))
        self.style.configure("TButton", padding=6)
        self.style.configure("TRadiobutton", padding=4)
        self.style.configure("Suggestion.TButton", padding=(3, 3), font=("TkDefaultFont", 8))
        self.style.configure("Delete.TButton", foreground="red")
        self.style.configure("Metadata.TButton", foreground="blue")

        # --- Output Directory (Unchanged) ---
        if not os.path.exists(OUTPUT_DIR):
            try: os.makedirs(OUTPUT_DIR); logging.info(f"Created output directory: {os.path.abspath(OUTPUT_DIR)}")
            except OSError as e: logging.warning(f"Could not create dir '{OUTPUT_DIR}': {e}")

        # --- Application State Variables (Unchanged) ---
        self.mode_var = tk.IntVar(value=1)
        self.site_var = tk.StringVar(value="Straits Times")
        self.keywords_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Initializing application...")
        self.scraping_in_progress = False
        self.metadata_in_progress = False
        self.results_data = []
        self.processed_data = []
        self.next_result_iid = 0
        self.next_processed_iid = 0
        self.current_theme_context = "General" # Default theme context
        self.log_running = False

        # --- Build GUI Layout (Unchanged order) ---
        self.create_notebook()
        self.create_status_bar()
        self.create_mode_frame()
        self.create_config_frame()
        self.create_results_frame()
        self.create_processed_frame()
        self.create_log_frame()
        self.start_log_monitor()

        self.status_var.set("Ready. Select mode and configure settings.")
        logging.info("GUI initialized successfully.")

    def create_notebook(self):
        """Creates the main ttk.Notebook widget that holds the application tabs."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.config_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.processed_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Configuration")
        self.notebook.add(self.results_tab, text="Scraped Results")
        self.notebook.add(self.processed_tab, text="Processed Metadata")
        self.notebook.add(self.log_tab, text="Log")

    def create_mode_frame(self):
        """Creates the frame containing radio buttons for selecting the scraping mode."""
        mode_frame = ttk.LabelFrame(self.config_tab, text="Scraping Mode")
        mode_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Mode 1: Site Search (Uses Keywords)", variable=self.mode_var, value=1, command=self.update_mode_ui).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Mode 2: Direct URL List", variable=self.mode_var, value=2, command=self.update_mode_ui).pack(anchor=tk.W, padx=10, pady=5)

    def create_config_frame(self):
        """Creates the main configuration area displayed within the 'Configuration' tab."""
        self.config_frame = ttk.Frame(self.config_tab)
        self.config_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        # --- Mode 1 Frame ---
        self.site_frame = ttk.LabelFrame(self.config_frame, text="Site Selection & Keywords (Mode 1)")
        ttk.Label(self.site_frame, text="Select News Site:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        site_combo = ttk.Combobox(self.site_frame, textvariable=self.site_var, values=list(SITE_CONFIG.keys()), state="readonly", width=20)
        site_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ttk.Label(self.site_frame, text="Search Keywords:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        kw_entry = ttk.Entry(self.site_frame, textvariable=self.keywords_var, width=50)
        kw_entry.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        sugg_frame = ttk.LabelFrame(self.site_frame, text="Keyword Suggestions (Sets Keywords & Theme Context)")
        sugg_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky=tk.W+tk.E)
        r, c, max_c = 0, 0, 3
        for theme_name, keyword_string in KEYWORD_THEMES:
            btn = ttk.Button(sugg_frame, text=theme_name, command=lambda n=theme_name, k=keyword_string: self.set_keywords_and_theme(n, k), style="Suggestion.TButton")
            btn.grid(row=r, column=c, padx=3, pady=3, sticky=tk.W); c += 1; r += (c // max_c); c %= max_c
        self.site_frame.columnconfigure(1, weight=1)
        # --- Mode 2 Frame ---
        self.urls_frame = ttk.LabelFrame(self.config_frame, text="URL List (Mode 2 - One URL per line)")
        urls_inner = ttk.Frame(self.urls_frame); urls_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        urls_scroll = ttk.Scrollbar(urls_inner); urls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.urls_text = tk.Text(urls_inner, height=15, width=70, wrap=tk.WORD, yscrollcommand=urls_scroll.set, undo=True)
        self.urls_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        urls_scroll.config(command=self.urls_text.yview)
        ttk.Button(self.urls_frame, text="Insert Example URLs", command=self.insert_example_urls).pack(anchor=tk.W, padx=5, pady=5)
        # --- Action Buttons ---
        buttons_frame = ttk.Frame(self.config_frame); buttons_frame.pack(fill=tk.X, padx=5, pady=10, side=tk.BOTTOM)
        self.start_button = ttk.Button(buttons_frame, text="Start Scraping (Append Results)", command=self.start_scraping)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.update_mode_ui()

    def create_results_frame(self):
        """Creates the widgets and layout for the 'Scraped Results' tab."""
        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tree_frame = ttk.Frame(results_frame); tree_frame.pack(fill=tk.BOTH, expand=True)
        tree_y_scroll = ttk.Scrollbar(tree_frame); tree_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL); tree_x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        initial_cols = ("Media", "Theme", "Title", "URL", "Date") # Keep Theme here
        self.results_tree = ttk.Treeview(tree_frame, columns=initial_cols, show="headings", yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set, selectmode='extended')
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        tree_y_scroll.config(command=self.results_tree.yview); tree_x_scroll.config(command=self.results_tree.xview)
        self.results_tree.heading("Media", text="Media", command=lambda: self.sort_treeview_column(self.results_tree, "Media", False))
        self.results_tree.heading("Theme", text="Theme", command=lambda: self.sort_treeview_column(self.results_tree, "Theme", False)) # Add sorting for Theme
        self.results_tree.heading("Title", text="Title", command=lambda: self.sort_treeview_column(self.results_tree, "Title", False))
        self.results_tree.heading("URL", text="URL", command=lambda: self.sort_treeview_column(self.results_tree, "URL", False))
        self.results_tree.heading("Date", text="Date (Initial)", command=lambda: self.sort_treeview_column(self.results_tree, "Date", False))
        self.results_tree.column("Media", width=90, minwidth=70, stretch=False)
        self.results_tree.column("Theme", width=120, minwidth=90, stretch=False) # Give Theme column width
        self.results_tree.column("Title", width=350, minwidth=200, stretch=True)
        self.results_tree.column("URL", width=350, minwidth=200, stretch=True)
        self.results_tree.column("Date", width=110, minwidth=90, stretch=False, anchor=tk.CENTER)
        self.results_tree.bind("<Double-1>", self.on_result_double_click)
        res_buttons = ttk.Frame(results_frame); res_buttons.pack(fill=tk.X, pady=5)
        self.metadata_button = ttk.Button(res_buttons, text="Generate Metadata", command=self.start_metadata_generation, style="Metadata.TButton", state=tk.DISABLED)
        self.metadata_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(res_buttons, text="Copy Selected URL(s)", command=lambda: self.copy_selected_url(tree=self.results_tree)).pack(side=tk.LEFT, padx=5)
        ttk.Button(res_buttons, text="Delete Selected", command=self.delete_selected_result, style="Delete.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(res_buttons, text="Clear All Results", command=self.clear_all_results).pack(side=tk.LEFT, padx=5)
        self.result_count_var = tk.StringVar(value="Results: 0 articles")
        ttk.Label(res_buttons, textvariable=self.result_count_var).pack(side=tk.RIGHT, padx=10)

    def create_processed_frame(self):
        """Creates the widgets and layout for the 'Processed Metadata' tab."""
        processed_frame = ttk.Frame(self.processed_tab)
        processed_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        tree_frame = ttk.Frame(processed_frame); tree_frame.pack(fill=tk.BOTH, expand=True)
        tree_y_scroll = ttk.Scrollbar(tree_frame); tree_y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL); tree_x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        processed_cols = ('Media', 'Theme', 'Date', 'Title', 'Journalist', 'URL', 'Relevance', 'Category') # Keep Theme column here too
        self.processed_tree = ttk.Treeview(tree_frame, columns=processed_cols, show="headings", yscrollcommand=tree_y_scroll.set, xscrollcommand=tree_x_scroll.set, selectmode='extended')
        self.processed_tree.pack(fill=tk.BOTH, expand=True)
        tree_y_scroll.config(command=self.processed_tree.yview); tree_x_scroll.config(command=self.processed_tree.xview)
        self.processed_tree.heading("Media", text="Media", command=lambda: self.sort_treeview_column(self.processed_tree, "Media", False))
        self.processed_tree.heading("Theme", text="Theme", command=lambda: self.sort_treeview_column(self.processed_tree, "Theme", False)) # Add Theme sorting
        self.processed_tree.heading("Date", text="Date (Processed)", command=lambda: self.sort_treeview_column(self.processed_tree, "Date", False))
        self.processed_tree.heading("Title", text="Title", command=lambda: self.sort_treeview_column(self.processed_tree, "Title", False))
        self.processed_tree.heading("Journalist", text="Journalist", command=lambda: self.sort_treeview_column(self.processed_tree, "Journalist", False))
        self.processed_tree.heading("URL", text="URL", command=lambda: self.sort_treeview_column(self.processed_tree, "URL", False))
        self.processed_tree.heading("Relevance", text="Relevance", command=lambda: self.sort_treeview_column(self.processed_tree, "Relevance", True))
        self.processed_tree.heading("Category", text="Category", command=lambda: self.sort_treeview_column(self.processed_tree, "Category", False)) # Category derived by Weaver
        self.processed_tree.column("Media", width=90, minwidth=70, stretch=False)
        self.processed_tree.column("Theme", width=120, minwidth=90, stretch=False) # Give Theme column width
        self.processed_tree.column("Date", width=110, minwidth=90, stretch=False, anchor=tk.CENTER)
        self.processed_tree.column("Title", width=280, minwidth=150, stretch=True)
        self.processed_tree.column("Journalist", width=130, minwidth=80, stretch=False)
        self.processed_tree.column("URL", width=280, minwidth=150, stretch=True)
        self.processed_tree.column("Relevance", width=70, minwidth=50, stretch=False, anchor=tk.CENTER)
        self.processed_tree.column("Category", width=160, minwidth=100, stretch=False)
        self.processed_tree.bind("<Double-1>", self.on_processed_double_click)
        proc_buttons = ttk.Frame(processed_frame); proc_buttons.pack(fill=tk.X, pady=5)
        ttk.Button(proc_buttons, text="Export Processed to Excel", command=self.export_processed_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(proc_buttons, text="Copy Selected URL(s)", command=lambda: self.copy_selected_url(tree=self.processed_tree)).pack(side=tk.LEFT, padx=5)
        self.processed_count_var = tk.StringVar(value="Processed: 0 articles")
        ttk.Label(proc_buttons, textvariable=self.processed_count_var).pack(side=tk.RIGHT, padx=10)

    def create_log_frame(self):
        """Creates the widgets for the 'Log' tab."""
        log_frame = ttk.Frame(self.log_tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        log_scroll = ttk.Scrollbar(log_frame); log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_display = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD, yscrollcommand=log_scroll.set, font=("Consolas", 9), state=tk.DISABLED)
        self.log_display.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.log_display.yview)
        self.log_display.tag_config("error", foreground="red", font=("Consolas", 9, "bold"))
        self.log_display.tag_config("warning", foreground="#E69138")
        self.log_display.tag_config("fatal", foreground="white", background="red", font=("Consolas", 9, "bold"))
        self.log_display.tag_config("info", foreground="black")
        self.log_display.tag_config("debug", foreground="grey")
        ttk.Button(log_frame, text="Clear Log Display", command=self.clear_log).pack(anchor=tk.W, padx=5, pady=5)

    def create_status_bar(self):
        """Creates the status bar label at the very bottom of the main window."""
        if not hasattr(self, 'status_var'): self.status_var = tk.StringVar(value="Error: Status bar init failed.")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

    # --- GUI Control and Logic Methods ---

    def update_mode_ui(self):
        """Shows or hides the Mode 1/Mode 2 configuration frames based on radio button selection."""
        mode = self.mode_var.get()
        status_exists = hasattr(self, 'status_var')
        if mode == 1:
            self.site_frame.pack(fill=tk.X, padx=5, pady=5, expand=False)
            self.urls_frame.pack_forget()
            if status_exists: self.status_var.set("Mode 1: Select site and enter keywords for searching.")
            # Try to update theme context if keywords match
            keywords = self.keywords_var.get().strip()
            # self.current_theme_context = KEYWORD_TO_THEME_MAP.get(keywords, "User Defined") # Default if no exact match
            self.current_theme_context = KEYWORD_TO_THEME_MAP.get(keywords, keywords) # Default if no exact match
        else: # Mode 2
            self.site_frame.pack_forget()
            self.urls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            if status_exists: self.status_var.set("Mode 2: Enter or paste URLs (one per line).")
            self.current_theme_context = "Direct URL List"
        log_queue.put((f"Switched to Mode {mode}. Current Theme context set to: '{self.current_theme_context}'", "INFO"))

    def set_keywords_and_theme(self, theme_name, keyword_string):
        """Sets the keywords entry field and theme context when a suggestion button is clicked."""
        self.keywords_var.set(keyword_string)
        self.current_theme_context = theme_name # Set context directly from button
        log_queue.put((f"Set keywords via suggestion: '{keyword_string}' (Theme: '{theme_name}')", "INFO"))
        self.status_var.set(f"Keywords set for theme '{theme_name}'. Ready for Mode 1 scrape.")

    def insert_example_urls(self):
        """Inserts a predefined list of example URLs into the Mode 2 text area, confirming overwrite."""
        examples = [
            "https://www.straitstimes.com/singapore/courts-crime/man-jailed-for-cheating-carousell-buyers-over-concert-tickets",
            "https://www.channelnewsasia.com/singapore/former-lawyer-vasant-nathalal-paid-prosecutor-advance-case-information-3950636",
            "https://www.bbc.com/news/world-latin-america-67675318",
            "https://edition.cnn.com/2023/12/11/politics/us-strikes-drone-site-yemen/index.html",
            "www.no-scheme-example.com",
            "https://httpbin.org/delay/4",
            "https://www.bbc.com/future/article/20220518-how-italy-and-chile-foiled-an-1m-international-smugglers-cactus-heist",
            "https://www.straitstimes.com/asia/australianz/australia-calls-for-global-action-to-fight-online-misinformation",
            "invalid-url-format",
        ]
        if self.urls_text.get(1.0, tk.END).strip():
            if not messagebox.askyesno("Confirm Replace", "Replace current URLs with examples?"): return
        self.urls_text.delete(1.0, tk.END)
        self.urls_text.insert(tk.END, "\n".join(examples))
        log_queue.put(("Inserted example URLs into Mode 2 text area.", "INFO"))

    def start_log_monitor(self):
        """Starts the periodic checking of the log queue."""
        if not self.log_running:
            self.log_running = True
            self.root.after(100, self.process_log_queue)
            logging.info("Log monitor started.")

    def stop_log_monitor(self):
        """Stops the periodic checking of the log queue."""
        self.log_running = False
        logging.info("Log monitor stopped.")

    def process_log_queue(self):
        """Periodically checks the log queue and updates the GUI log display."""
        try:
            while True:
                message, level = log_queue.get_nowait()
                self.log_display.config(state=tk.NORMAL)
                tag = level.lower()
                # --- FIX: Use tag_names() to check for existence ---
                if tag not in self.log_display.tag_names(): # Correct method
                    tag = 'info'
                # --- END FIX ---
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                log_entry = f"[{ts}] {message}\n"
                self.log_display.insert(tk.END, log_entry, tag)
                self.log_display.see(tk.END)
                self.log_display.config(state=tk.DISABLED)
                if level in ["ERROR", "WARNING", "FATAL", "INFO"]:
                    status_msg = f"{level}: {message}"
                    max_len = 120
                    truncated_msg = status_msg[:max_len] + ('...' if len(status_msg) > max_len else '')
                    if hasattr(self, 'status_var') and self.status_var:
                        self.status_var.set(truncated_msg)
                log_queue.task_done()
        except queue.Empty: pass
        except Exception as e:
             print(f"CRITICAL ERROR updating log display: {e}")
             logging.error(f"Error processing log queue or updating GUI log display: {e}", exc_info=True)
        if self.log_running:
            self.root.after(100, self.process_log_queue)

    def start_scraping(self):
        """Validates inputs and initiates the Selenium scraping process."""
        if self.scraping_in_progress: messagebox.showinfo("In Progress", "Scraping task already running."); return
        if self.metadata_in_progress: messagebox.showinfo("In Progress", "Metadata generation running. Please wait."); return

        mode = self.mode_var.get()
        site, keywords, urls = None, None, []

        if mode == 1:
            site = self.site_var.get()
            keywords: str = self.keywords_var.get().strip()
            if not keywords: messagebox.showerror("Input Error", "Keywords required for Mode 1."); return

            # --- Infer Theme Context from Keywords ---
            inferred_theme = KEYWORD_TO_THEME_MAP.get(keywords) # Check if exact keyword string matches a theme
            if inferred_theme:
                 # If keywords match a button's keywords, use that theme name
                 self.current_theme_context = inferred_theme
                 log_msg = f"Starting Mode 1: Site='{site}', Keywords='{keywords}', Inferred Theme='{self.current_theme_context}'"
            else:
                 # If keywords don't match a predefined theme, set context accordingly
                 ## self.current_theme_context = "Manual Keywords"
                 self.current_theme_context = keywords
                 log_msg = f"Starting Mode 1: Site='{site}', Keywords='{keywords}', Theme='{self.current_theme_context}'"
            # --- End Theme Inference ---

            status_msg = f"Starting scrape on '{site}' for keywords '{keywords[:30]}...'..."

        elif mode == 2:
            urls_text = self.urls_text.get(1.0, tk.END).strip()
            if not urls_text: messagebox.showerror("Input Error", "URL list required for Mode 2."); return
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            if not urls: messagebox.showerror("Input Error", "No valid URLs found in list."); return
            self.current_theme_context = "Direct URL List" # Set specific context
            log_msg = f"Starting Mode 2: {len(urls)} URLs. Theme='{self.current_theme_context}'"
            status_msg = f"Starting scrape for {len(urls)} direct URLs..."

        log_queue.put((log_msg, "INFO"))
        self.status_var.set(status_msg)
        self.scraping_in_progress = True
        self.start_button.config(state=tk.DISABLED)
        self.metadata_button.config(state=tk.DISABLED)
        self.notebook.select(self.log_tab)

        scraper_thread = ScraperThread(mode, site, keywords, urls, self.on_scraping_complete)
        scraper_thread.start()

    def on_scraping_complete(self, initial_results):
        """Callback executed in GUI thread after ScraperThread finishes."""
        self.root.after(0, self._update_gui_post_scrape, initial_results)

    def _update_gui_post_scrape(self, initial_results):
        """Updates the 'Scraped Results' tab after initial scrape."""
        if initial_results is None: initial_results = []
        added_count, skipped_duplicates_count = 0, 0
        existing_urls = {item.get('URL') for item in self.results_data if isinstance(item, dict)}
        try: existing_urls.update(self.results_tree.item(iid, 'values')[3] for iid in self.results_tree.get_children())
        except (IndexError, tk.TclError) as e: logging.warning(f"Could not get existing URLs from results tree: {e}")

        # Use the theme context that was set when the scrape started
        media_source = self.site_var.get() if self.mode_var.get() == 1 else "Direct URL"
        theme_context = self.current_theme_context # Use the accurate context

        for result in initial_results:
            if not isinstance(result, dict): continue
            title = result.get('Title', 'N/A')
            url = result.get('URL', 'N/A')
            date = result.get('Date', 'N/A')
            is_valid_url = url != "N/A" and url.startswith('http')
            if is_valid_url:
                if url not in existing_urls:
                    result['Media'] = media_source
                    result['Theme'] = theme_context # Assign the determined theme
                    self.results_data.append(result)
                    existing_urls.add(url)
                    try:
                        iid = f"R{self.next_result_iid}"
                        values_tuple = (result['Media'], result['Theme'], title, url, date) # Match initial_cols order
                        self.results_tree.insert("", tk.END, iid=iid, values=values_tuple)
                        self.next_result_iid += 1; added_count += 1
                    except tk.TclError as e_tree:
                         log_queue.put((f"GUI Error adding to Results Tree: {e_tree}. Data: {result}", "ERROR"))
                         if self.results_data and self.results_data[-1]['URL'] == url: self.results_data.pop()
                         existing_urls.discard(url)
                else: skipped_duplicates_count += 1; logging.debug(f"Skipping duplicate URL: {url}")
            else: logging.warning(f"Skipping result - invalid URL: T='{title[:50]}...', U='{url}'")

        self.scraping_in_progress = False
        self.start_button.config(state=tk.NORMAL)
        self.metadata_button.config(state=tk.NORMAL if self.results_data else tk.DISABLED)
        total_count = len(self.results_data)
        self.result_count_var.set(f"Results: {total_count} articles")
        status_msg = f"Initial scrape complete. Added {added_count} unique articles."
        if skipped_duplicates_count > 0: status_msg += f" Skipped {skipped_duplicates_count} duplicates."
        status_msg += f" Total articles: {total_count}."
        self.status_var.set(status_msg)
        if added_count > 0: self.notebook.select(self.results_tab)
        log_queue.put((status_msg, "INFO"))

    # --- Metadata Generation Methods (Unchanged logic, uses context from results_data) ---
    def start_metadata_generation(self):
        """Initiates the process to fetch detailed metadata using Weaver."""
        if self.metadata_in_progress: messagebox.showinfo("In Progress", "Metadata generation already running."); return
        if self.scraping_in_progress: messagebox.showinfo("In Progress", "Scraping is running. Please wait."); return
        if not self.results_data: messagebox.showinfo("No Data", "No scraped results to process."); return

        selected_iids = self.results_tree.selection()
        results_to_process = []
        log_prefix = ""
        if selected_iids:
             num_selected = len(selected_iids)
             if not messagebox.askyesno("Confirm Selection", f"Generate metadata for the {num_selected} selected article(s)?"): return
             log_prefix = f"Starting metadata generation for {num_selected} selected articles..."
             try: selected_urls = {self.results_tree.item(iid, 'values')[3] for iid in selected_iids}
             except (IndexError, tk.TclError) as e: messagebox.showerror("Error Reading Selection", f"Could not get URL data: {e}"); return
             results_to_process = [item for item in self.results_data if item.get('URL') in selected_urls]
        else:
             num_all = len(self.results_data)
             if not messagebox.askyesno("Confirm Process All", f"No articles selected. Generate metadata for all {num_all} articles?"): return
             log_prefix = f"Starting metadata generation for all {num_all} articles..."
             results_to_process = list(self.results_data)

        if not results_to_process: messagebox.showinfo("No Data Selected", "No articles found for processing."); return

        log_queue.put((log_prefix, "INFO"))
        self.status_var.set(log_prefix)
        self.metadata_in_progress = True
        self.metadata_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.notebook.select(self.log_tab)

        meta_thread = threading.Thread(target=self._metadata_worker, args=(results_to_process,), daemon=True, name="MetadataThread")
        meta_thread.start()

    def _metadata_worker(self, articles_to_process):
        """Worker function calling Weaver.process_article_url for each article."""
        processed_results_batch = []
        total_in_batch = len(articles_to_process)
        log_queue.put((f"Metadata worker started for {total_in_batch} articles.", "INFO"))
        for i, item_data in enumerate(articles_to_process):
            url = item_data.get('URL')
            title = item_data.get('Title', 'N/A')
            theme = item_data.get('Theme', 'Unknown') # Get theme from initial data
            media = item_data.get('Media', 'Unknown')
            initial_date = item_data.get('Date', 'N/A')
            if not url or not url.startswith('http'):
                log_queue.put((f"[{i+1}/{total_in_batch}] Skipping invalid URL: '{url}'", "WARNING"))
                processed_results_batch.append({'Media': media, 'Theme': theme, 'Date': initial_date, 'Title': title, 'Journalist': 'N/A', 'URL': url or 'Invalid', 'Relevance': 1, 'Category': 'Invalid Input URL'})
                continue
            log_queue.put((f"[{i+1}/{total_in_batch}] Requesting Weaver metadata for: {url}", "INFO"))
            try:
                weaver_data = Weaver.process_article_url(url, title)
                if isinstance(weaver_data, dict):
                    processed_date = weaver_data.get('processed_date', 'N/A')
                    combined_data = {
                        'Media': media, 'Theme': theme, # Carry over theme
                        'Date': processed_date if processed_date != 'N/A' else initial_date,
                        'Title': title, 'Journalist': weaver_data.get('journalist', 'N/A'),
                        'URL': url, 'Relevance': weaver_data.get('relevance_score', 1),
                        'Category': weaver_data.get('category', 'Error')
                    }
                    processed_results_batch.append(combined_data)
                    log_queue.put((f"[{i+1}/{total_in_batch}] Metadata received: Cat='{combined_data['Category']}', Rel={combined_data['Relevance']}", "DEBUG"))
                else:
                    log_queue.put((f"[{i+1}/{total_in_batch}] Weaver returned unexpected data for {url}.", "ERROR"))
                    processed_results_batch.append({'Media': media, 'Theme': theme, 'Date': initial_date, 'Title': title, 'Journalist': 'N/A', 'URL': url, 'Relevance': 1, 'Category': 'Weaver Module Error'})
            except Exception as e_weaver_call:
                log_queue.put((f"[{i+1}/{total_in_batch}] CRITICAL Error calling Weaver for {url}: {e_weaver_call}", "ERROR"))
                logging.exception(f"Weaver call failed for URL: {url}")
                processed_results_batch.append({'Media': media, 'Theme': theme, 'Date': initial_date, 'Title': title, 'Journalist': 'N/A', 'URL': url, 'Relevance': 1, 'Category': 'Metadata Fetch Error'})
            time.sleep(random.uniform(0.5, 1.0))
        log_queue.put((f"Metadata worker finished batch of {total_in_batch} articles.", "INFO"))
        self.root.after(0, self.on_metadata_generation_complete, processed_results_batch)

    def on_metadata_generation_complete(self, processed_results_batch):
        """Callback in GUI thread after metadata worker finishes."""
        log_queue.put(("Metadata generation complete. Updating view.", "INFO"))
        self.metadata_in_progress = False
        self.start_button.config(state=tk.NORMAL)
        self.metadata_button.config(state=tk.NORMAL if self.results_data else tk.DISABLED)
        if processed_results_batch is None: processed_results_batch = []

        existing_processed_urls_in_tree = {self.processed_tree.item(iid, 'values')[5] for iid in self.processed_tree.get_children()} # URL index 5
        added_to_tree_count = 0
        added_or_updated_in_data = 0

        for result_dict in processed_results_batch:
             if not isinstance(result_dict, dict): continue
             url = result_dict.get('URL')
             if not url: continue
             self.processed_data.append(result_dict) # Append to internal list
             added_or_updated_in_data += 1
             if url not in existing_processed_urls_in_tree:
                 try:
                     values = ( # Match processed_cols order
                         result_dict.get('Media', 'N/A'), result_dict.get('Theme', 'N/A'), result_dict.get('Date', 'N/A'),
                         result_dict.get('Title', 'N/A'), result_dict.get('Journalist', 'N/A'), result_dict.get('URL', 'N/A'),
                         result_dict.get('Relevance', 'N/A'), result_dict.get('Category', 'N/A')
                     )
                     iid = f"P{self.next_processed_iid}"
                     self.processed_tree.insert("", tk.END, iid=iid, values=values)
                     self.next_processed_iid += 1
                     existing_processed_urls_in_tree.add(url)
                     added_to_tree_count +=1
                 except tk.TclError as e: log_queue.put((f"GUI Error adding processed item to Treeview: {e}", "ERROR"))
                 except Exception as e: log_queue.put((f"Unexpected error adding processed item to tree: {e}", "ERROR"))
             else: log_queue.put((f"Skipping add to processed tree (URL exists): {url}", "DEBUG"))

        total_processed_in_view = len(self.processed_tree.get_children())
        self.processed_count_var.set(f"Processed: {total_processed_in_view} articles")
        num_processed_in_batch = len(processed_results_batch)
        status_msg = f"Metadata complete for {num_processed_in_batch} articles. Added {added_to_tree_count} new items to view."
        self.status_var.set(status_msg)
        log_queue.put((f"Displayed {added_to_tree_count} new processed articles. Total view: {total_processed_in_view}. Total memory: {len(self.processed_data)}.", "INFO"))
        self.notebook.select(self.processed_tab)

    # --- Other GUI Methods (Unchanged: delete, double-click, open URL, copy URL, export, open file, clear all, clear log, sort) ---
    def delete_selected_result(self):
        selected_iids = self.results_tree.selection()
        if not selected_iids: messagebox.showinfo("Delete", "No articles selected to delete."); return
        num_selected = len(selected_iids)
        if not messagebox.askyesno("Confirm Delete", f"Delete {num_selected} selected item(s) from 'Scraped Results'?\nCannot be undone."): return
        urls_to_delete, iids_to_delete_from_tree, errors = set(), [], 0
        for iid in selected_iids:
            try: url = self.results_tree.item(iid, 'values')[3]; urls_to_delete.add(url); iids_to_delete_from_tree.append(iid)
            except (IndexError, tk.TclError): errors += 1
        original_data_count = len(self.results_data)
        self.results_data = [item for item in self.results_data if not (isinstance(item, dict) and item.get('URL') in urls_to_delete)]
        data_deleted_count = original_data_count - len(self.results_data)
        tree_deleted_count = 0
        for iid in iids_to_delete_from_tree:
            try: self.results_tree.delete(iid); tree_deleted_count += 1
            except tk.TclError: errors += 1
        new_count = len(self.results_data); self.result_count_var.set(f"Results: {new_count} articles")
        status = f"Deleted {tree_deleted_count} item(s)."
        if data_deleted_count != tree_deleted_count: status += f" (Data removed: {data_deleted_count})."; log_queue.put("Warning: Data/Tree delete count mismatch.", "WARNING")
        if errors > 0: status += f" Errors: {errors}."
        status += f" Total: {new_count}."
        self.status_var.set(status); log_queue.put((status, "INFO"))
        self.metadata_button.config(state=tk.DISABLED if not self.results_data else tk.NORMAL)

    def on_result_double_click(self, event):
        selected_item_iid = self.results_tree.focus()
        if not selected_item_iid: return
        try: url = self.results_tree.item(selected_item_iid, 'values')[3]; self.open_url_in_browser(url)
        except IndexError: log_queue.put((f"Error accessing URL for initial result: {selected_item_iid}", "ERROR"))
        except Exception as e: log_queue.put((f"Error on initial result double-click: {e}", "ERROR"))

    def on_processed_double_click(self, event):
        selected_item_iid = self.processed_tree.focus()
        if not selected_item_iid: return
        try: url = self.processed_tree.item(selected_item_iid, 'values')[5]; self.open_url_in_browser(url)
        except IndexError: log_queue.put((f"Error accessing URL for processed result: {selected_item_iid}", "ERROR"))
        except Exception as e: log_queue.put((f"Error on processed result double-click: {e}", "ERROR"))

    def open_url_in_browser(self, url):
        if url and url != "N/A" and url.startswith(('http://', 'https://')):
            try: log_queue.put((f"Opening URL: {url}", "INFO")); webbrowser.open(url, new=2)
            except Exception as e: err_msg = f"Failed to open URL: {e}"; log_queue.put((err_msg, "ERROR")); messagebox.showerror("Browser Error", f"Could not open URL:\n{url}\n\nError: {e}")
        elif url and url != "N/A": warn_msg = f"URL '{url}' invalid (missing prefix?)."; log_queue.put((warn_msg, "WARNING")); messagebox.showwarning("Invalid URL", warn_msg)
        else: log_queue.put(("No URL to open.", "WARNING")); messagebox.showinfo("No URL", "No valid URL available.")

    def copy_selected_url(self, tree=None):
        target_tree = tree if tree else self.results_tree
        selected_iids = target_tree.selection()
        if not selected_iids: messagebox.showinfo("Copy URL", "No items selected."); return
        urls_to_copy, errors = [], 0
        url_col_index = 5 if target_tree == self.processed_tree else 3
        for iid in selected_iids:
            try:
                url = target_tree.item(iid, 'values')[url_col_index]
                if url and url != "N/A" and url.startswith('http'): urls_to_copy.append(url)
                else: log_queue.put((f"Skip copy {iid}: Invalid URL ('{url}').", "WARNING"))
            except IndexError: errors += 1; log_queue.put((f"Copy URL error accessing data {iid}", "ERROR"))
            except Exception as e: errors += 1; log_queue.put((f"Copy URL error {iid}: {e}", "ERROR"))
        if not urls_to_copy: messagebox.showinfo("Copy URL", "No valid URLs in selection."); return
        try:
            cb_text = "\n".join(urls_to_copy); self.root.clipboard_clear(); self.root.clipboard_append(cb_text)
            num_copied = len(urls_to_copy); status = f"Copied {num_copied} URL(s)."
            if errors > 0: status += f" ({errors} error(s))."
            self.status_var.set(status); log_queue.put((status, "INFO"))
        except tk.TclError as e_clip: err = f"Clipboard error: {e_clip}."; log_queue.put((err, "ERROR")); messagebox.showerror("Clipboard Error", err)
        except Exception as e_copy: err = f"Copy error: {e_copy}"; log_queue.put((err, "ERROR")); messagebox.showerror("Copy Error", err)

    def export_processed_to_excel(self):
        if not self.processed_data: messagebox.showinfo("Export Error", "No processed data to export."); return
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        default_filename = f"Arachne_Processed_Metadata_{timestamp}.xlsx"
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")], initialdir=OUTPUT_DIR, initialfile=default_filename, title="Save Processed Metadata As...")
        if not filepath: self.status_var.set("Export cancelled."); return
        log_queue.put((f"Exporting {len(self.processed_data)} items to: {filepath}", "INFO"))
        self.status_var.set("Exporting to Excel..."); self.root.update_idletasks()
        try:
            export_cols = ['Media', 'Theme', 'Date', 'Title', 'Journalist', 'URL', 'Relevance', 'Category']
            df = pd.DataFrame(self.processed_data, columns=export_cols)
            df.to_excel(filepath, index=False, engine='openpyxl')
            success_msg = f"Exported {len(df)} articles to: {os.path.basename(filepath)}"
            log_queue.put((success_msg, "INFO")); self.status_var.set(success_msg)
            if messagebox.askyesno("Export Complete", f"Exported successfully to:\n{filepath}\n\nOpen file?"): self.open_file(filepath)
        except ImportError: err = "Export Error: 'openpyxl' library required. Install: pip install openpyxl"; log_queue.put((err, "ERROR")); messagebox.showerror("Export Error", err); self.status_var.set("Export failed: Missing library.")
        except PermissionError: err = f"Export Error: Permission denied writing to:\n{filepath}\nIs the file open?"; log_queue.put((err, "ERROR")); messagebox.showerror("Export Error", err); self.status_var.set("Export failed: Permission denied.")
        except Exception as e: err = f"Export Error: {e}"; log_queue.put((err, "ERROR")); logging.exception("Excel export error:"); messagebox.showerror("Export Error", err); self.status_var.set("Export failed: Unexpected error.")

    def open_file(self, filepath):
        """Attempts to open the specified file using the system's default application."""
        try: # Start of the try block
            log_queue.put((f"Attempting to open file using system default application: {filepath}", "INFO"))
            # Use platform-specific commands to open the file
            if sys.platform == "win32":
                os.startfile(filepath) # Preferred method on Windows
            elif sys.platform == "darwin":
                subprocess.call(["open", filepath]) # Standard command on macOS
            else: # Assume Linux/Unix-like
                subprocess.call(["xdg-open", filepath]) # Standard command on Linux (requires xdg-utils)
        # --- Correctly Indented Exception Handling ---
        except FileNotFoundError:
            # Error if the specified file does not exist OR if the command (startfile/open/xdg-open) isn't found
            err_msg = f"Cannot open file: Command or file not found.\nEnsure the file exists and the necessary system command is available.\n\nPath: {filepath}"
            log_queue.put((err_msg, "ERROR"))
            messagebox.showerror("File/Command Not Found Error", err_msg)
        except OSError as e:
             # Error if the system command fails (e.g., no associated application)
             err_msg = f"Could not open file: The operating system failed to open the file.\nThis might mean no default application is associated with this file type, or the necessary system command failed.\n\n(OS Error: {e})"
             log_queue.put((err_msg, "ERROR"))
             messagebox.showwarning("File Open Error", err_msg + f"\n\nPlease try opening the file manually:\n{filepath}")
        except Exception as e:
            # Catch any other unexpected errors during the file opening attempt
            err_msg = f"An unexpected error occurred while trying to open the file: {e}"
            log_queue.put((err_msg, "WARNING")) # Warning as the file might exist but couldn't be opened
            messagebox.showwarning("File Open Error", err_msg + f"\n\nPlease try opening the file manually:\n{filepath}")
        # --- End of Exception Handling ---

    def clear_all_results(self):
        if not self.results_data and not self.processed_data: self.status_var.set("No results to clear."); return
        if messagebox.askyesno("Confirm Clear All", "Clear ALL scraped and processed data?\nCannot be undone."):
            self.results_data = []; self.processed_data = []
            try:
                for item in self.results_tree.get_children(): self.results_tree.delete(item)
                for item in self.processed_tree.get_children(): self.processed_tree.delete(item)
            except tk.TclError as e: log_queue.put((f"Error clearing treeviews: {e}", "WARNING"))
            self.result_count_var.set("Results: 0 articles")
            self.processed_count_var.set("Processed: 0 articles")
            self.status_var.set("All results cleared.")
            self.next_result_iid = 0; self.next_processed_iid = 0
            self.metadata_button.config(state=tk.DISABLED)
            log_queue.put(("All results cleared by user.", "INFO"))

    def clear_log(self):
        if messagebox.askyesno("Confirm Clear Log", "Clear the log display?"):
            try:
                self.log_display.config(state=tk.NORMAL); self.log_display.delete(1.0, tk.END)
                ts = datetime.now().strftime("%H:%M:%S"); log_entry = f"[{ts}] Log display cleared.\n"
                self.log_display.insert(tk.END, log_entry, "info"); self.log_display.config(state=tk.DISABLED)
            except Exception as e:
                print(f"Error clearing log display: {e}")
                try:
                    self.log_display.config(state=tk.DISABLED)
                except:
                    pass

    def sort_treeview_column(self, tree, col, reverse):
        try:
            data = [(str(tree.set(item, col) or ''), item) for item in tree.get_children('')]
            def sort_key_func(item_tuple):
                value_str = item_tuple[0].strip()
                if col == 'Relevance':
                    try: return int(value_str)
                    except (ValueError, TypeError): return -9999
                elif col == 'Date':
                    formats_to_try = ['%d/%m/%Y', '%b %d %Y', '%d %b %Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%b-%Y', '%Y/%m/%d']
                    date_part = re.sub(r'\s+\d{1,2}:\d{2}(:\d{2})?(\s*(AM|PM|GMT|[+-]\d{2}:?\d{2}|[A-Z]{3,}))?$', '', value_str, flags=re.IGNORECASE).strip().replace(',', '')
                    for fmt in formats_to_try:
                        try: return datetime.strptime(date_part, fmt)
                        except (ValueError, TypeError): continue
                    return datetime.min
                else: return value_str.lower()
            data.sort(key=sort_key_func, reverse=reverse)
            for index, (val, item) in enumerate(data): tree.move(item, '', index)
            tree.heading(col, command=lambda t=tree, c=col, r=not reverse: self.sort_treeview_column(t, c, r))
        except Exception as e_sort: log_queue.put((f"Error sorting tree column '{col}': {e_sort}", "ERROR")); logging.exception(f"Treeview sort failed (Column: {col}):")


# --- Main Execution Block ---
def main():
    """Sets up the main application window, creates ScraperApp, runs event loop."""
    root = None
    try:
        root = tk.Tk()
        app = ScraperApp(root)
        icon_path = "spider.ico"
        if os.path.exists(icon_path):
            try: root.iconbitmap(icon_path); logging.info(f"Icon set from: {icon_path}")
            except tk.TclError as e: log_queue.put((f"Warn: Could not load icon '{icon_path}': {e}", "WARNING"))
            except Exception as e: log_queue.put((f"Warn: Unexpected error loading icon '{icon_path}': {e}", "WARNING"))
        else: log_queue.put((f"Info: Icon file '{icon_path}' not found.", "INFO"))
        log_queue.put((f"Arachne Scraper {app.root.title().split(' V')[-1]} started. Ready.", "INFO"))
        def on_closing():
            app.stop_log_monitor()
            if app.scraping_in_progress or app.metadata_in_progress:
                 if messagebox.askokcancel("Confirm Quit", "Processing active. Quit anyway?"):
                     log_queue.put(("User quit while processing active.", "WARNING")); root.destroy()
                 else: app.start_log_monitor(); return
            else: log_queue.put(("Exiting application.", "INFO")); root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
    except Exception as e_main_startup:
        logging.critical(f"CRITICAL STARTUP ERROR: {e_main_startup}", exc_info=True)
        try:
             if root: root.withdraw()
             messagebox.showerror("Fatal Startup Error", f"Critical error on startup:\n\n{e_main_startup}\n\nCheck log ({log_filename}). Closing.")
        except tk.TclError: print(f"FATAL STARTUP ERROR: {e_main_startup}. Check log '{log_filename}'.")
        sys.exit(1)

if __name__ == "__main__":
    try: main()
    except Exception as e_unhandled:
         logging.critical(f"FATAL UNHANDLED EXCEPTION: {e_unhandled}", exc_info=True)
         print(f"FATAL UNHANDLED ERROR. Check log '{log_filename}'. Error: {e_unhandled}")
         sys.exit(1)