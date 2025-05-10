# --- START OF FILE Weaver.py ---

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from urllib.parse import urlparse
import logging
import time

log = logging.getLogger(__name__)

# --- Keyword Definitions (Unchanged) ---
# Theme: Organised Crime
ORGANISED_CRIME_KEYWORDS = [
    'organized crime', 'organised crime', 'mafia', 'cartel', 'syndicate', 'gang',
    'criminal network', 'criminal organization', 'criminal organisation', 'mob',
    'racket', 'racketeering', 'criminal enterprise', 'crime family', 'criminal group',
    'underworld', 'yakuza', 'triads', 'criminal clan', 'money laundering',
    'ndrangheta', 'cosa nostra', 'camorra', 'sinaloa', 'medellin', 'cali cartel',
    'jalisco', 'zetas', 'gulf cartel', 'hells angels'
]
# Theme: Drug Trafficking
DRUG_TRAFFICKING_KEYWORDS = [
    'drug trafficking', 'drug smuggling', 'narcotics trade', 'drug trade', 'narcotic',
    'drug cartel', 'drug ring', 'drug dealer', 'drug kingpin', 'drug lord', 'drug mule',
    'cocaine', 'heroin', 'methamphetamine', 'meth lab', 'drug production', 'marijuana trafficking',
    'cannabis trafficking', 'opium', 'fentanyl', 'drug seizure', 'drug bust', 'narco state'
]
# Theme: Cybercrime
CYBERCRIME_KEYWORDS = [
    'cybercrime', 'cyber crime', 'cybersecurity', 'cyber security', 'hacking', 'hacker',
    'data breach', 'phishing', 'malware', 'ransomware', 'digital fraud', 'online scam',
    'dark web', 'cyber attack', 'ddos', 'botnet', 'identity theft', 'carding',
    'business email compromise', 'bec scam', 'crypto scam', 'cryptocurrency scam'
]
# Theme: Forensics
FORENSICS_KEYWORDS = [
    'forensic science', 'criminal investigation', 'csi', 'crime scene', 'dna analysis',
    'fingerprint', 'ballistics', 'forensic pathology', 'digital forensics', 'trace evidence',
    'forensic accounting', 'forensic expert', 'autopsy report', 'crime lab'
]
# Theme: Misinformation
MISINFORMATION_KEYWORDS = [
    'misinformation', 'fake news', 'propaganda', 'false information', 'information disorder',
    'rumor mill', 'conspiracy theory', 'hoax', 'fact check', 'media literacy'
]
# Theme: Disinformation
DISINFORMATION_KEYWORDS = [
    'disinformation', 'influence operation', 'psyops', 'psychological operation', 'troll farm',
    'state-sponsored disinformation', 'foreign interference', 'malicious campaign', 'hybrid warfare'
]
# Theme: Medical Fraud
MEDICAL_FRAUD_KEYWORDS = [
    'medical fraud', 'healthcare fraud', 'health care fraud', 'malpractice', 'medical negligence',
    'billing fraud', 'upcoding', 'kickback scheme', 'phantom billing', 'unnecessary procedure',
    'prescription fraud', 'counterfeit drugs', 'quackery'
]
# --- General Crime Keywords ---
GENERAL_CRIME_KEYWORDS = [
    'extortion', 'bribery', 'corruption', 'human trafficking', 'sex trafficking',
    'arms trafficking', 'weapons smuggling', 'counterfeit', 'illegal gambling',
    'protection racket', 'hitman', 'contract killing', 'assassination', 'criminal conspiracy',
    'fraud scheme', 'black market', 'illegal trade', 'smuggling', 'contraband', 'theft', 'robbery',
    'burglary', 'assault', 'murder', 'homicide', 'arrested', 'charged', 'convicted', 'sentenced',
    'police raid', 'investigation'
]
# --- High Relevance Phrases ---
HIGH_RELEVANCE_CRIME_PHRASES = [
    'criminal kingpin', 'major drug bust', 'international criminal organization', 'transnational crime',
    'major cyber attack', 'large scale fraud', 'significant data breach'
]
# --- URL/Title Keywords ---
URL_TITLE_RELEVANCE_KEYWORDS = [
    'crime', 'criminal', 'drug', 'narco', 'cartel', 'mafia', 'gang', 'fraud', 'scam',
    'cyber', 'hack', 'forensic', 'investigation', 'trafficking', 'misinformation',
    'disinformation', 'propaganda', 'malpractice'
]

# --- Helper Functions ---
def _fetch_and_parse(url):
    # (Unchanged)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        try: soup = BeautifulSoup(response.text, 'lxml')
        except: soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e: log.error(f"Request failed for {url}: {e}"); return None
    except Exception as e: log.error(f"Error parsing {url}: {e}"); return None

def _extract_date(soup):
    """Extracts publication date, attempts parsing, and standardizes to DD/MM/YYYY."""
    if not soup: return None
    date_str = None

    # Prioritized list of methods to find a date string
    methods = [
        # 1. Semantic HTML <time> tag (most reliable)
        lambda s: s.find('time', datetime=True),
        # 2. Common Meta Tags
        lambda s: s.select_one('meta[property="article:published_time"]'),
        lambda s: s.select_one('meta[name="pubdate"]'),
        lambda s: s.select_one('meta[name="publishdate"]'),
        lambda s: s.select_one('meta[name="date"]'),
        lambda s: s.select_one('meta[itemprop="datePublished"]'),
        lambda s: s.select_one('meta[name="dc.date.issued"]'),
        lambda s: s.select_one('meta[name="article.published"]'), # Another meta variation
        lambda s: s.select_one('meta[name="parsely-pub-date"]'),
        lambda s: s.select_one('meta[name="sailthru.date"]'),
        # 3. JSON-LD (requires parsing)
        lambda s: s.find_all('script', type='application/ld+json'),
        # 4. Specific, reliable text selectors (Add site-specific ones high up)
        lambda s: s.select_one('div.article-publish span'), # CNA Specific from screenshot
        lambda s: s.select_one('span[data-testid="card-metadata-lastUpdated"]'), # BBC Specific
        lambda s: s.select_one('.byline__time'), # Often reliable
        lambda s: s.select_one('.date-line'),   # Another common pattern
        # 5. More generic text selectors (lower priority)
        lambda s: s.select_one('.date'),
        lambda s: s.select_one('.time'),
        lambda s: s.select_one('.timestamp'),
        lambda s: s.select_one('.byline__datetime'),
        lambda s: s.select_one('.entry-date'),
        lambda s: s.select_one('.published'),
        lambda s: s.select_one('.post-date'),
        lambda s: s.select_one('.article-date'),
    ]

    # --- Extraction Loop ---
    for i, method_func in enumerate(methods):
        if date_str: break # Stop if we found a date string already
        try:
            result = method_func(soup)
            if not result: continue

            # JSON-LD handling
            if isinstance(result, list) and result and result[0].name == 'script':
                 log.debug(f"Checking JSON-LD scripts (Method {i+1})")
                 for script in result:
                     # ... (JSON parsing logic remains the same as previous version) ...
                     try:
                         import json
                         data = json.loads(script.string or "")
                         items_to_check = []
                         if isinstance(data, dict): items_to_check = data.get('@graph', [data])
                         elif isinstance(data, list): items_to_check = data
                         for item in items_to_check:
                             if isinstance(item, dict):
                                 pub_date = item.get('datePublished') or item.get('publishDate') or item.get('dateCreated')
                                 if pub_date and isinstance(pub_date, str):
                                     date_str = pub_date; log.debug(f"Found date in JSON-LD: {date_str}"); break
                         if date_str: break
                     except (json.JSONDecodeError, TypeError, AttributeError) as e: log.warning(f"Error parsing JSON-LD for date: {e}")
                 if date_str: break # Found in JSON-LD

            # Meta Tags
            elif hasattr(result, 'name') and result.name == 'meta' and result.get('content'):
                potential_str = result['content'].strip()
                if potential_str: date_str = potential_str; log.debug(f"Found date in Meta ({result.attrs}): {date_str}"); break

            # Time Tag
            elif hasattr(result, 'name') and result.name == 'time' and result.get('datetime'):
                potential_str = result['datetime'].strip()
                if potential_str: date_str = potential_str; log.debug(f"Found date in <time datetime>: {date_str}"); break

            # Text Selectors
            elif hasattr(result, 'name'): # Is a Tag object
                potential_str = result.get_text(strip=True)
                # --- CRITICAL: Check if the text is relative ---
                # If text contains common relative terms, SKIP IT unless it *also* contains a year.
                # This prevents "7 months ago" from being chosen over an absolute date found later.
                if potential_str and any(term in potential_str.lower() for term in ['ago', 'hour', 'minute', 'yesterday', 'today']) \
                   and not re.search(r'\b(19|20)\d{2}\b', potential_str): # Check if a 4-digit year is missing
                    log.debug(f"Skipping potential relative date text: '{potential_str}'")
                    continue # Skip this relative text and try the next method
                # --- End Relative Check ---
                elif potential_str:
                    date_str = potential_str
                    log.debug(f"Found potential date text via selector: {date_str}")
                    break # Use this text

        except Exception as e:
            log.warning(f"Error during date extraction method {i+1}: {e}")

    if not date_str:
        log.warning("Could not extract any processable date string.")
        return None

    # --- Check if Extracted String is STILL Relative (redundancy) ---
    if any(term in date_str.lower() for term in ['ago', 'hour', 'minute', 'yesterday', 'today']) and not re.search(r'\b(19|20)\d{2}\b', date_str):
         log.warning(f"Final extracted string '{date_str}' appears relative. Rejecting.")
         return None # Cannot parse relative dates

    # --- Parsing Attempt ---
    log.debug(f"Attempting to parse date string: '{date_str}'")
    # Cleaning: Remove prefixes but try to KEEP TIME initially for formats that need it.
    date_str_cleaned = re.sub(r'^(Published|Updated|Posted|On|Last updated|First published)[:\s]*', '', date_str, flags=re.IGNORECASE).strip()
    # Remove extra info after pipe only if it doesn't look like part of the date/time
    if '|' in date_str_cleaned and not re.search(r'\d\s*\|', date_str_cleaned):
         date_str_cleaned = re.sub(r'\s*\|\s*.*$', '', date_str_cleaned).strip()
    # Remove trailing timezone *names* (like EST, PST) but keep offsets (+0800) for now
    date_str_cleaned = re.sub(r'\s+([A-Z]{3,5})$', '', date_str_cleaned).strip()
    date_str_cleaned = re.sub(r'\s{2,}', ' ', date_str_cleaned).strip()

    # Expanded list of formats, prioritizing those with time components
    formats_to_try = [
        # ISO formats (often have time/zone)
        '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z', '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S %z', '%Y-%m-%d %H:%M:%S',
        # RFC formats (often have time/zone)
        '%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y %H:%M:%S %z',
        # Formats with time (AM/PM or 24hr) - Added CNA format here
        '%d %b %Y %I:%M%p',   # 21 Dec 2022 06:04AM (No comma, common variation) - CNA format
        '%d %B %Y %I:%M%p',   # 21 December 2022 06:04AM
        '%d %b %Y, %I:%M %p', # 21 Dec 2022, 06:04 AM
        '%d %B %Y, %I:%M %p', # 21 December 2022, 06:04 AM
        '%d %b %Y %H:%M',     # 21 Dec 2022 18:05
        '%d %B %Y %H:%M',     # 21 December 2022 18:05
        '%b %d, %Y, %I:%M %p',# Dec 21, 2022, 6:04 PM
        '%B %d, %Y, %I:%M %p',# December 21, 2022, 6:04 PM
        # Date-only formats (Month Name)
        '%b %d %Y',           # Feb 11 2017
        '%B %d %Y',           # February 11 2017
        '%d %b %Y',           # 11 Feb 2017
        '%d %B %Y',           # 11 February 2017
        '%b %d, %Y',          # Feb 11, 2017
        '%B %d, %Y',          # February 11, 2017
        # Date-only formats (Numeric Month)
        '%m/%d/%Y', '%m-%d-%Y', # MM/DD/YYYY or MM-DD-YYYY
        '%d/%m/%Y', '%d-%m-%Y', # DD/MM/YYYY or DD-MM-YYYY
        '%Y/%m/%d', '%Y-%m-%d', # YYYY/MM/DD or YYYY-MM-DD
    ]

    parsed_date = None
    for fmt in formats_to_try:
        try:
            temp_date_str = date_str_cleaned # Use the cleaned string for this attempt
            # Handle timezone offset string manipulation if needed by the format
            if '%z' in fmt:
                 temp_date_str = temp_date_str.replace('Z', '+0000')
                 # Remove colon in offset like +08:00 -> +0800 if needed by fmt
                 temp_date_str = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', temp_date_str)
                 # Add basic UTC offset if format requires it and none is present
                 if not re.search(r'[+-]\d{4}$', temp_date_str): temp_date_str += '+0000'
            elif '%Z' in fmt:
                 # Remove timezone name if format expects it but we stripped it (e.g., 'GMT')
                 # This is less reliable, better to have formats without %Z if possible
                 pass # Usually stripping TZ names is safer

            parsed_date = datetime.strptime(temp_date_str, fmt)
            log.info(f"Successfully parsed date '{date_str_cleaned}' (Original: '{date_str}') with format '{fmt}'")
            # Standardize output format to DD/MM/YYYY
            return parsed_date.strftime('%d/%m/%Y')
        except (ValueError, TypeError):
            continue # Parsing failed with this format, try the next one

    # If loop finishes without success
    log.warning(f"Failed to parse cleaned date string: '{date_str_cleaned}' (Original: '{date_str}') with any known format.")
    return None # Return None if all parsing attempts fail

def _extract_journalist(soup):
    # (Unchanged)
    if not soup: return None; journalist = None
    meta_selectors = ['meta[name="author"]', 'meta[property="author"]', 'meta[name="cXenseParse:author"]']
    for selector in meta_selectors:
        meta_tag = soup.select_one(selector)
        if meta_tag and meta_tag.get('content'):
            name = meta_tag['content'].strip()
            if name and len(name) > 2 and name.lower() not in ['staff', 'reuters', 'afp', 'ap', 'associated press', 'bloomberg']:
                log.debug(f"J meta ({selector}): {name}")
                return name
    script_tags = soup.find_all('script', type='application/ld+json')
    for script in script_tags:
        try:
            import json; data = json.loads(script.string or ""); items_to_check = []
            if isinstance(data, dict): items_to_check = data.get('@graph', [data])
            elif isinstance(data, list): items_to_check = data
            for item in items_to_check:
                if isinstance(item, dict):
                    author_data = item.get('author') or item.get('creator'); potential_name = None
                    if isinstance(author_data, dict) and author_data.get('name'):
                        potential_name = author_data['name']
                    elif isinstance(author_data, str): potential_name = author_data
                    elif isinstance(author_data, list):
                        names = [a.get('name') for a in author_data if isinstance(a, dict) and a.get('name')]
                        if names:
                            potential_name = ", ".join(n.strip() for n in names if n and len(n) > 2)
                    if potential_name and isinstance(potential_name, str):
                        name = potential_name.strip()
                        if name and len(name) > 2 and name.lower() not in ['staff', 'correspondent']:
                            log.debug(f"J JSON-LD: {name}")
                            return name
        except (json.JSONDecodeError, TypeError, AttributeError) as e: log.warning(f"Error parsing JSON-LD author: {e}")
    byline_selectors = [
        '[data-testid="author-name"]', '[data-testid="byline-new-contributors"] span', '[data-component="byline-block"] span',
        'a[rel="author"]', '.author-name', '.author a', '.byline-name', '.byline a', '.journalist', 'span.author', 'div.author',
        'h6.h6--author-name', 'div.author-card__author-name'
    ]
    for selector in byline_selectors:
        elem = soup.select_one(selector)
        if elem:
            name = elem.get_text(strip=True)
            name = re.sub(r'^(By|From)\s+', '', name, flags=re.IGNORECASE).strip()
            if name and len(name) > 2 and len(name) < 70 and name.lower() not in ['staff', 'contributor', 'correspondent', 'ago']:
                log.debug(f"J selector '{selector}': {name}")
                return name
    log.warning("Could not extract journalist name via standard methods.")
    return None

def _extract_article_text(soup):
    # (Unchanged)
    if not soup: return ""
    article_text_parts = []
    content_selectors = [
        'article[itemprop="articleBody"]', 'div[itemprop="articleBody"]', 'div.article-content', 'div.story-body', 'div.entry-content', 'div.article-body', 'div.main-content', 'div.content', 'div.story',
        'div.ssrcss-11r1m41-RichTextComponentWrapper', 'div.article__body-content', 'div[data-component="text-block"]', 'div.body-content', 'section.article-content', 'div.paragraph--type--text', 'main#main-content', 'div#main', 'article'
    ]
    found_main_content = False
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            log.debug(f"Found {len(elements)} container(s) using selector: '{selector}'")
            for elem in elements:
                for unwanted in elem(['script', 'style', 'aside', 'figure', 'figcaption', '.advertisement', '.related-articles', '.visually-hidden', 'form', 'button', 'iframe', '.ad', '[class*="sidebar"]', '[class*="related"]', '[class*="share"]', '[class*="promo"]', '[class*="tool"]']): unwanted.decompose()
                paragraphs = elem.find_all('p', recursive=True)
                if paragraphs: container_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True));
                else: container_text = elem.get_text(separator=' ', strip=True);
                if container_text: article_text_parts.append(container_text)
            if article_text_parts: found_main_content = True; break
    joined_text = ' '.join(article_text_parts)
    if not found_main_content or len(joined_text) < 250:
        needs_fallback_reason = "Specific container extraction failed" if not found_main_content else "yielded short text"
        log.warning(f"{needs_fallback_reason}, trying fallback (all paragraphs).")
        article_text_parts = []
        all_paragraphs = soup.find_all('p')
        meaningful_paragraphs = []
        for p in all_paragraphs:
            p_text = p.get_text(strip=True)
            if len(p_text) > 40 and not p.find_parent(['nav', 'footer', 'header', 'aside', 'figure', 'figcaption', '[class*="caption"]', '[class*="advert"]', '[class*="legal"]', '[class*="related"]']):
                 if not re.match(r'^\s*(Copyright|©|Updated:|Published:|\d{4}-\d{2}-\d{2})', p_text, re.IGNORECASE): meaningful_paragraphs.append(p_text)
        if meaningful_paragraphs: joined_text = ' '.join(meaningful_paragraphs); log.info(f"Extracted text from {len(meaningful_paragraphs)} paragraphs using fallback.")
        else:
             log.warning("Fallback paragraph extraction failed, using stripped body text.")
             body_tag = soup.find('body')
             if body_tag:
                 for unwanted in body_tag(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript']): unwanted.decompose()
                 joined_text = body_tag.get_text(separator=' ', strip=True)
             else: log.error("Could not find body tag for final fallback."); joined_text = ""
    cleaned_text = re.sub(r'\s{2,}', ' ', joined_text).strip()
    if not cleaned_text: log.warning("Could not extract meaningful article text.")
    return cleaned_text

# --- Relevance Scoring and Categorization (Unchanged) ---
def _calculate_theme_score(keywords, searchable_text):
    score = 0;  # Simple keyword counter
    for keyword in keywords:
        if keyword in searchable_text: score += 2 if ' ' in keyword else 1
    return score

def assess_relevance(title, url, article_text=""):
    if not article_text and not title: return 1, "Error - No Text"
    searchable_text = (title + " " + article_text).lower(); title_lower = title.lower()
    parsed_url = urlparse(url); url_path_query = (parsed_url.path + '?' + parsed_url.query).lower()
    is_audio = 'audio' in url_path_query or 'podcast' in url_path_query
    scores = {"Organised Crime": _calculate_theme_score(ORGANISED_CRIME_KEYWORDS, searchable_text), "Drug Trafficking": _calculate_theme_score(DRUG_TRAFFICKING_KEYWORDS, searchable_text), "Cybercrime": _calculate_theme_score(CYBERCRIME_KEYWORDS, searchable_text), "Forensics": _calculate_theme_score(FORENSICS_KEYWORDS, searchable_text), "Misinformation": _calculate_theme_score(MISINFORMATION_KEYWORDS, searchable_text), "Disinformation": _calculate_theme_score(DISINFORMATION_KEYWORDS, searchable_text), "Medical Fraud": _calculate_theme_score(MEDICAL_FRAUD_KEYWORDS, searchable_text), "General Crime": _calculate_theme_score(GENERAL_CRIME_KEYWORDS, searchable_text) * 0.5}
    crime_boost = 0
    if any(phrase in searchable_text for phrase in HIGH_RELEVANCE_CRIME_PHRASES): crime_boost = 2; scores["Organised Crime"] += crime_boost; scores["Drug Trafficking"] += crime_boost / 2; scores["Cybercrime"] += crime_boost; scores["Medical Fraud"] += crime_boost / 2
    category = "Other"; relevance = 1; min_score_threshold = 2.0
    eligible_themes = {theme: score for theme, score in scores.items() if score >= min_score_threshold and theme != "General Crime"}
    if eligible_themes:
        theme_priority = ["Organised Crime", "Drug Trafficking", "Cybercrime", "Medical Fraud", "Disinformation", "Misinformation", "Forensics"]
        sorted_eligible = sorted(eligible_themes.items(), key=lambda item: (item[1], -theme_priority.index(item[0]) if item[0] in theme_priority else 99), reverse=True)
        best_theme, best_score = sorted_eligible[0]; category = best_theme
        if best_score >= 9: relevance = 5
        elif best_score >= 6: relevance = 4
        elif best_score >= 3: relevance = 3
        else: relevance = 2
        if category in ["Organised Crime", "Drug Trafficking"] and scores["Organised Crime"] >= 2 and scores["Drug Trafficking"] >= 2:
             combined_oc_dt = scores["Organised Crime"] + scores["Drug Trafficking"];
             if combined_oc_dt >= 6: category = "Organised Crime & Drug Trafficking"; relevance = 5 if combined_oc_dt >= 12 else (4 if combined_oc_dt >= 8 else 3)
    elif scores["General Crime"] >= 1.5: category = "General Crime"; relevance = 2
    elif scores["General Crime"] >= 0.5: category = "General Crime"; relevance = 1
    title_url_text = title_lower + " " + url_path_query
    if any(term in title_url_text for term in URL_TITLE_RELEVANCE_KEYWORDS):
        if relevance < 5: relevance = max(relevance + 1, 2); log.debug(f"Relevance boosted by title/URL to {relevance}.")
    if is_audio and relevance > 1: relevance -= 1; log.debug(f"Relevance reduced for audio to {relevance}.")
    log_msg = (f"Relevance assessment for '{title[:50]}...': Category='{category}', Relevance={relevance}. " f"Scores: OC={scores['Organised Crime']:.1f}, DT={scores['Drug Trafficking']:.1f}, Cyber={scores['Cybercrime']:.1f}, For={scores['Forensics']:.1f}, " f"Misc={scores['Misinformation']:.1f}, Dis={scores['Disinformation']:.1f}, Med={scores['Medical Fraud']:.1f}, Gen={scores['General Crime']:.1f}. Boost={crime_boost}")
    log.info(log_msg)
    return relevance, category

# --- Main Processing Function ---
def process_article_url(url, title=""):
    """Fetches, parses, analyzes (metadata + relevance), and returns article data."""
    log.info(f"Weaver processing URL: {url}")
    soup = _fetch_and_parse(url)
    if not soup:
        log.error(f"Failed to fetch or parse URL: {url}")
        return {'processed_date': 'N/A', 'journalist': 'N/A', 'article_text': '', 'relevance_score': 1, 'category': 'Processing Error'}

    original_title = title
    if not title:
        title_tag = soup.find('title'); title = title_tag.get_text(strip=True) if title_tag else ""
        if not title: h1_tag = soup.find('h1'); title = h1_tag.get_text(strip=True) if h1_tag else ""
        if title:
            seps = ['|', ' - ', ' :: ', '«', '»'] # Clean title
            for sep in seps:
                if sep in title:
                    parts = title.split(sep)
                    title = parts[0].strip() if len(parts[0]) > 10 else title
                    break
        if not title: title = "Title Not Found"
        log.info(f"Extracted title from page: {title[:60]}...")
    if title == "Title Not Found" and original_title: title = original_title

    processed_date = _extract_date(soup) # Now handles parsing and format standardization
    journalist = _extract_journalist(soup)
    article_text = _extract_article_text(soup)

    # --- Journalist Fallback Logic ---
    final_journalist = "N/A"
    if journalist:
        final_journalist = journalist
    else:
        log.info("Journalist not found by standard methods, attempting fallbacks...")
        agency_found = None
        # 1. General Agency Fallback (Check start of text)
        if article_text:
            first_chunk = article_text[:150].upper() # Check first 150 chars, case-insensitive check
            agencies_to_check = ["AFP", "REUTERS", "NYTIMES"] # Add more if needed
            for agency in agencies_to_check:
                 # Look for pattern like "(AGENCY)" or "AGENCY -" near start
                 if f"({agency})" in first_chunk or first_chunk.strip().startswith(f"{agency} -"):
                     agency_found = agency
                     final_journalist = agency_found
                     log.info(f"Journalist fallback: Found agency '{agency_found}' in first text chunk.")
                     break # Stop checking once one is found

        # 2. BBC Specific Fallback (Only if no author/agency found)
        if not agency_found:
            try:
                domain = urlparse(url).netloc.replace('www.', '')
                if domain == 'bbc.com':
                    final_journalist = "BBC"
                    log.info("Journalist fallback: Set to 'BBC' (no specific author/agency found).")
            except Exception as e: log.error(f"Error during domain parsing for BBC fallback: {e}")

    relevance_score, category = assess_relevance(title, url, article_text)

    return {
        'processed_date': processed_date if processed_date else 'N/A', # DD/MM/YYYY or N/A
        'journalist': final_journalist,
        'article_text': article_text,
        'relevance_score': relevance_score,
        'category': category
    }

# --- Example Usage (for direct testing) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_urls = [
        # Dates
        "https://www.straitstimes.com/world/europe/rape-no-drug-turf-war-says-czech-anti-fake-news-chief", # Feb 11 2017
        "https://www.channelnewsasia.com/commentary/covid-vaccine-misinformation-fake-news-nz-baby-reject-blood-surgery-3145286", # 21 Dec 2022 06:04AM
        "https://www.bbc.com/news/world-us-canada-61497761", # May 19, 2022
        # Agencies
        "https://www.straitstimes.com/asia/australianz/australia-calls-for-global-action-to-fight-online-misinformation", # (REUTERS)
        "https://www.straitstimes.com/world/europe/france-detains-suspect-after-reports-of-man-with-knife-at-iranian-consulate-in-paris", # (AFP)
        "https://www.nytimes.com/2024/03/04/world/europe/ukraine-disinformation-russia.html" # NYT article - check if NYTIMES mention gets picked up if no author
    ]
    for test_url in test_urls:
        print("-" * 40)
        result = process_article_url(test_url)
        if result:
            print(f"URL: {test_url}")
            print(f"  Date:       {result['processed_date']}") # Should be DD/MM/YYYY
            print(f"  Journalist: {result['journalist']}")   # Check Agency
            print(f"  Category:   {result['category']}")
            print(f"  Relevance:  {result['relevance_score']}")
        else:
            print(f"Failed to process: {test_url}")
        print("-" * 40)
        time.sleep(0.5)