# News Scraping and Article Extraction Suite

This repository contains a collection of Python web scrapers for extracting search results from major news outlets [Straits Times, CNA, CNN, BBC], and an extraction tool to format collected articles.

## Project Structure

- **Arachne-II** - Web scraper for BBC News
- **Arachne-III_t3** - Web scraper for CNN
- **Arachne-IV** - Web scraper for The Straits Times
- **Arachne-Va** - Web scraper for Channel News Asia (CNA)
- **Tapestry-Ik** - Article content extractor (optimized for CNN articles)

## Installation

### Prerequisites

- Python 3.8+
- Google Chrome (for Selenium-based scrapers)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/news-scraping-suite.git
cd news-scraping-suite
```

2. Install required dependencies:
```bash
pip install pandas requests beautifulsoup4 selenium webdriver-manager lxml html5lib tqdm
```

## Usage

Each scraper operates independently and can be run individually.

### Arachne-II (BBC News Scraper)

A scraper for BBC News search results.

```python
# Run the scraper
python arachne-ii.py
```

Features:
- Extracts article titles, URLs, and summaries from BBC search results
- Processes multiple pages of search results
- Normalizes relative URLs to full BBC URLs
- Saves results to a time-stamped CSV file

### Arachne-III_t3 (CNN Scraper)

A Selenium-based scraper for CNN search results.

```python
# Run the scraper
python arachne-iii_t3.py
```

Features:
- Uses Selenium WebDriver for browser automation
- Handles dynamic page loading and JavaScript-rendered content
- Extracts article titles, URLs, dates, and summaries
- Takes screenshots for debugging if errors occur
- Saves results to time-stamped CSV files

### Arachne-IV (Straits Times Scraper)

A Selenium-based scraper for The Straits Times search results.

```python
# Run the scraper
python arachne-iv.py
```

Features:
- Uses Selenium WebDriver for browser automation
- Extracts article titles, URLs, and summaries from search results
- Handles dynamic content loading
- Implements detailed logging for debugging
- Saves results to time-stamped CSV files

### Arachne-Va (CNA Scraper)

A Selenium-based scraper for Channel News Asia search results with advanced anti-detection methods.

```python
# Run the scraper
python arachne-va.py
```

Features:
- Implements advanced anti-detection techniques
- Multiple fallback selectors for robust extraction
- Detailed error handling and logging
- Browser screenshot capability for debugging
- Saves both individual page results and consolidated results

### Tapestry-Ik (Article Content Extractor)

A sophisticated article content extraction tool optimized for CNN articles but also compatible with other news sources.

```python
# Run the content extractor
python tapestry-ik.py
```

Features:
- Extracts full article text, preserving structure
- Handles different news sources including CNN, BBC, Straits Times, and CNA
- Preserves text formatting, lists, and headings
- Extracts article metadata (title, date, source, etc.)
- Creates organized directory structure for extracted articles
- Handles paywalled content through archive services

## Configuration

Each scraper can be configured by editing the respective Python file:

- To change search terms or URLs, modify the URL strings in the `main()` function
- To adjust scraping behavior, modify the timeout and delay parameters
- To customize output, edit the filename patterns

## Ethical Considerations

Please use these tools responsibly and ethically:

- Respect the robots.txt file of each website
- Implement appropriate rate limiting to avoid overloading servers
- Be aware of and comply with the Terms of Service of each news outlet
- Use the extracted content for personal, research, or legitimate business purposes only
- Do not redistribute copyrighted content without permission

## Troubleshooting

Common issues and solutions:

- **WebDriver errors**: Make sure Google Chrome is installed and updated to the latest version
- **Element not found errors**: The website structure may have changed; update the CSS selectors
- **Rate limiting**: If you encounter 403 or 429 errors, increase the delay between requests
- **Blank results**: Check the debug screenshots and logs for more information

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

These tools are provided for educational and research purposes only. The developers assume no liability for any misuse or for any damages resulting from the use of this software.