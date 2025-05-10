# **News Article Scraping, Extraction, and Clustering System**

This comprehensive Python-based system includes multiple tools for scraping, extracting, analyzing, and clustering news articles from various sources. It combines web scraping, content extraction, metadata organization, and machine learning-based clustering to help you manage crime-related news data efficiently.

## **Tools Overview**

### 1. **Arachne - News Article Scraper and Analyzer**

A powerful application that scrapes news articles from major news sources and analyzes them for crime-related content. It offers a user-friendly GUI and advanced metadata extraction capabilities.

### 2. **Tapestry - Article Content Extraction Tool**

A tool for extracting full-text content from news articles, even from paywalled sources. It also organizes the extracted articles by metadata and categorizes them for better analysis.

### 3. **News Article Clustering and Organization System**

A complete system for clustering and organizing news articles into thematic groups using machine learning techniques, transforming unstructured article collections into well-organized datasets.

## 📋 **Features**

### **Arachne - News Article Scraper and Analyzer**

#### 🕸️ Web Scraping

* **Multi-source Support**: Scrapes from BBC, CNN, Straits Times, and CNA.
* **Dual Mode Operation**:

  * **Mode 1**: Search news sites using keywords.
  * **Mode 2**: Process specific article URLs.
* **Smart Content Extraction**: Handles different site layouts and structures automatically.

#### 🔍 Content Analysis

* **Relevance Scoring**: Automatic scoring (1-5) based on keyword matching.
* **Category Detection**: Classifies articles into specialized crime categories like Organized Crime, Drug Trafficking, Cybercrime, etc.

#### 📊 Data Management

* **Rich Metadata Extraction**:

  * Publication dates, journalist names, article content, relevance scores, and categories.
* **Excel Export**: Save processed data with full metadata.
* **Duplicate Detection**: Prevents duplicate article entries.

#### 🎨 User Interface

* **Intuitive GUI**: Clean tabbed interface built with Tkinter.
* **Real-time Logging**: View scraping progress and debug information.
* **Batch Processing**: Handle multiple articles efficiently.
* **Theme Context**: Automatically tracks the topic context of your searches.

### **Tapestry - Article Content Extraction Tool**

#### 🎯 Content Extraction

* **Multi-Source Support**: Extracts from BBC, CNN, Straits Times (ST), and CNA.
* **Paywall Bypass**: Automatically uses ArchiveButtons.com for paywalled articles.
* **Smart Content Parsing**: Preserves article structure, including headlines, subheadings, and paragraphs.

#### 📊 Data Processing

* **Excel Integration**: Processes article metadata from Excel files.
* **Automatic Organization**: Saves articles in multiple directory structures (by source, topic, relevance).
* **Metadata Preservation**: Embeds metadata headers in extracted content.
* **Journalist Extraction**: Extracts byline information from articles.
* **Date Formatting**: Standardizes dates to DD/MM/YYYY format.

### **News Article Clustering and Organization System**

#### 🔄 Clustering and Organizing Articles

* **MetaCluster-I.py**: Analyzes and clusters articles using NLP and machine learning techniques.
* **Cluster Organizer**: Organizes files based on clustering results into thematic groups.

## 🚀 **Installation**

### Prerequisites

* Python 3.8+
* Required libraries:

  * `pip install -r requirements.txt` for each tool

#### **Arachne Dependencies:**

```bash
pip install pandas requests beautifulsoup4 tqdm openpyxl pathlib
```

#### **Tapestry Dependencies:**

```bash
pip install pandas requests beautifulsoup4 tqdm openpyxl pathlib
```

#### **News Article Clustering and Organization System Dependencies:**

```bash
pip install scikit-learn pandas matplotlib
```

## 📖 **Usage**

### **Arachne:**

#### Mode 1: Keyword Search

1. Select "Mode 1: Site Search"
2. Choose a news site from the dropdown.
3. Enter keywords or click a theme suggestion button.
4. Click "Start Scraping."
5. After scraping, click "Generate Metadata" to extract detailed information.

#### Mode 2: Direct URL Processing

1. Select "Mode 2: Direct URL List."
2. Paste article URLs (one per line) in the text area.
3. Click "Start Scraping."
4. Click "Generate Metadata" for detailed analysis.

#### Exporting Results

* Navigate to the "Processed Metadata" tab.
* Click "Export Processed to Excel" to save results.

---

### **Tapestry:**

#### Basic Usage

```python
from tapestry import NewsArticleExtractor

# Create extractor instance
extractor = NewsArticleExtractor("your_articles.xlsx")

# Run extraction
extractor.run()
```

#### Input Format

Your Excel file should contain these columns:

* **URL**: Article URL
* **Title**: Article headline
* **Date**: Publication date
* **Media**: News source (BBC, CNN, ST, CNA)
* **Theme**: Topic/theme of the article
* **Relevance**: Relevance score (1-5)
* **Paywall**: (Optional) 'Y' or 'N'
* **Journalist**: (Optional) Will be extracted if not provided

---

### **News Article Clustering and Organization System:**

#### MetaCluster-I.py

1. Load your article dataset (Excel file) into the tool.
2. Run the clustering script to group articles into thematic clusters.
3. The output is a set of files organized by clusters.

#### Cluster Organizer

* Automatically organizes files based on clustering results into source, topic, and relevance-based directories.

## 📁 **Output Structure**

### **Arachne Excel Export Columns**

| Column     | Description                       |
| ---------- | --------------------------------- |
| Media      | Source news site                  |
| Theme      | Search theme or "Direct URL List" |
| Date       | Publication date (DD/MM/YYYY)     |
| Title      | Article headline                  |
| Journalist | Author/Reporter name              |
| URL        | Article URL                       |
| Relevance  | Score (1-5)                       |
| Category   | Detected crime category           |

### **Tapestry Output Format**

Each extracted article is saved as a text file with the following structure:

```
Title: [Article Title]
Date: DD/MM/YYYY
Source: [News Source]
Topic: [Article Theme]
Category: [Category if available]
Relevance: [1-5]
URL: [Original URL]
Paywall: Y/N
Journalist: [Author Name]

[Article Content with preserved structure]
```

### **News Article Clustering and Organization Output**

Articles are organized into directories based on:

* By Source (BBC, CNN, ST, CNA)
* By Topic (Cybercrime, Forensics, etc.)
* By Relevance (R1-R5)

## 🔧 **Configuration**

### **Arachne Site-Specific Settings**

* Configure site selectors, article patterns, and normalization rules for each news site.

### **Tapestry Paywall Detection**

* Automatically checks and handles paywalled articles using ArchiveButtons.com.

### **News Article Clustering Configuration**

* Customize clustering parameters such as the number of clusters or features to be extracted.

## 🛠️ **Troubleshooting**

### **Common Issues**

**Arachne**:

* **Chrome/Selenium Issues**: Ensure Chrome is up to date and WebDriver is correctly installed.

**Tapestry**:

* **Paywall Access**: Check ArchiveButtons.com status or try increasing delays.
* **Missing Content**: Verify the URL and site structure.

**Clustering**:

* Ensure the article dataset is properly formatted and contains the required metadata.

## 🤝 **Contributing**

### Extending Support

* **Arachne**: Add support for new news sites by updating configuration.
* **Tapestry**: Implement new extraction rules or paywall handling logic.
* **Clustering**: Improve clustering algorithms or add new clustering strategies.
