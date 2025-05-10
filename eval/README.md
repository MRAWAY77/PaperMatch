# Academic & News Article Evaluation Pipeline

A Python-based pipeline for extracting, processing, summarizing, and evaluating academic papers and news articles using the Claude API and advanced NLP techniques.

---

## 🚀 Overview

This end-to-end system includes:

1. **Summary extraction** from existing PDF files
2. **Title-to-filename matching** using fuzzy logic
3. **Organized file structuring** for source documents
4. **AI-generated summaries** via Claude 3.7 Sonnet API
5. **Comprehensive summary evaluation** using multiple metrics

Supports both **academic papers** and **news articles**, with separate handling logic for each type.

---

## 🔧 Features

* 📄 Multi-format PDF summary extraction
* 🧠 Intelligent fuzzy matching of titles to filenames
* 📁 Automated directory structuring (academic/news split)
* 🤖 Claude-powered summarization with rate limiting
* 📊 Evaluation with ROUGE, BERTScore, METEOR, coverage & readability
* 💾 Incremental progress saving to prevent data loss
* ⚙️ Easily extensible for new sources or metrics

---

## 📦 Requirements

* Python 3.8+
* Claude API key (via Anthropic)
* \~2GB of free disk space
* Adequate Claude API quota

---

## 📁 Project Structure

```
.
├── config.py
├── eval_summaries.py
├── query_eval.py
├── requirements.txt
├── setup.sh
├── summary-evaluation-code.py
└── utils/
    ├── ap_api_summarizer_nk.py
    ├── ap_copier.py
    ├── document_manager.py
    ├── fuzzy_match_program.py
    ├── na_api_summarizer_nk.py
    ├── na_copier.py
    ├── pdf_extractor.py
    ├── simple_academic_extractor.py
    └── simple_news_extractor.py
```

---

## ⚙️ Installation

1. Clone the repository

2. Install dependencies:

   ```bash
   bash setup.sh
   ```

3. Configure your Claude API key:
   Replace `YOUR_API_KEY_HERE` in the summarizer scripts or use environment variables for security.

---

## 🧪 Usage Guide

### Phase I: Extract Summaries from PDFs

```bash
# For academic papers
python simple_academic_extractor.py

# For news articles
python simple_news_extractor.py

# Optional advanced extraction
python document_manager.py
```

Output: `academic_paper_summaries_simple.xlsx`, `news_article_summaries_simple.xlsx`

---

### Phase II: Match Titles to Filenames

```bash
python fuzzy_match_program.py
```

* Uses fuzzy logic to associate summary titles with actual document filenames
* Output: `PaperMatch_*.xlsx`

---

### Phase III: Organize Files by Category

```bash
# Academic papers
python ap_copier.py

# News articles
python na_copier.py
```

* Organizes into `Raw_AP/` and `Raw_NA/` folders

---

### Phase IV: Generate AI Summaries

```bash
# Academic papers
python ap_api_summarizer_nk.py

# News articles
python na_api_summarizer_nk.py
```

* Includes rate limiting & incremental saving
* Output: `academic_paper_summaries.xlsx`, `news_summaries.xlsx`

---

### Phase V: Evaluate AI Summaries

```bash
python summary-evaluation-code.py
```

Metrics included:

* ROUGE-1, ROUGE-2, ROUGE-L
* BERTScore
* METEOR
* Keyword overlap
* Content coverage
* Readability scores
* Composite weighted score

Output: `academic_evaluation_results.xlsx`, `news_evaluation_results.xlsx`

---

## ⚙️ Configuration

### `config.py` or in-script variables

```python
API_KEY = "your-anthropic-api-key"
TOKEN_LIMIT_PER_MINUTE = 20000
MAX_DOCUMENT_TOKENS = 20000
MAX_SUMMARY_WORDS = 200
```

---

## 📂 Input & Output Files

| **Type**            | **Filename(s)**                                                               |
| ------------------- | ----------------------------------------------------------------------------- |
| Source PDFs         | PDF files in root or input folders                                            |
| Extracted Summaries | `academic_paper_summaries_simple.xlsx` / `news_article_summaries_simple.xlsx` |
| Matching Index      | `file_index-ap.xlsx` / `file_index-na.xlsx`                                   |
| Reference Summaries | `Ref_academic_summaries.xlsx` / `Ref_news_summaries.xlsx`                     |
| Matched File Output | `PaperMatch_*.xlsx`                                                           |
| AI Summaries        | `academic_paper_summaries.xlsx` / `news_summaries.xlsx`                       |
| Evaluation Results  | `academic_evaluation_results.xlsx` / `news_evaluation_results.xlsx`           |

---

## ✅ Best Practices

* 🔐 **Keep API keys secure** – use `.env` files or system env vars
* 💾 **Backup** all input files and summary outputs regularly
* 📊 **Check evaluation results** manually for quality assurance
* 🔄 **Process in batches** to monitor errors and avoid API timeouts
* 🛠 **Check logs** for script errors and warnings

---

## 🛡 Security Notice

> **Never commit your API keys to GitHub or public repositories.**
> Use environment variables, `.env` files, or secrets managers.

---
