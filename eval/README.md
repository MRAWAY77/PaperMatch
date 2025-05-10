# 📘 Academic & News Article Evaluation Pipeline

A comprehensive Python-based pipeline for extracting, summarizing, and evaluating academic papers and news articles using Claude API and advanced NLP techniques.

---

## 📦 Overview

This project provides an end-to-end pipeline for:

1. Extracting summaries from PDF documents  
2. Matching titles to source files using fuzzy matching  
3. Organizing matched files into structured directories  
4. Generating new summaries using Claude 3.5/3.7 Sonnet  
5. Evaluating AI-generated summaries with multiple metrics  

It supports both academic papers and news articles with tailored workflows for each.

---

## 🚀 Features

- ✅ Multi-format PDF summary extraction  
- 🔍 Intelligent title-filename fuzzy matching  
- 📂 Structured folder organization (`Raw_AP`, `Raw_NA`)  
- 🤖 AI summarization using Claude (Anthropic API)  
- 📊 Evaluation via ROUGE, BERTScore, METEOR, readability, and more  
- 🔁 Rate limiting and progress tracking  
- 🔧 Modular and extensible codebase  

---

## 💾 Download

You can download sample files, reference summaries, and input PDFs here:  
📁 [Evaluation](https://drive.google.com/drive/folders/1fZOJew1WHRoouukRcZebKN-im7S7k8jj?usp=drive_link)

---

## 🛠 Prerequisites

- Python 3.8+  
- Anthropic API key (Claude access)  
- ~2GB free disk space  
- Sufficient Claude API quota  

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd your-repo
   ````

2. Install dependencies:

   ```bash
   bash setup.sh
   ```

3. Configure your API key:

   - Replace `"YOUR_API_KEY_HERE"` in:

     - `ap_api_summarizer_nk.py`
     - `na_api_summarizer_nk.py`

---

## 🗂 Project Structure

```
.
├── config.py
├── eval_summaries.py
├── query_eval.py
├── requirements.txt
├── setup.sh
├── summary-evaluation-code.py
├── README.md
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

## 📖 Usage Guide

### 🔹 Phase I: Extract Existing Summaries

```bash
python simple_academic_extractor.py
python simple_news_extractor.py
# or
python document_manager.py
```

- Extract summaries from academic/news PDFs
- Output saved in Excel format

---

### 🔹 Phase II: Match Titles to Files

```bash
python fuzzy_match_program.py
```

- Uses fuzzy matching to associate extracted titles with filenames
- Produces `PaperMatch_*.xlsx` files

---

### 🔹 Phase III: Organize Matched Files

```bash
python ap_copier.py
python na_copier.py
```

- Copies matched documents to `Raw_AP/` and `Raw_NA/` directories

---

### 🔹 Phase IV: Generate AI Summaries

```bash
python ap_api_summarizer_nk.py
python na_api_summarizer_nk.py
```

- Calls Claude API to generate summaries
- Supports large documents with truncation
- Saves incremental progress

---

### 🔹 Phase V: Evaluate AI Summaries

```bash
python summary-evaluation-code.py
```

- Compares generated summaries with reference summaries
- Computes:

  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore
  - METEOR
  - Keyword and content overlap
  - Readability metrics
  - Final weighted evaluation score

---

## 🔧 Configuration

### Claude API Settings

Edit in `ap_api_summarizer_nk.py` and `na_api_summarizer_nk.py`:

```python
API_KEY = "your-anthropic-api-key"
TOKEN_LIMIT_PER_MINUTE = 20000
MAX_DOCUMENT_TOKENS = 20000
MAX_SUMMARY_WORDS = 200
```

---

### Input Files

| Type                | Filename                                                        |
| ------------------- | --------------------------------------------------------------- |
| Academic Index File | `file_index-ap.xlsx`                                            |
| News Index File     | `file_index-na.xlsx`                                            |
| Reference Summaries | `Ref_academic_summaries.xlsx`<br> `Ref_news_summaries.xlsx`     |

---

### Output Files

| Description              | Output Filename                                                                |
| ------------------------ | ------------------------------------------------------------------------------ |
| Extracted summaries      | `academic_paper_summaries_simple.xlsx`<br>`news_article_summaries_simple.xlsx` |
| Matched titles and files | `PaperMatch_*.xlsx`                                                            |
| AI-generated summaries   | `academic_paper_summaries.xlsx`<br>`news_summaries.xlsx`                       |
| Evaluation results       | `academic_evaluation_results.xlsx`<br>`news_evaluation_results.xlsx`           |

---

## ✅ Best Practices

- 💾 **Backup original files** before processing
- 📊 **Monitor API usage** to avoid quota exhaustion
- 🔍 **Manually review samples** for summary quality
- ⚙️ **Process in small batches** for better control
- 📁 **Keep error/debug logs** for traceability
- 🔒 **Never commit API keys** to version control

---
