# 📘 PaperMatch: AI-Enhanced Insights for Academic & Media Discovery via Telegram

![PaperMatch Telegram Example](assets/telegram.png)

## 🚀 Executive Summary

PaperMatch is an AI-driven solution designed to streamline the process of discovering academic research and interpreting media narratives around key topics.

With the exponential rise in scholarly publications and fragmented media coverage, researchers face information overload and lack tools that connect academic rigor with real-world discourse. PaperMatch bridges that gap by offering:

* 🧠 Summarization of academic papers using LLMs (Llama3 via Ollama)
* 🌍 News analysis and clustering by topic
* 🤖 A single interface (via Telegram) to query and receive structured reports

Backed by insights from academic research on information behavior and media framing (e.g., Bawden & Robinson, 2009; Tenopir et al., 2012; Entman, 1993), PaperMatch offers both relevance and context in one unified platform.

---

## 🧑‍💻 Project Credits

| Name                | Student ID | Contributions                                                                                                |
| ------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| Alvin Wong Ann Ying | A0266486M  | Project lead, academic scraping, graph/embedding pipeline, LLM summarization, Telegram API, code integration, evaluation workflow, all README documentation, system demo, report and pptx proposal, LaTex integration |
| Bertrand Tan Yu-Jin | A0292314J  | News scraping, clustering for media content, evaluation workflow, README documentation, pptx proposal        |

---

## 🏗️ System Architecture

![System Architecture](assets/architecture.png)

---

## 🎥 Demo

[![Watch the Demo](assets/demo_thumbnail.png)](https://youtu.be/6D-vXmser9Q)

---

## 📘 User Guide

🛠️ Supported OS: Ubuntu 24.04

### 🔧 1. Prerequisites

Install Ollama and pull the Llama3.3 model:

```bash
git clone https://github.com/MRAWAY77/PaperMatch.git
cd PaperMatch/query/
ollama pull llama3.3 # depending on what models you want to use
```

You can also choose from other available models you’ve already pulled:

* `llama3:latest` (4.7 GB)
* `qwen3:latest` (5.2 GB)
* `llama3.3:latest` (42 GB)
* `qwen3:30b-a3b` (18 GB)
* `mistral-small3.1:latest` (15 GB)
* `gemma3:latest` (3.3 GB)

More models are available at [https://ollama.com/search](https://ollama.com/search).

---

### 📦 2. Local Installation

Set up the environment:

```bash
conda create -n papermatch python=3.10.16 -y
conda activate papermatch
pip install -r requirements.txt
```

Next, download and extract the required folders into the project root directory:

📁 Download from Google Drive
🔗 [https://drive.google.com/drive/folders/1dBYjVw8gvEo9h2YO9v3y-TY6y9i8uvvH?usp=drive\_link](https://drive.google.com/drive/folders/1dBYjVw8gvEo9h2YO9v3y-TY6y9i8uvvH?usp=drive_link)

Download and unzip:

* datasets.zip
* cluster_embeddings.zip
* EVALUATION/ (Only needed for reviewing the evaluation artifacts in detail)

Make sure the extracted folders are placed according to the directory structure shown in Section 3.

---

### 📁 3. Project Directory Structure

The directory tree should look like this:

```
papermatch/
├── academic_papers.csv
├── assets/
│   ├── architecture.png
│   ├── MDDI_blend.jpg
│   ├── MDDI.jpg
│   ├── sample_report.pdf
│   └── telegram.png
├── datasets/
│   ├── Academic_Clusters/
│   ├── News_Clusters/
│   └── academic_metadata.csv
├── graph_network/
│   ├── cluster_embeddings/
│   ├── graph_network.py
│   ├── cluster_embeddings.py
│   ├── output_graphs/
│   └── sim_score_graph_network.txt
├── project_report/
│   ├── PLP_Project_Proposal_Presentation.pdf
│   ├── PaperMatch.pdf
├── eval/
│   ├── config.py
│   ├── eval_summaries.py
│   ├── query_eval.py
├── query/
│   ├── config.py
│   ├── llm.py
│   ├── query.py
│   ├── main.py
│   ├── eval_logs/
│   └── ...
├── scrapers/
├── news_webscraper/
│   ├── Arachne.py
│   ├── assets/
│   │   ├── arachne_interface_01.jpg
│   │   ├── arachne_interface_02.jpg
│   │   └── arachne_interface_03.jpg
│   ├── cluster_organizer.py
│   ├── MetaCluster-I.py
│   ├── README.md
│   ├── Tapestry-Ik.py
│   └── Weaver.py
├── paperwithcode_chrome.py
└── paperwithcode_firefox.py
├── utils/
│   ├── categorise_papers.py
│   ├── helper.ipynb
│   ├── sorting.py
│   ├── topic_modeling.py
│   └── topic_model_results/
├── LICENSE
├── README.md
├── requirements.txt
└── setup.sh
```

---

### 📲 4. Telegram API Configuration

To connect PaperMatch to a Telegram group:

1. Visit [https://my.telegram.org](https://my.telegram.org) and create an app.
2. Retrieve your api\_id and api\_hash.
3. Create a group chat and copy the group link.

Update query/config.py:

```python
api_id = <insert your api_id>
api_hash = '<insert your api_hash>'
phone_number = '<insert your phone number>'
TARGET_CHANNEL = '<insert telegram group link>'
```

---

### ▶️ 5. Running the App

Start the Telegram-integrated query interface:

```bash
python query/main.py
```

Once the script is running and connected to your Telegram group, interact with the bot using the following command:

📩  Send this in your Telegram group:

/ask \[your query here]

This command triggers the AI pipeline, which retrieves relevant academic papers, summarizes them, matches media coverage, and generates a contextualized PDF report directly in your Telegram chat.

Example:

/ask What are the latest Deepfake trends?

This will generate an AI-enhanced academic-media insight report based on your query.

---

### 📄 6. Sample Output Report

Example report generated by the system:

📄 Sample Report: [View PDF](assets/sample_report.pdf)

---

## 📚 Project Report

See the full project report in the `project_report` folder.

---

## 🙋 Support Personnel

For any issues or inquiries, please contact the support personnel listed below:

| Name           | Role                   | Contact Email                 |
|----------------|------------------------|-------------------------------|
| Alvin Wong     | Team Lead/AI & SW Eng  | <alvinwongannying@gmail.com>  |
| Bertrand Tan   | Member/AI Eng          | <bertrand.tanyj@gmail.com>    |

---
