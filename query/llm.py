import os
import glob
import json
import config
import requests
import pandas as pd
import pdfplumber
import string
import easyocr
import spacy
import logging
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from datetime import datetime
from PIL import Image
import json

logging.getLogger("pdfminer").setLevel(logging.ERROR)

def load_config_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Override configuration dynamically
    config.query = data["query"]
    config.TOPIC = data["classified_topic"]

    # Loop through all academic and news clusters
    config.ACADEMIC_CLUSTERS = list(dict.fromkeys([p["cluster"] for p in data["top_academic_papers"]]))
    config.NEWS_CLUSTERS = list(dict.fromkeys([a["cluster"] for a in data["top_news_articles"]]))

    # Rebuild directory paths for each academic cluster
    config.ACADEMIC_PARENT_DIRS = [
        os.path.join(config.ACAD_BASE_DIR, config.TOPIC, cluster) for cluster in config.ACADEMIC_CLUSTERS
    ]

    # Rebuild directory paths for each news cluster
    config.NEWS_PARENT_DIRS = [
        os.path.join(config.NEWS_BASE_DIR, config.TOPIC, cluster) for cluster in config.NEWS_CLUSTERS
    ]

    # Get CSV paths for each academic cluster
    config.CSV_PATHS = [
        glob.glob(os.path.join(academic_parent_dir, "*.csv"))[0]
        if glob.glob(os.path.join(academic_parent_dir, "*.csv")) else None
        for academic_parent_dir in config.ACADEMIC_PARENT_DIRS
    ]

    # Override query and top documents dynamically
    config.academic_papers = [p["paper"] for p in data["top_academic_papers"]]
    config.news_articles = [(a["cluster"], a["article"]) for a in data["top_news_articles"]]

    # === Debug print to confirm config is loaded ===
    print("Loaded config:")
    print("QUERY:", config.query)
    print("TOPIC:", config.TOPIC)
    print("ACADEMIC_CLUSTERS:", config.ACADEMIC_CLUSTERS)
    print("NEWS_CLUSTERS:", config.NEWS_CLUSTERS)
    print("ACADEMIC_PAPERS:", config.academic_papers)
    print("NEWS_ARTICLES:", config.news_articles)
    print("ACADEMIC CSV PATHS:", config.CSV_PATHS)

# === HELPERS ===

reader = easyocr.Reader(['en'], gpu=True)
nlp = spacy.load("en_core_web_trf")

def read_pdf(file_path):
    """Reads and extracts text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    
    cleaned_text = clean_text(text)
    return cleaned_text

def read_txt(file_path):
    """Reads and cleans text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    
    cleaned = clean_text(raw_text)
    return cleaned

# Function to clean and lemmatize the text
def clean_text(text):
    # Process the text with spacy NLP pipeline
    doc = nlp(text)
    
    # Filter out non-English words, stop words, punctuation, and lemmatize
    cleaned_text = " ".join([
        token.lemma_ for token in doc 
        if token.is_alpha and not token.is_stop and token.text not in string.punctuation
    ])
    
    return cleaned_text

def load_paper_url_mapping(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['filename'], df['paper_url']))

def prepare_academic_inputs():
    result = []
    for academic_cluster, csv_path in zip(config.ACADEMIC_CLUSTERS, config.CSV_PATHS):
        url_map = load_paper_url_mapping(csv_path) if csv_path else {}
        for paper in config.academic_papers:
            full_path = os.path.join(config.ACADEMIC_PARENT_DIRS[config.ACADEMIC_CLUSTERS.index(academic_cluster)], paper)
            try:
                content = read_pdf(full_path)
                result.append({
                "type": "academic",
                "title": paper,
                "cluster": academic_cluster,
                "path": full_path,
                "url": url_map.get(paper, "URL_NOT_FOUND"),
                "content": content
            })
            except FileNotFoundError:
                continue
    return result

def prepare_news_inputs():
    result = []
    for news_cluster in config.NEWS_CLUSTERS:
        for filename in config.news_articles:
            if filename[0] == news_cluster:
                full_path = os.path.join(config.NEWS_PARENT_DIRS[config.NEWS_CLUSTERS.index(news_cluster)], filename[1])
                with open(full_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

                title = next((line.split(": ", 1)[1] for line in lines if line.startswith("Title: ")), "TITLE_NOT_FOUND")
                url = next((line.split(": ", 1)[1] for line in lines if line.startswith("URL: ")), "URL_NOT_FOUND")
                try:
                    content = read_txt(full_path)
                    result.append({
                    "type": "news",
                    "title": title,
                    "cluster": news_cluster,
                    "path": full_path,
                    "url": url,
                    "content": content
                })
                except FileNotFoundError:
                    continue
    return result

def send_to_llm(input_data):
    payload = {
        "model": config.MODEL,
        "prompt": f"Summarize this document:\n\n{input_data['content']}"
    }
    try:
        response = requests.post(config.OLLAMA_URL, json=payload, stream=True)
        if not response.ok:
            return f"ERROR: {response.status_code} - {response.text}"

        # Ollama streams JSON lines, each line is a JSON object with a "response" key
        summary_parts = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        summary_parts.append(chunk["response"])
                except json.JSONDecodeError as e:
                    return f"JSON ERROR: {e}"

        return "".join(summary_parts).strip()

    except Exception as e:
        return f"EXCEPTION: {str(e)}"

# def merge_images_with_transparency(img1_path, output_path, alpha=0.2):
#     img1 = Image.open(img1_path).convert("RGBA")
#     white_bg = Image.new("RGBA", img1.size, (255, 255, 255, 255))
#     blended = Image.blend(white_bg, img1, alpha=alpha)
#     blended.save(output_path, format="PNG")

def add_background(canvas, doc, img_path):
    canvas.saveState()
    canvas.drawImage(img_path, 0, 0, width=A4[0], height=A4[1], mask='auto')
    canvas.restoreState()

def generate_pdf_report(results, query, topic, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    bold_style = ParagraphStyle(
    name='BoldLarge',
    parent=styles['Normal'],
    fontName='Helvetica-Bold',
    fontSize=10,
    leading=12
)

    story = []
    story.append(Paragraph(f"<b>Query:</b> {query}", normal_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Topic:</b> {topic}", normal_style))
    story.append(Spacer(1, 24))

    # Table header
    table_data = [[
        Paragraph("<b>Type</b>", bold_style),
        Paragraph("<b>Title</b>", bold_style),
        Paragraph("<b>URL</b>", bold_style),
        Paragraph("<b>Summary</b>", bold_style)
    ]]

    # Table rows
    for item in results:
        table_data.append([
            Paragraph(item["type"], bold_style),
            Paragraph(item["title"], bold_style),
            Paragraph(item["url"], bold_style),
            Paragraph(item["summary"], bold_style)
        ])

    table = Table(table_data, colWidths=[1.0 * inch, 1.5 * inch, 2.0 * inch, 3.5 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    story.append(table)
    story.append(Spacer(1, 24))

    # Raw JSON section
    story.append(Paragraph("<b>Raw JSON Output:</b>", styles["Heading3"]))
    json_dump = json.dumps(results, indent=2)
    for line in json_dump.splitlines():
        story.append(Paragraph(f"<font size=6>{line}</font>", bold_style))

    # Build PDF with background
    doc.build(
        story,
        onFirstPage=lambda c, d: add_background(c, d, config.IMAGE_A),
        onLaterPages=lambda c, d: add_background(c, d, config.IMAGE_A)
    )

# === MAIN RUN ===
def llm(report_callback=None):
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Look for all JSON files status and valid classified topic
    json_files = glob.glob("eval_logs/*.json")
    candidate_jsons = []

    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get("status") == "new" and data.get("classified_topic") != "Unknown":
                    candidate_jsons.append((file, os.path.getmtime(file)))
        except Exception as e:
            print(f"Skipping invalid JSON: {file} — {e}")

    if not candidate_jsons:
        raise ValueError("No valid JSON files found.")

    # Sort by most recent first
    candidate_jsons.sort(key=lambda x: x[1], reverse=True)

    for json_path, _ in candidate_jsons:
        print(f"\nProcessing JSON: {json_path}")

        # Mark as 'processed'
        try:
            with open(json_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data['status'] = 'processed'
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
        except Exception as e:
            print(f"Failed to mark as processed: {e}")
            continue

        try:
            # Load config
            load_config_from_json(json_path)

            # Prepare input documents
            academic_inputs = prepare_academic_inputs()
            news_inputs = prepare_news_inputs()
            all_docs = academic_inputs + news_inputs

            results = []
            for doc in all_docs:
                print(f"Processing: {doc['title']} ({doc['type']})")
                summary = send_to_llm(doc)
                results.append({
                    "type": doc["type"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "summary": summary
                })

            # Generate PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{config.TOPIC}_{timestamp}.pdf"
            output_pdf = os.path.join("eval_logs", output_filename)
            generate_pdf_report(results, config.query, config.TOPIC, output_pdf)

            # Mark JSON as 'completed'
            with open(json_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                data['status'] = 'completed'
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()
            print(f"Marked as completed: {json_path}")

            # Callback
            if report_callback:
                try:
                    report_callback(output_pdf)
                except Exception as e:
                    print(f"Error in report_callback: {e}")

        except Exception as e:
            print(f"Error while processing {json_path}: {e}")
            continue

    end_time = datetime.now()
    duration = end_time - start_time
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s")

if __name__ == "__main__":
    llm(report_callback=None)