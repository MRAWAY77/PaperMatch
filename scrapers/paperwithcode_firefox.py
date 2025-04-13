import os
import requests
import time
import random
import json
import re
from tqdm import tqdm
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from bs4 import BeautifulSoup
from fpdf import FPDF
import pdfkit

# === Configuration ===
download_dir = "B_papers"
os.makedirs(download_dir, exist_ok=True)

# Realistic Firefox User-Agent headers
firefox_user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
]

user_agent = random.choice(firefox_user_agents)

headers = {
    "User-Agent": user_agent,
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

task_slugs = [
    "decoder", "deepfake-detection", "drug-discovery", "drug-response-prediction",
    "exposure-fairness", "inference-attack"
]

def clean_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text.strip().replace(" ", "_")[:150]

def setup_driver():
    options = Options()
    options.headless = False  # Set to True if you want headless mode

    # Setting Firefox preferences directly in Options
    options.set_preference("general.useragent.override", user_agent)
    options.set_preference("dom.webdriver.enabled", False)
    options.set_preference("useAutomationExtension", False)

    service = Service()  # auto-detects geckodriver in PATH
    return webdriver.Firefox(service=service, options=options)

def wait_for_user_to_solve_captcha(driver):
    print("⚠️ CAPTCHA detected! Please manually solve the CAPTCHA and then press Enter to continue...")
    input("Press Enter once you've solved the CAPTCHA and the page has reloaded...")

def download_arxiv_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url, headers=headers)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded: {save_path}")
        else:
            print(f"❌ Failed to download PDF: {pdf_url} (Status code: {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading PDF: {e}")

def download_papers_for_task(task_slug, n=100):
    base_url = f"https://paperswithcode.com/task/{task_slug}/latest"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    metadata_file = f'metadata_log_{task_slug}_{timestamp}.json'
    
    seen_paper_urls = set()
    metadata_list = []
    paper_id = 1
    page = 1
    downloaded_count = 0
    empty_page_count = 0
    max_empty_pages = 5

    print(f"\n🚀 Starting task: {task_slug}")
    driver = setup_driver()

    with tqdm(total=n, desc=f"Downloading {task_slug}", ncols=100) as pbar:
        while downloaded_count < n:
            print(f"\n📄 Fetching page {page}...")
            url = f"{base_url}?page={page}"
            driver.get(url)
            time.sleep(3)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            links = soup.select("a[href^='/paper/']")
            new_papers_this_page = 0

            for link in links:
                if downloaded_count >= n:
                    break

                paper_path = link['href'].split('#')[0]
                paper_url = "https://paperswithcode.com" + paper_path
                if paper_url in seen_paper_urls:
                    continue
                seen_paper_urls.add(paper_url)

                print(f"🔗 Visiting paper: {paper_url}")
                driver.get(paper_url)
                time.sleep(3)

                try:
                    # CAPTCHA detection
                    if driver.find_elements(By.XPATH, "//div[@class='g-recaptcha']"):
                        wait_for_user_to_solve_captcha(driver)

                    # Click the "Paper" button if it exists
                    paper_buttons = driver.find_elements(By.XPATH, "//button[text()='Paper']")
                    if paper_buttons:
                        paper_buttons[0].click()
                        time.sleep(3)

                    # Click the "PDF" link
                    pdf_button = driver.find_element(By.LINK_TEXT, "PDF")
                    pdf_button.click()
                    time.sleep(5)

                    # CAPTCHA again?
                    if driver.find_elements(By.XPATH, "//div[@class='g-recaptcha']"):
                        wait_for_user_to_solve_captcha(driver)

                    # Get PDF URL
                    pdf_url = driver.current_url
                    print(f"📄 Final PDF URL: {pdf_url}")

                    pdf_filename = f"{task_slug}_{paper_id}.pdf"
                    pdf_path = os.path.join(download_dir, pdf_filename)
                    download_arxiv_pdf(pdf_url, pdf_path)

                    title = pdf_url.split("/")[-1]
                    metadata_list.append({
                        "title": title,
                        "filename": pdf_filename,
                        "paper_url": paper_url,
                        "pdf_url": pdf_url,
                        "publication_date": "Unknown"
                    })

                    downloaded_count += 1
                    paper_id += 1
                    new_papers_this_page += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"❌ Failed on {paper_url}: {e}")
                    continue

            if new_papers_this_page == 0:
                empty_page_count += 1
                if empty_page_count >= max_empty_pages:
                    print("🚫 Stopping: max empty pages reached.")
                    break
            else:
                empty_page_count = 0

            page += 1

    driver.quit()

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Finished downloading {downloaded_count} papers for task: {task_slug}")

# Run downloader for each task
for slug in task_slugs:
    download_papers_for_task(slug, n=100)