import os
import requests
import time
import random
import json
import re
from tqdm import tqdm
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from fpdf import FPDF
import pdfkit

# === Configuration ===
download_dir = "scraped_papers"
os.makedirs(download_dir, exist_ok=True)
headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.6261.111 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# List of task slugs to scrape papers from
task_slugs = [
    "adversarial-attack", "adversarial-defense", "backdoor-attack", 
    "backdoor-defense", "benchmarking", "data-poisoning"
]

def clean_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    return text.strip().replace(" ", "_")[:150]

def setup_driver():
    options = Options()
    options.headless = False
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--remote-debugging-port=9222")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/91.0.864.59",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/89.0",
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")

    return webdriver.Chrome(options=options)

def wait_for_user_to_solve_captcha(driver):
    """
    Pause script and wait for user to solve CAPTCHA manually.
    After CAPTCHA is solved, page URL will load as expected.
    """
    print("⚠️ CAPTCHA detected! Please manually solve the CAPTCHA and then press Enter to continue...")
    input("Press Enter once you've solved the CAPTCHA and the page has reloaded...")

def download_arxiv_pdf(pdf_url, save_path):
    """
    Download a PDF file directly from an arXiv PDF URL.
    """
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
                    # Handle CAPTCHA
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

                    # Handle CAPTCHA after clicking the PDF link
                    if driver.find_elements(By.XPATH, "//div[@class='g-recaptcha']"):
                        wait_for_user_to_solve_captcha(driver)

                    # After CAPTCHA, get the page content
                    pdf_url = driver.current_url
                    print(f"📄 Final PDF URL: {pdf_url}")

                    # Save PDF
                    pdf_filename = f"{task_slug}_{paper_id}.pdf"
                    pdf_path = os.path.join(download_dir, pdf_filename)
                    download_arxiv_pdf(pdf_url, pdf_path)
                    
                    # Extract the title or paper ID from the URL
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

    # Save metadata
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Finished downloading {downloaded_count} papers for task: {task_slug}")

# Run downloader for each task slug
for slug in task_slugs:
    download_papers_for_task(slug, n=100)
