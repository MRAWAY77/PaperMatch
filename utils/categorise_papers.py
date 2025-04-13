import os
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
import shutil

# === Keyword dictionary for each category ===
categories = {
    "Medical_Fraud_and_Malpractice": [
        "medical fraud", "healthcare fraud", "insurance fraud", "billing fraud", "medicare fraud",
        "medicaid fraud", "upcoding", "phantom billing", "patient brokering", "kickback scheme",
        "medical identity theft", "prescription fraud", "fraudulent diagnosis", "unnecessary procedure",
        "health insurance fraud", "pharmaceutical fraud",
        "malpractice", "unethical treatment", "experimental treatment", "informed consent violation",
        "patient exploitation", "medical negligence", "falsified credentials", "medical license fraud",
        "counterfeit medicine", "unlicensed practice", "medical data manipulation", "clinical trial fraud"
    ],
    "Misinformation_and_Fake_News": [
        "fake news", "disinformation", "misinformation", "propaganda", "conspiracy theory",
        "information warfare", "social media manipulation", "deepfake", "fact-checking", "media literacy",
        "information disorder", "filter bubble", "echo chamber", "algorithmic bias", "synthetic media",
        "information operations", "coordinated inauthentic behavior", "influence operation", "astroturfing",
        "computational propaganda", "source verification", "media bias", "journalistic integrity",
        "information source", "primary source", "secondary source", "citation needed", "unverified claim",
        "anonymous source", "information provenance", "source attribution", "fact versus opinion",
        "source criticism"
    ],
    "Organised_Crime_and_Drug_Trafficking": [
        "organized crime", "organised crime", "criminal syndicate", "mafia", "crime network",
        "criminal organization", "criminal organisation", "mob", "racketeering", "criminal enterprise",
        "illegal operation", "crime family", "criminal group", "underworld", "yakuza", "triads",
        "criminal clan", "criminal conspiracy", "crime syndicate",
        "drug trafficking", "drug trade", "narcotics trade", "illegal drug", "drug smuggling",
        "cocaine trade", "heroin distribution", "methamphetamine", "drug cartel", "drug syndicate",
        "drug network", "illicit drug", "drug smuggler", "controlled substance", "drug ring",
        "international drug trade", "drug bust", "narcotics trafficking", "illegal drug trade",
        "narcotic distribution", "heroin trafficking", "methamphetamine trade", "drug distribution",
        "illegal drug network", "drug interdiction", "drug supply chain"
    ],
    "Cybercrime_and_Digital_Fraud": [
        "cyber attack", "malware", "ransomware", "phishing", "data breach", "hacking", "cybercrime",
        "cyber security", "cyber criminal", "dark web", "cyber fraud", "identity theft", "cyber espionage",
        "botnet", "DDoS attack", "cyber warfare", "computer virus", "data theft", "online fraud",
        "cryptocurrency crime",
        "zero-day exploit", "social engineering", "encryption", "keylogger", "backdoor", "brute force attack",
        "SQL injection", "man-in-the-middle", "password cracking", "spyware", "trojan horse", "rootkit",
        "cryptojacking", "extortion"
    ],
    "Forensic_Science_and_Criminal_Investigation": [
        "DNA analysis", "fingerprint analysis", "ballistics", "toxicology", "forensic pathology",
        "crime scene investigation", "digital forensics", "blood pattern analysis", "forensic anthropology",
        "trace evidence", "forensic entomology", "forensic psychology", "autopsy", "serology",
        "chain of custody", "forensic odontology", "chromatography", "spectroscopy", "PCR amplification",
        "mass spectrometry", "microscopy", "luminol test", "substance identification", "comparative analysis",
        "facial reconstruction", "voice analysis", "handwriting analysis", "geographic profiling"
    ]
}

raw_pdf_dir = "raw_papers"
output_base_dir = "categorized_papers"
output_csv = "categorized_summary.csv"

# Create output folders
os.makedirs(output_base_dir, exist_ok=True)
for cat in categories:
    os.makedirs(os.path.join(output_base_dir, cat), exist_ok=True)

results = []

# List all PDF files
pdf_files = [os.path.join(raw_pdf_dir, f) for f in os.listdir(raw_pdf_dir) if f.lower().endswith(".pdf")]

print(f"🔍 Scanning {len(pdf_files)} PDF files...\n")

for pdf_path in tqdm(pdf_files):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = text.lower()

        matched = {}

        for cat, keywords in categories.items():
            match_count = sum(1 for kw in keywords if kw.lower() in text)
            if match_count >= 1:
                matched[cat] = match_count

        if matched:
            for cat, count in matched.items():
                dest_path = os.path.join(output_base_dir, cat, os.path.basename(pdf_path))
                shutil.copy(pdf_path, dest_path)

                results.append({
                    "file": os.path.basename(pdf_path),
                    "category": cat,
                    "matched_keywords": count
                })

    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")

# Save results summary
if results:
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Categorization complete! Summary saved to: {output_csv}")
else:
    print("\n⚠️ No matches found for any file.")