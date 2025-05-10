# pip install pandas thefuzz python-Levenshtein

import os
import shutil
import pandas as pd
from thefuzz import fuzz
import re


def clean_filename(filename):
    """Clean and normalize a filename for better matching."""
    if pd.isna(filename):
        return ""

    # Convert to string and strip whitespace
    filename = str(filename).strip()

    # Remove file extensions
    filename = os.path.splitext(filename)[0]

    # Remove common prefixes/suffixes that might differ between files
    filename = re.sub(r'^(news_|article_|doc_)', '', filename, flags=re.IGNORECASE)

    # Replace special characters with spaces
    filename = re.sub(r'[_\-\.]', ' ', filename)

    # Remove extra whitespace and convert to lowercase
    filename = ' '.join(filename.split()).lower()

    return filename


def fuzzy_match_news_articles(summary_file, index_file, similarity_threshold=75):
    """
    Process news articles using fuzzy matching to pair filenames from summary_file
    with paths from index_file and copy the text files to Raw_NA folder.
    """
    print(f"Processing news articles using {summary_file} and {index_file} with fuzzy matching...")

    try:
        # Read the Excel files
        summary_df = pd.read_excel(summary_file)
        index_df = pd.read_excel(index_file)

        # Ensure destination folder exists
        if not os.path.exists("Raw_NA"):
            print("Creating Raw_NA folder...")
            os.makedirs("Raw_NA")

        # Prepare index data with cleaned filenames for matching
        index_data = []
        for _, row in index_df.iterrows():
            if 'filename' in row and 'path' in row and not pd.isna(row['filename']) and not pd.isna(row['path']):
                original_filename = str(row['filename']).strip()
                cleaned_filename = clean_filename(original_filename)
                index_data.append({
                    'original_filename': original_filename,
                    'cleaned_filename': cleaned_filename,
                    'path': str(row['path']).strip()
                })

        print(f"Found {len(index_data)} files in the index")

        # Process each row in the summary file
        success_count = 0
        no_match_count = 0

        for index, row in summary_df.iterrows():
            if 'filename' not in row or pd.isna(row['filename']):
                continue

            summary_filename = str(row['filename']).strip()
            cleaned_summary_filename = clean_filename(summary_filename)

            if not cleaned_summary_filename:
                continue

            print(f"Looking for matches for: {summary_filename}")

            # Find the best match using fuzzy matching
            best_match = None
            best_score = 0
            best_match_details = None

            for item in index_data:
                # Calculate similarity score
                score = fuzz.ratio(cleaned_summary_filename, item['cleaned_filename'])

                # Also try partial ratio which can help with substrings
                partial_score = fuzz.partial_ratio(cleaned_summary_filename, item['cleaned_filename'])
                score = max(score, partial_score)

                if score > best_score:
                    best_score = score
                    best_match = item
                    best_match_details = f"{item['original_filename']} (Score: {score})"

            if best_match and best_score >= similarity_threshold:
                source_path = best_match['path']

                # Make sure the file exists
                if os.path.exists(source_path):
                    # Create destination path
                    dest_path = os.path.join("Raw_NA", os.path.basename(source_path))

                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    success_count += 1
                    print(f"MATCH FOUND: {summary_filename} -> {best_match_details}")
                    print(f"Copied {source_path} to {dest_path}")
                else:
                    print(f"WARNING: File not found at path: {source_path}")
            else:
                no_match_count += 1
                if best_match:
                    print(f"NO GOOD MATCH: {summary_filename} -> Best candidate: {best_match_details}")
                else:
                    print(f"NO MATCH FOUND for: {summary_filename}")

        print(f"\nSummary:")
        print(f"Successfully copied {success_count} news articles to Raw_NA folder")
        print(f"Could not find matches for {no_match_count} files")

    except Exception as e:
        print(f"Error processing news articles: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to process news articles."""
    # File paths
    news_summary_file = "PaperMatch_news_article_summaries_updated.xlsx"
    news_index_file = "file_index-na.xlsx"

    # Verify files exist
    if not os.path.exists(news_summary_file):
        print(f"Error: {news_summary_file} not found")
        return

    if not os.path.exists(news_index_file):
        print(f"Error: {news_index_file} not found")
        return

    # Process with fuzzy matching
    fuzzy_match_news_articles(news_summary_file, news_index_file)


if __name__ == "__main__":
    main()