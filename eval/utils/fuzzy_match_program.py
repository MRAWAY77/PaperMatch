# pip install pandas openpyxl

import pandas as pd
import re
import os
from difflib import SequenceMatcher

def normalize_text(text):
    """Normalize text for comparison by converting to lowercase and removing punctuation."""
    if not isinstance(text, str):
        if pd.isna(text):
            return ""
        text = str(text)

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_title_from_filename(filename):
    """Extract potential title from filename based on pattern."""
    if not isinstance(filename, str):
        return ""

    # Split by underscore
    parts = filename.split('_')
    if len(parts) < 3:
        return filename

    # Take parts after the second underscore
    title_part = '_'.join(parts[2:])

    # Remove file extension
    if title_part.endswith('.txt'):
        title_part = title_part[:-4]

    # Replace hyphens with spaces
    title_part = title_part.replace('-', ' ')

    return title_part


def calculate_string_similarity(str1, str2):
    """Calculate similarity between two strings using SequenceMatcher."""
    if not str1 or not str2:
        return 0

    # Normalize strings
    s1 = normalize_text(str1)
    s2 = normalize_text(str2)

    # For very different length strings, return low similarity
    if len(s1) < 3 or len(s2) < 3:
        return 0
    if len(s1) > len(s2) * 3 or len(s2) > len(s1) * 3:
        return 0

    # Calculate similarity ratio
    return SequenceMatcher(None, s1, s2).ratio()


def calculate_word_overlap(str1, str2):
    """Calculate word overlap between two strings."""
    if not str1 or not str2:
        return 0

    # Normalize strings
    s1 = normalize_text(str1)
    s2 = normalize_text(str2)

    # Get words (filter out short words)
    words1 = [w for w in s1.split() if len(w) > 2]
    words2 = [w for w in s2.split() if len(w) > 2]

    if not words1 or not words2:
        return 0

    # Count matching words
    match_count = sum(1 for word in words1 if word in words2)

    # Calculate overlap score (percentage of words in str1 found in str2)
    return match_count / len(words1)


def calculate_combined_similarity(title, filename):
    """Calculate combined similarity score between title and filename."""
    extracted_title = extract_title_from_filename(filename)

    string_sim = calculate_string_similarity(title, extracted_title)
    word_overlap = calculate_word_overlap(title, extracted_title)

    # Weight the scores (word overlap is more important)
    return (string_sim * 0.4) + (word_overlap * 0.6)


def find_best_match(title, filenames_df):
    """Find the best matching filename for a given title."""
    best_match = None
    best_score = 0.15  # Minimum threshold for a match

    for _, row in filenames_df.iterrows():
        filename = row['filename']

        score = calculate_combined_similarity(title, filename)

        if score > best_score:
            best_score = score
            best_match = filename

    return best_match


def match_titles_to_filenames(paper_match_file, file_index_file, output_file=None):
    """
    Match titles in PaperMatch file to filenames in file_index and update the filename column.

    Args:
        paper_match_file: Path to the PaperMatch Excel file
        file_index_file: Path to the file_index Excel file
        output_file: Path to save the updated PaperMatch file (if None, overwrites original)

    Returns:
        DataFrame with updated filename column
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = paper_match_file

    # Load both files
    print(f"Loading files...")
    paper_match_df = pd.read_excel(paper_match_file)
    file_index_df = pd.read_excel(file_index_file)

    print(f"Loaded {len(paper_match_df)} titles and {len(file_index_df)} filenames")

    # Print column names for debugging
    print(f"Paper match columns: {paper_match_df.columns.tolist()}")
    print(f"File index columns: {file_index_df.columns.tolist()}")

    # Check for column name (case-insensitive)
    title_column = None
    for col in paper_match_df.columns:
        if col.lower() == 'title':
            title_column = col
            break

    if title_column is None:
        raise ValueError(
            "No 'title' column found in paper match file. Available columns: " + str(paper_match_df.columns.tolist()))

    # Track statistics
    total_titles = len(paper_match_df)
    matched_titles = 0
    not_matched_titles = 0

    # Iterate through titles and find best matches
    print("Finding best matches for each title...")
    for index, row in paper_match_df.iterrows():
        if pd.isna(row[title_column]):  # Use the detected column name
            continue

        title = row[title_column]  # Use the detected column name

        # Find best match
        best_match = find_best_match(title, file_index_df)

        if best_match:
            # Update filename column
            paper_match_df.at[index, 'filename'] = best_match
            matched_titles += 1

            # Print progress every 10 matches
            if matched_titles % 10 == 0:
                print(f"Processed {matched_titles} matches...")
        else:
            not_matched_titles += 1

    # Print summary
    print(f"\nMatching complete!")
    print(f"Total titles: {total_titles}")
    print(f"Matched titles: {matched_titles}")
    print(f"Unmatched titles: {not_matched_titles}")

    # Save updated file
    paper_match_df.to_excel(output_file, index=False)
    print(f"Updated file saved to {output_file}")

    return paper_match_df


if __name__ == "__main__":
    # File paths
    paper_match_file = "PaperMatch_news_article_summaries_simple.xlsx"
    file_index_file = "file_index-na.xlsx"
    output_file = "PaperMatch_news_article_summaries_updated.xlsx"

    # Run the matching process
    updated_df = match_titles_to_filenames(paper_match_file, file_index_file, output_file)

    print("Fuzzy matching complete!")