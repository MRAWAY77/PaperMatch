import os
import shutil
import pandas as pd


def process_academic_papers(summary_file, index_file):
    """
    Process academic papers by matching filenames from summary_file with paths from index_file
    and copying the PDFs to Raw_AP folder.
    """
    print(f"Processing academic papers using {summary_file} and {index_file}...")
    try:
        # Read the Excel files
        summary_df = pd.read_excel(summary_file)
        index_df = pd.read_excel(index_file)

        # Create a mapping of filename to path from the index file
        file_mapping = {}
        for _, row in index_df.iterrows():
            if 'filename' in row and 'path' in row and not pd.isna(row['filename']) and not pd.isna(row['path']):
                file_mapping[str(row['filename']).strip()] = str(row['path']).strip()

        print(f"Found {len(file_mapping)} filename-to-path mappings in {index_file}")

        # Process each row in the summary file
        success_count = 0
        for _, row in summary_df.iterrows():
            if 'filename' not in row or pd.isna(row['filename']):
                continue

            filename = str(row['filename']).strip()

            # Look up the path in the mapping
            if filename in file_mapping:
                source_path = file_mapping[filename]

                # Make sure the file exists
                if os.path.exists(source_path):
                    # Create destination path
                    dest_path = os.path.join("Raw_AP", os.path.basename(source_path))

                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    success_count += 1
                    print(f"Copied {source_path} to {dest_path}")
                else:
                    print(f"Warning: Could not find file at path {source_path} for {filename}")
            else:
                print(f"Warning: No path found for {filename} in index file")

        print(f"Successfully copied {success_count} academic papers to Raw_AP folder")
    except Exception as e:
        print(f"Error processing academic papers: {str(e)}")


def process_news_articles(summary_file, index_file):
    """
    Process news articles by matching filenames from summary_file with paths from index_file
    and copying the text files to Raw_NA folder.
    """
    print(f"Processing news articles using {summary_file} and {index_file}...")
    try:
        # Read the Excel files
        summary_df = pd.read_excel(summary_file)
        index_df = pd.read_excel(index_file)

        # Create a mapping of filename to path from the index file
        file_mapping = {}
        for _, row in index_df.iterrows():
            if 'filename' in row and 'path' in row and not pd.isna(row['filename']) and not pd.isna(row['path']):
                file_mapping[str(row['filename']).strip()] = str(row['path']).strip()

        print(f"Found {len(file_mapping)} filename-to-path mappings in {index_file}")

        # Process each row in the summary file
        success_count = 0
        for _, row in summary_df.iterrows():
            if 'filename' not in row or pd.isna(row['filename']):
                continue

            filename = str(row['filename']).strip()

            # Look up the path in the mapping
            if filename in file_mapping:
                source_path = file_mapping[filename]

                # Make sure the file exists
                if os.path.exists(source_path):
                    # Create destination path
                    dest_path = os.path.join("Raw_NA", os.path.basename(source_path))

                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    success_count += 1
                    print(f"Copied {source_path} to {dest_path}")
                else:
                    print(f"Warning: Could not find file at path {source_path} for {filename}")
            else:
                print(f"Warning: No path found for {filename} in index file")

        print(f"Successfully copied {success_count} news articles to Raw_NA folder")
    except Exception as e:
        print(f"Error processing news articles: {str(e)}")


def main():
    """Main function to coordinate the processing of all files."""
    # Check if destination folders exist
    if not os.path.exists("Raw_AP"):
        print("Warning: Raw_AP folder not found. Creating it...")
        os.makedirs("Raw_AP")

    if not os.path.exists("Raw_NA"):
        print("Warning: Raw_NA folder not found. Creating it...")
        os.makedirs("Raw_NA")

    # Process academic papers
    academic_summary_file = "PaperMatch_academic_paper_summaries_simple.xlsx"
    academic_index_file = "file_index-ap.xlsx"  # Note: Using the corrected filename

    if os.path.exists(academic_summary_file) and os.path.exists(academic_index_file):
        process_academic_papers(academic_summary_file, academic_index_file)
    else:
        if not os.path.exists(academic_summary_file):
            print(f"Warning: {academic_summary_file} not found")
        if not os.path.exists(academic_index_file):
            print(f"Warning: {academic_index_file} not found")

    # Process news articles
    news_summary_file = "PaperMatch_news_article_summaries_updated.xlsx"
    news_index_file = "file_index-na.xlsx"  # Note: Using the corrected filename

    if os.path.exists(news_summary_file) and os.path.exists(news_index_file):
        process_news_articles(news_summary_file, news_index_file)
    else:
        if not os.path.exists(news_summary_file):
            print(f"Warning: {news_summary_file} not found")
        if not os.path.exists(news_index_file):
            print(f"Warning: {news_index_file} not found")


if __name__ == "__main__":
    main()