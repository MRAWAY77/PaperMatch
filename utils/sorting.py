import pandas as pd

# Load the CSV
df = pd.read_csv("categorized_summary.csv")

# Ensure matched_keywords is treated as an integer
df["matched_keywords"] = pd.to_numeric(df["matched_keywords"], errors="coerce").fillna(0).astype(int)

# Group by category and matched keyword count
summary = df.groupby(["category", "matched_keywords"]).size().reset_index(name="num_papers")

# Sort by category and descending keyword count
summary = summary.sort_values(by=["category", "matched_keywords"], ascending=[True, False])

# Print results nicely
for category in summary["category"].unique():
    print(f"\n📁 {category.replace('_', ' ')}")
    sub_df = summary[summary["category"] == category]
    for _, row in sub_df.iterrows():
        print(f"  {row['num_papers']} papers matched {row['matched_keywords']} unique keywords")


'''
📁 Cybercrime and Digital Fraud
  1 papers matched 13 unique keywords
  1 papers matched 12 unique keywords
  1 papers matched 11 unique keywords
  2 papers matched 8 unique keywords
  2 papers matched 7 unique keywords
  1 papers matched 6 unique keywords
  10 papers matched 5 unique keywords
  15 papers matched 4 unique keywords
  25 papers matched 3 unique keywords
  67 papers matched 2 unique keywords
  331 papers matched 1 unique keywords

📁 Forensic Science and Criminal Investigation
  1 papers matched 4 unique keywords
  3 papers matched 3 unique keywords
  9 papers matched 2 unique keywords
  167 papers matched 1 unique keywords

📁 Medical Fraud and Malpractice
  1 papers matched 2 unique keywords
  11 papers matched 1 unique keywords

📁 Misinformation and Fake News
  1 papers matched 8 unique keywords
  3 papers matched 6 unique keywords
  4 papers matched 5 unique keywords
  4 papers matched 4 unique keywords
  32 papers matched 3 unique keywords
  82 papers matched 2 unique keywords
  171 papers matched 1 unique keywords

📁 Organized Crime and Drug Trafficking
  2 papers matched 4 unique keywords
  1 papers matched 3 unique keywords
  10 papers matched 2 unique keywords
  349 papers matched 1 unique keywords

'''