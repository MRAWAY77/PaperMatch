# pip install pandas numpy nltk textstat rouge bert-score scikit-learn openpyxl

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import string
import re
import textstat
from rouge import Rouge
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.translate.meteor_score import meteor_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """Clean and preprocess text for analysis."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove unnecessary whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = text.strip()
    
    return text

def compute_rouge_scores(reference, candidate):
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    try:
        rouge = Rouge()
        if not reference or not candidate:
            return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
        
        scores = rouge.get_scores(candidate, reference)[0]
        return scores
    except Exception as e:
        print(f"Error computing ROUGE scores: {e}")
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

def compute_bertscore(reference, candidate):
    """Compute BERTScore for semantic similarity."""
    try:
        if not reference or not candidate:
            return 0.0
        
        # BERTScore requires lists of references and candidates
        P, R, F1 = bert_score([candidate], [reference], lang='en')
        return F1.item()  # Return F1 score as a scalar
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        return 0.0

def compute_meteor(reference, candidate):
    """Compute METEOR score."""
    try:
        if not reference or not candidate:
            return 0.0
        
        # Tokenize the texts
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # Calculate METEOR score
        score = meteor_score([ref_tokens], cand_tokens)
        return score
    except Exception as e:
        print(f"Error computing METEOR score: {e}")
        return 0.0

def extract_keywords(text, top_n=20):
    """Extract important keywords from text using TF-IDF."""
    try:
        if not text or len(text) < 10:
            return []
        
        # Tokenize and clean text
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        
        # If not enough words for meaningful extraction
        if len(words) < 5:
            return words
        
        # Use TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores for each word
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Return just the words
            return [word for word, score in sorted_keywords if score > 0]
        except:
            # Fallback to simple word frequency for very short texts
            counter = Counter(words)
            return [word for word, count in counter.most_common(top_n)]
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def compute_keyword_overlap(ref_keywords, cand_keywords):
    """Compute the overlap of keywords between reference and candidate."""
    try:
        if not ref_keywords or not cand_keywords:
            return 0.0
        
        # Convert to sets for intersection/union
        ref_set = set(ref_keywords)
        cand_set = set(cand_keywords)
        
        # Calculate Jaccard similarity
        if len(ref_set) == 0 and len(cand_set) == 0:
            return 1.0  # Both have no keywords
        
        if len(ref_set) == 0 or len(cand_set) == 0:
            return 0.0  # One has keywords, the other doesn't
            
        jaccard = len(ref_set.intersection(cand_set)) / len(ref_set.union(cand_set))
        return jaccard
    except Exception as e:
        print(f"Error computing keyword overlap: {e}")
        return 0.0

def compute_content_coverage(reference, candidate):
    """Compute content coverage based on important keywords."""
    try:
        if not reference or not candidate:
            return 0.0
        
        # Create TF-IDF vectorizer to find important keywords
        vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
        
        # Fit on reference to extract important keywords
        try:
            vectorizer.fit([reference])
            tfidf_matrix = vectorizer.transform([reference, candidate])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get keyword presence in both documents
            ref_keywords = set()
            cand_keywords = set()
            
            # Extract non-zero TF-IDF values for each document
            ref_vector = tfidf_matrix[0].toarray()[0]
            cand_vector = tfidf_matrix[1].toarray()[0]
            
            for i, val in enumerate(ref_vector):
                if val > 0:
                    ref_keywords.add(feature_names[i])
            
            for i, val in enumerate(cand_vector):
                if val > 0:
                    cand_keywords.add(feature_names[i])
            
            # Calculate overlap
            common_keywords = ref_keywords.intersection(cand_keywords)
            if len(ref_keywords) == 0:
                return 0.0
                
            coverage = len(common_keywords) / len(ref_keywords)
            return coverage
        except:
            # Handle case with very short texts
            return 0.0
    except Exception as e:
        print(f"Error computing content coverage: {e}")
        return 0.0

def compute_summary_statistics(text):
    """Compute basic summary statistics."""
    try:
        if not text:
            return {
                'word_count': 0,
                'sentence_count': 0,
                'avg_sentence_length': 0,
                'readability_score': 0
            }
        
        # Count words and sentences
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        word_count = len(words)
        sentence_count = len(sentences)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Calculate readability using Flesch-Kincaid grade level
        readability_score = textstat.flesch_kincaid_grade(text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability_score
        }
    except Exception as e:
        print(f"Error computing summary statistics: {e}")
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'readability_score': 0
        }

def evaluate_summaries(reference_file, candidate_file, output_file, file_type):
    """Evaluate summaries and save results to an output Excel file."""
    try:
        # Read input files
        ref_df = pd.read_excel(reference_file)
        cand_df = pd.read_excel(candidate_file)
        
        # Standardize column names
        ref_col_mapping = {col: col.lower() for col in ref_df.columns}
        ref_df = ref_df.rename(columns=ref_col_mapping)
        
        cand_col_mapping = {col: col.lower() for col in cand_df.columns}
        cand_df = cand_df.rename(columns=cand_col_mapping)
        
        # Match files by filename
        results = []
        match_count = 0
        
        # Map reference files by filename for quick lookup
        ref_map = {row['filename']: row for _, row in ref_df.iterrows() if 'filename' in row}
        
        # If filename column doesn't exist in reference, try with capitalized 'Filename'
        if len(ref_map) == 0 and 'filename' in ref_df.columns:
            ref_map = {row['filename']: row for _, row in ref_df.iterrows()}
        
        # Process each candidate summary
        for _, cand_row in cand_df.iterrows():
            result_row = {}
            
            # Copy metadata from candidate row
            for col in cand_df.columns:
                if col in cand_row:
                    result_row[f'candidate_{col}'] = cand_row[col]
            
            # Get filename
            filename = cand_row.get('filename', '')
            result_row['filename'] = filename
            
            # Find matching reference
            ref_row = ref_map.get(filename)
            
            if ref_row is not None:
                match_count += 1
                
                # Copy metadata from reference row
                for col in ref_df.columns:
                    if col in ref_row and col != 'filename':
                        result_row[f'reference_{col}'] = ref_row[col]
                
                # Get the summaries
                cand_summary_col = 'summary' if 'summary' in cand_row else 'Summary'
                ref_summary_col = 'summary' if 'summary' in ref_row else 'Summary'
                
                cand_summary = str(cand_row.get(cand_summary_col, ''))
                ref_summary = str(ref_row.get(ref_summary_col, ''))
                
                # Preprocess the texts
                cand_summary = preprocess_text(cand_summary)
                ref_summary = preprocess_text(ref_summary)
                
                # Calculate evaluation metrics
                rouge_scores = compute_rouge_scores(ref_summary, cand_summary)
                result_row['rouge1_f1'] = rouge_scores['rouge-1']['f']
                result_row['rouge2_f1'] = rouge_scores['rouge-2']['f']
                result_row['rougeL_f1'] = rouge_scores['rouge-l']['f']
                
                # Compute BERTScore
                result_row['bertscore_f1'] = compute_bertscore(ref_summary, cand_summary)
                
                # Compute METEOR score
                result_row['meteor_score'] = compute_meteor(ref_summary, cand_summary)
                
                # Extract keywords instead of entities
                ref_keywords = extract_keywords(ref_summary)
                cand_keywords = extract_keywords(cand_summary)
                result_row['keyword_overlap'] = compute_keyword_overlap(ref_keywords, cand_keywords)
                
                # Compute content coverage
                result_row['content_coverage'] = compute_content_coverage(ref_summary, cand_summary)
                
                # Compute summary statistics
                ref_stats = compute_summary_statistics(ref_summary)
                cand_stats = compute_summary_statistics(cand_summary)
                
                result_row['reference_word_count'] = ref_stats['word_count']
                result_row['reference_sentence_count'] = ref_stats['sentence_count']
                result_row['reference_avg_sentence_length'] = ref_stats['avg_sentence_length']
                result_row['reference_readability'] = ref_stats['readability_score']
                
                result_row['candidate_word_count'] = cand_stats['word_count']
                result_row['candidate_sentence_count'] = cand_stats['sentence_count']
                result_row['candidate_avg_sentence_length'] = cand_stats['avg_sentence_length']
                result_row['candidate_readability'] = cand_stats['readability_score']
                
                # Calculate length ratio (candidate/reference)
                if ref_stats['word_count'] > 0:
                    result_row['length_ratio'] = cand_stats['word_count'] / ref_stats['word_count']
                else:
                    result_row['length_ratio'] = 0
                
                # Compute overall score (weighted average of metrics)
                overall_score = (
                    0.25 * result_row['rouge1_f1'] +
                    0.15 * result_row['rouge2_f1'] +
                    0.15 * result_row['rougeL_f1'] +
                    0.20 * result_row['bertscore_f1'] +
                    0.10 * result_row['meteor_score'] +
                    0.10 * result_row['keyword_overlap'] +
                    0.05 * result_row['content_coverage']
                )
                result_row['overall_score'] = overall_score
            else:
                # No matching reference found
                result_row['match_found'] = False
                for metric in ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bertscore_f1', 
                              'meteor_score', 'keyword_overlap', 'content_coverage',
                              'reference_word_count', 'reference_sentence_count',
                              'reference_avg_sentence_length', 'reference_readability',
                              'candidate_word_count', 'candidate_sentence_count',
                              'candidate_avg_sentence_length', 'candidate_readability',
                              'length_ratio', 'overall_score']:
                    result_row[metric] = None
            
            results.append(result_row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add summary statistics at the end
        if len(results_df) > 0 and 'overall_score' in results_df.columns:
            metrics_to_summarize = [
                'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'bertscore_f1', 
                'meteor_score', 'keyword_overlap', 'content_coverage', 'overall_score'
            ]
            
            summary_row = {'filename': f'AVERAGE ({file_type})'}
            for metric in metrics_to_summarize:
                summary_row[metric] = results_df[metric].mean()
            
            results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Save results to output file
        results_df.to_excel(output_file, index=False)
        
        print(f"Evaluation complete for {file_type}. Matched {match_count} out of {len(cand_df)} files.")
        print(f"Results saved to {output_file}")
        
        return results_df
    except Exception as e:
        print(f"Error evaluating summaries: {e}")
        return pd.DataFrame()

def main():
    """Main function to run the evaluation."""
    print("Starting summary evaluation...")
    
    # Academic papers evaluation
    academic_results = evaluate_summaries(
        "Ref_academic_summaries.xlsx",
        "PaperMatch_academic_paper_summaries_final.xlsx",
        "academic_evaluation_results.xlsx",
        "Academic Papers"
    )
    
    # News articles evaluation
    news_results = evaluate_summaries(
        "Ref_news_summaries.xlsx",
        "PaperMatch_news_article_summaries_final.xlsx",
        "news_evaluation_results.xlsx",
        "News Articles"
    )
    
    print("Evaluation completed successfully!")
    
    # Print summary of results
    if not academic_results.empty:
        print("\nAcademic Papers Summary:")
        print(f"Average ROUGE-1: {academic_results['rouge1_f1'].mean():.4f}")
        print(f"Average ROUGE-2: {academic_results['rouge2_f1'].mean():.4f}")
        print(f"Average ROUGE-L: {academic_results['rougeL_f1'].mean():.4f}")
        print(f"Average BERTScore: {academic_results['bertscore_f1'].mean():.4f}")
        print(f"Average METEOR: {academic_results['meteor_score'].mean():.4f}")
        print(f"Average Keyword Overlap: {academic_results['keyword_overlap'].mean():.4f}")
        print(f"Average Content Coverage: {academic_results['content_coverage'].mean():.4f}")
        print(f"Average Overall Score: {academic_results['overall_score'].mean():.4f}")
    
    if not news_results.empty:
        print("\nNews Articles Summary:")
        print(f"Average ROUGE-1: {news_results['rouge1_f1'].mean():.4f}")
        print(f"Average ROUGE-2: {news_results['rouge2_f1'].mean():.4f}")
        print(f"Average ROUGE-L: {news_results['rougeL_f1'].mean():.4f}")
        print(f"Average BERTScore: {news_results['bertscore_f1'].mean():.4f}")
        print(f"Average METEOR: {news_results['meteor_score'].mean():.4f}")
        print(f"Average Keyword Overlap: {news_results['keyword_overlap'].mean():.4f}")
        print(f"Average Content Coverage: {news_results['content_coverage'].mean():.4f}")
        print(f"Average Overall Score: {news_results['overall_score'].mean():.4f}")

if __name__ == "__main__":
    main()
