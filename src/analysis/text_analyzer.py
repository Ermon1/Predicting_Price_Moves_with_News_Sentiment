import pandas as pd
import re
from collections import Counter
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader

def run_text_analysis():
    """Fast text analysis"""
    config = config_loader.load_config("data_sources")
    
    df = data_loader.load_tabular_data(config["data_sources"]["inputs"]["financial_news"])
    
    # Common keywords
    all_text = ' '.join(df["headline"].dropna().str.lower())
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text)
    common_words = dict(Counter(words).most_common(10))
    
    # Financial terms
    financial_terms = ['earnings', 'stock', 'price', 'target', 'growth', 'profit', 'fda', 'approval']
    term_counts = {}
    for term in financial_terms:
        count = df["headline"].str.contains(term, case=False, na=False).sum()
        if count > 0:
            term_counts[term] = count
    
    return {
        "common_keywords": common_words,
        "financial_terms": term_counts,
        "topic_patterns": {
            "price_targets": df["headline"].str.contains(r'\$', na=False).sum(),
            "fda_mentions": df["headline"].str.contains('fda', case=False, na=False).sum()
        }
    }