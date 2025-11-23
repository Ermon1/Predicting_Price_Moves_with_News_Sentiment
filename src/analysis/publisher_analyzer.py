import pandas as pd
import re
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader

def run_publisher_analysis():
    """Fast publisher analysis"""
    config = config_loader.load_config("data_sources")
    
    df = data_loader.load_tabular_data(config["data_sources"]["inputs"]["financial_news"])
    
    publisher_counts = df["publisher"].value_counts()
    
    # Email domain analysis
    email_mask = df["publisher"].str.contains(r'@', na=False)
    domains = {}
    if email_mask.any():
        domains = dict(df[email_mask]["publisher"].str.extract(r'@([\w.]+)')[0].value_counts().head(5))
    
    return {
        "publisher_activity": {
            "total_publishers": len(publisher_counts),
            "top_10_publishers": dict(publisher_counts.head(10)),
            "concentration": f"{(publisher_counts.head(10).sum() / len(df) * 100):.1f}%"
        },
        "email_domains": domains,
        "publisher_specialization": dict(df.groupby('publisher')['stock'].nunique().sort_values(ascending=False).head(5))
    }