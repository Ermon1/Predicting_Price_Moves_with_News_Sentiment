import pandas as pd
import numpy as np
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader

def run_descriptive_analysis():
    """Fast descriptive statistics analysis"""
    config = config_loader.load_config("data_sources")
    params = config["data_sources"]["parameters"]["task1"]
    
    df = data_loader.load_tabular_data(config["data_sources"]["inputs"]["financial_news"])
    
    return {
        "textual_stats": {
            "headline_length": {
                "mean": float(df["headline"].str.len().mean()),
                "max": int(df["headline"].str.len().max()),
                "min": int(df["headline"].str.len().min())
            }
        },
        "publisher_stats": {
            "total_publishers": df["publisher"].nunique(),
            "top_publishers": dict(df["publisher"].value_counts().head(10))
        },
        "date_stats": {
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "total_articles": len(df)
        }
    }