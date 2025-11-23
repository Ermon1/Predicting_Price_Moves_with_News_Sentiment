import pandas as pd
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader

def run_temporal_analysis():
    """Fast temporal analysis"""
    config = config_loader.load_config("data_sources")
    
    df = data_loader.load_tabular_data(config["data_sources"]["inputs"]["financial_news"])
    df['date'] = pd.to_datetime(df['date'])
    
    # Publication frequency over time
    daily_counts = df.set_index('date').resample('D').size()
    
    # Time patterns
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    
    return {
        "publication_frequency": {
            "daily_mean": float(daily_counts.mean()),
            "daily_max": int(daily_counts.max()),
            "spike_threshold": float(daily_counts.mean() + 2 * daily_counts.std())
        },
        "time_patterns": {
            "busiest_hour": int(df['hour'].mode().iloc[0]),
            "busiest_day": df['day_of_week'].mode().iloc[0],
            "hourly_distribution": dict(df['hour'].value_counts().sort_index().head(5))
        }
    }