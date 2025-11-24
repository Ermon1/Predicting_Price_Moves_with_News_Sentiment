import pandas as pd
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader


def run_temporal_analysis():
    """Fast temporal analysis with proper timezone handling"""
    config = config_loader.load_config("data_sources")

    df = data_loader.load_tabular_data(
        config["data_sources"]["inputs"]["financial_news"]
    )

    # Fix: Use utc=True to handle mixed timezones and convert to timezone-naive
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")

    # Remove any rows with invalid dates
    original_count = len(df)
    df = df.dropna(subset=["date"])
    valid_count = len(df)

    # Convert all to timezone-naive for consistent operations
    df["date"] = df["date"].dt.tz_localize(None)

    # Publication frequency over time
    daily_counts = df.set_index("date").resample("D").size()

    # Time patterns
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()

    return {
        "publication_frequency": {
            "daily_mean": float(daily_counts.mean()),
            "daily_max": int(daily_counts.max()),
            "spike_threshold": float(daily_counts.mean() + 2 * daily_counts.std()),
        },
        "time_patterns": {
            "busiest_hour": (
                int(df["hour"].mode().iloc[0]) if not df["hour"].mode().empty else None
            ),
            "busiest_day": (
                df["day_of_week"].mode().iloc[0]
                if not df["day_of_week"].mode().empty
                else None
            ),
            "hourly_distribution": dict(df["hour"].value_counts().sort_index().head(5)),
        },
        "data_quality": {
            "total_original_records": original_count,
            "valid_records": valid_count,
            "invalid_dates_dropped": original_count - valid_count,
        },
    }
