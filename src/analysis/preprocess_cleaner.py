import re
import pandas as pd
from pathlib import Path
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader


def run_task1_preprocessing():
    """
    Full preprocessing for Task-1 (ready for Task-2):
      - normalize timestamps
      - deduplicate rows
      - clean and canonicalize publishers
      - validate stock tickers
      - clean headlines
      - compute basic text metrics
      - persist artifacts for downstream tasks

    Output directory: datasets/preprocess_cleaner
    """

    # --- Load dataset ---
    config = config_loader.load_config("data_sources")
    df = data_loader.load_tabular_data(
        config["data_sources"]["inputs"]["financial_news"]
    )

    # --------------------------
    # Timestamp normalization
    # --------------------------
    df["datetime_utc"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date"] = df["datetime_utc"].dt.date
    df["day_of_week"] = df["datetime_utc"].dt.day_name()

    # --------------------------
    # Deduplicate
    # --------------------------
    df = df.drop_duplicates(subset=["headline", "stock", "date"])

    # --------------------------
    # Publisher normalization
    # --------------------------
    def normalize_publisher(x: str):
        if pd.isna(x):
            return None
        x = x.strip().lower()
        if "@" in x:
            x = x.split("@")[-1]
        x = re.sub(r"https?://", "", x)
        x = x.split("/")[0]
        x = re.sub(r"[^a-z0-9. ]+", "", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x or None

    df["publisher_raw"] = df["publisher"].astype(str)
    df["publisher_norm"] = (
        df["publisher_raw"].apply(normalize_publisher).fillna("unknown")
    )

    # assign stable publisher IDs by frequency
    freq_order = df["publisher_norm"].value_counts().index.tolist()
    publisher_map = {p: f"P{idx:04d}" for idx, p in enumerate(freq_order, 1)}
    df["publisher_id"] = df["publisher_norm"].map(publisher_map)

    # --------------------------
    # Stock ticker validation
    # --------------------------
    def clean_ticker(x: str):
        if pd.isna(x):
            return None
        x = x.upper().strip()
        x = re.sub(r"[^A-Z0-9]", "", x)
        return x or None

    df["stock"] = df["stock"].apply(clean_ticker)
    df = df[df["stock"].notna()]

    # --------------------------
    # Headline text cleaning
    # --------------------------
    def clean_headline(x: str):
        if pd.isna(x):
            return None
        x = x.strip()
        x = re.sub(r"<.*?>", "", x)  # remove HTML tags
        x = re.sub(r"\s+", " ", x)
        return x or None

    df["headline_clean"] = df["headline"].apply(clean_headline)
    df = df[df["headline_clean"].notna()]

    # --------------------------
    # Basic text metrics
    # --------------------------
    df["headline_char_count"] = df["headline_clean"].str.len()
    df["headline_word_count"] = df["headline_clean"].str.split().str.len()

    # --------------------------
    # Persist artifacts as CSV
    # --------------------------
    output_dir = Path("datasets/preprocess_cleaner")
    output_dir.mkdir(parents=True, exist_ok=True)

    # cleaned dataset
    df.to_csv(output_dir / "financial_news_cleaned.csv", index=False)

    # publisher mapping
    mapping = (
        df[["publisher_raw", "publisher_norm", "publisher_id"]]
        .drop_duplicates()
        .sort_values("publisher_id")
    )
    mapping.to_csv(output_dir / "publisher_mapping.csv", index=False)

    # publisher summary
    summary = (
        df["publisher_norm"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "publisher_norm", "publisher_norm": "count"})
    )
    summary.to_csv(output_dir / "publisher_summary.csv", index=False)

    # --------------------------
    # Return summary dict
    # --------------------------
    return {
        "rows": df.shape[0],
        "unique_publishers": df["publisher_norm"].nunique(),
        "artifact_dir": str(output_dir),
    }
