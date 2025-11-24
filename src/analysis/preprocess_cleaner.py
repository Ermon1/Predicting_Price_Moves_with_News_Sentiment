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
      - persist artifacts as CSV for downstream tasks

    All paths and parameters are read from config/data_sources.yaml
    """

    # -------------------------
    # Load config
    # -------------------------
    config = config_loader.load_config("data_sources")["data_sources"]

    # -------------------------
    # Resolve paths
    # -------------------------
    input_path = Path(config["inputs"]["financial_news"])
    cleaned_csv_path = Path(config["inputs"]["preprocessed_news"])
    output_dir = cleaned_csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    publisher_mapping_path = output_dir / "publisher_mapping.csv"
    publisher_summary_path = output_dir / "publisher_summary.csv"

    # -------------------------
    # Columns
    # -------------------------
    date_column = config["parameters"]["task1"]["date_column"]
    text_columns = config["parameters"]["task1"]["text_columns"]
    categorical_columns = config["parameters"]["task1"]["categorical_columns"]

    # -------------------------
    # Load dataset
    # -------------------------
    df = data_loader.load_tabular_data(input_path)

    # -------------------------
    # Schema validation
    # -------------------------
    required_cols = set(text_columns + categorical_columns + [date_column])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"[Task1] Missing columns in input CSV: {missing}. "
            f"Check dataset OR config/data_sources.yaml"
        )

    # ======================================
    # 1. Timestamp normalization
    # ======================================
    df["datetime_utc"] = pd.to_datetime(df[date_column], utc=True, errors="coerce")
    df["date"] = df["datetime_utc"].dt.date
    df["day_of_week"] = df["datetime_utc"].dt.day_name()

    # rows with invalid date → useless for time series
    df = df[df["datetime_utc"].notna()]

    # ======================================
    # 2. Deduplicate across business keys
    # ======================================
    df = df.drop_duplicates(subset=["headline", "stock", "date"])

    # ======================================
    # 3. Publisher normalization
    # ======================================
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
    df["publisher_norm"] = df["publisher_raw"].apply(normalize_publisher)
    df["publisher_norm"] = df["publisher_norm"].fillna("unknown")

    # Stable IDs from frequency ranking
    freq = df["publisher_norm"].value_counts().index.tolist()
    publisher_map = {p: f"P{idx:04d}" for idx, p in enumerate(freq, 1)}
    df["publisher_id"] = df["publisher_norm"].map(publisher_map)

    # ======================================
    # 4. Clean ticker
    # ======================================
    def clean_ticker(x: str):
        if pd.isna(x):
            return None
        x = x.upper().strip()
        x = re.sub(r"[^A-Z0-9]", "", x)
        return x or None

    df["stock"] = df["stock"].apply(clean_ticker)
    df = df[df["stock"].notna()]

    # ======================================
    # 5. Headline cleaning
    # ======================================
    def clean_headline(x: str):
        if pd.isna(x):
            return None
        x = x.strip()
        x = re.sub(r"<.*?>", "", x)
        x = re.sub(r"\s+", " ", x)
        return x or None

    df["headline_clean"] = df["headline"].apply(clean_headline)
    df = df[df["headline_clean"].notna()]

    # ======================================
    # 6. Text features
    # ======================================
    df["headline_char_count"] = df["headline_clean"].str.len()
    df["headline_word_count"] = df["headline_clean"].str.split().str.len()

    # ======================================
    # 7. Save artifacts
    # ======================================
    df.to_csv(cleaned_csv_path, index=False)

    mapping = (
        df[["publisher_raw", "publisher_norm", "publisher_id"]]
        .drop_duplicates()
        .sort_values("publisher_id")
    )
    mapping.to_csv(publisher_mapping_path, index=False)

    summary = (
        df["publisher_norm"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "publisher_norm", "publisher_norm": "count"})
    )
    summary.to_csv(publisher_summary_path, index=False)

    # ======================================
    # 8. Return structured output
    # ======================================
    return {
        "rows": df.shape[0],
        "unique_publishers": df["publisher_norm"].nunique(),
        "artifact_dir": str(output_dir),
        "cleaned_csv": str(cleaned_csv_path),
    }
