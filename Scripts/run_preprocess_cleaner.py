#!/usr/bin/env python3
from src.analysis.preprocess_cleaner import run_task1_preprocessing
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    try:
        result = run_task1_preprocessing()

        print("\n🧹 Task-1 Preprocessing Complete")
        print(f"   Total rows after cleaning: {result['rows']}")
        print(f"   Unique Publishers:        {result['unique_publishers']}")
        print(f"   Artifacts saved at:       {result['artifact_dir']}")
        print(f"   Cleaned CSV:              {result['cleaned_csv']}")


    except Exception as e:
        print("\n❌ TASK FAILED")
        print(e)
