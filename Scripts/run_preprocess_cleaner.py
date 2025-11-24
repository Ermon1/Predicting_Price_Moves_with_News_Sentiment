from src.analysis.preprocess_cleaner import run_task1_preprocessing

if __name__ == "__main__":
    results = run_task1_preprocessing()

    print("🧹 Task-1 Preprocessing Complete")
    print(f"   Total rows after cleaning: {results['rows']}")
    print(f"   Unique Publishers: {results['unique_publishers']}")
    print(f"   Artifacts saved at: {results['artifact_dir']}")

    # Optionally, show top 5 publishers by frequency
    import pandas as pd
    summary_path = f"{results['artifact_dir']}/publisher_summary.csv"
    summary_df = pd.read_csv(summary_path)

    print("\n🏢 Top 5 Publishers by Frequency:")
    for idx, row in summary_df.head(5).iterrows():
        print(f"   {row['publisher_norm']}: {row['count']} articles")
