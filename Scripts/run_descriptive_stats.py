from src.analysis.descriptive_stats import run_descriptive_analysis

if __name__ == "__main__":
    results = run_descriptive_analysis()
    print("📊 Descriptive Stats:")
    print(f"   Articles: {results['date_stats']['total_articles']:,}")
    print(f"   Publishers: {results['publisher_stats']['total_publishers']}")
    print(f"   Avg Headline Length: {results['textual_stats']['headline_length']['mean']:.1f}")