from src.analysis.time_series_analyzer import run_temporal_analysis

if __name__ == "__main__":
    results = run_temporal_analysis()
    print("📅 Temporal Analysis:")
    print(f"   Busiest Hour: {results['time_patterns']['busiest_hour']}:00")
    print(f"   Busiest Day: {results['time_patterns']['busiest_day']}")
    print(f"   Daily Avg: {results['publication_frequency']['daily_mean']:.1f}")
    print(f"   Valid Records: {results['data_quality']['valid_records']:,}")
    print(f"   Invalid Dates Dropped: {results['data_quality']['invalid_dates_dropped']:,}")