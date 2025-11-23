#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.temporal_analysis import run_temporal_analysis

if __name__ == "__main__":
    results = run_temporal_analysis()
    print("📅 Temporal Analysis:")
    print(f"   Busiest Hour: {results['time_patterns']['busiest_hour']}:00")
    print(f"   Busiest Day: {results['time_patterns']['busiest_day']}")
    print(f"   Daily Avg: {results['publication_frequency']['daily_mean']:.1f}")