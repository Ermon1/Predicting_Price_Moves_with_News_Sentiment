#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.publisher_analysis import run_publisher_analysis

if __name__ == "__main__":
    results = run_publisher_analysis()
    print("🏢 Publisher Analysis:")
    print(f"   Total Publishers: {results['publisher_activity']['total_publishers']}")
    print(f"   Top 10 Concentration: {results['publisher_activity']['concentration']}")
    print(f"   Email Domains: {len(results['email_domains'])}")