#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.text_analysis import run_text_analysis

if __name__ == "__main__":
    results = run_text_analysis()
    print("📝 Text Analysis:")
    print(f"   Top Words: {list(results['common_keywords'].keys())[:3]}")
    print(f"   Financial Terms: {list(results['financial_terms'].keys())[:3]}")
    print(f"   FDA Mentions: {results['topic_patterns']['fda_mentions']}")