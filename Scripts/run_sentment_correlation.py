#!/usr/bin/env python3
"""
Simple Task 3 Runner - Uses the core SentimentCorrelationEngine
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import project modules
from src.data.data_loader import data_loader
from src.utility.configLoader import config_loader
from src.analysis.sentiment_correlation import SentimentCorrelationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Run the complete Task 3 analysis"""
    print("🚀 Starting Task 3: Sentiment-Stock Correlation Analysis")

    try:
        # Initialize with project dependencies
        engine = SentimentCorrelationEngine(config_loader, data_loader, project_root)

        # Run analysis step by step
        print("\n📊 Step 1: Loading news data...")
        news_df = engine.load_news_data()

        print("🔍 Step 2: Analyzing sentiment...")
        engine.analyze_sentiment(news_df)

        print("📈 Step 3: Getting stock symbols...")
        symbols = engine.get_stock_symbols_from_news(news_df)

        print("💹 Step 4: Loading stock data...")
        engine.load_stock_data(symbols)

        print("🔗 Step 5: Aligning datasets...")
        engine.align_datasets()

        print("📊 Step 6: Calculating correlations...")
        results = engine.calculate_correlations()

        # Display results
        print("\n" + "=" * 70)
        print("📋 CORRELATION RESULTS")
        print("=" * 70)

        if (
            engine.correlation_results is not None
            and not engine.correlation_results.empty
        ):
            # Sort by correlation strength
            sorted_results = engine.correlation_results.sort_values(
                "pearson_correlation", ascending=False
            )

            for _, row in sorted_results.iterrows():
                significance = "***" if row["significant_95"] else ""
                print(
                    f"{row['symbol']:6} | r = {row['pearson_correlation']:7.3f} | "
                    f"p = {row['pearson_p_value']:7.3f} | n = {row['sample_size']:3} {significance}"
                )

            # Summary statistics
            print("\n" + "=" * 70)
            print("📈 SUMMARY STATISTICS")
            print("=" * 70)
            total_stocks = len(engine.correlation_results)
            significant = len(
                engine.correlation_results[engine.correlation_results["significant_95"]]
            )
            avg_correlation = engine.correlation_results["pearson_correlation"].mean()

            print(f"Stocks analyzed: {total_stocks}")
            print(f"Significant correlations (p < 0.05): {significant}")
            print(f"Average correlation: {avg_correlation:.3f}")

            # Strongest correlations
            strongest_positive = engine.correlation_results.loc[
                engine.correlation_results["pearson_correlation"].idxmax()
            ]
            strongest_negative = engine.correlation_results.loc[
                engine.correlation_results["pearson_correlation"].idxmin()
            ]

            print(
                f"\nStrongest positive: {strongest_positive['symbol']} (r = {strongest_positive['pearson_correlation']:.3f})"
            )
            print(
                f"Strongest negative: {strongest_negative['symbol']} (r = {strongest_negative['pearson_correlation']:.3f})"
            )

        else:
            print("No correlation results available.")

        print("\n✅ Task 3 completed successfully!")

    except Exception as e:
        print(f"❌ Error in Task 3 analysis: {e}")
        raise


if __name__ == "__main__":
    main()
