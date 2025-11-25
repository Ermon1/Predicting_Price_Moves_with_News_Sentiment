"""
Sentiment-Stock Correlation Analysis Core Engine
Task 3: Correlation between news and stock movement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from textblob import TextBlob
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class SentimentCorrelationEngine:
    """
    Complete pipeline for analyzing correlation between news sentiment and stock returns
    Uses project's data_loader and config_loader
    """

    def __init__(self, config_loader, data_loader, project_root):
        self.config_loader = config_loader
        self.data_loader = data_loader
        self.project_root = project_root

        self.sentiment_data = None
        self.stock_data = None
        self.merged_data = None
        self.correlation_results = None

        # Load analysis configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load analysis configuration from project config"""
        try:
            self.config = self.config_loader.load_config("sentiment_analysis")
            logger.info("✅ Loaded sentiment analysis configuration")
        except:
            # Default configuration if config not found
            self.config = {
                "correlation": {
                    "min_sample_size": 3,
                    "significance_level": 0.05,
                    "methods": ["pearson", "spearman"],
                },
                "sentiment": {"positive_threshold": 0.1, "negative_threshold": -0.1},
                "stock_data": {"default_period": "2y"},
            }
            logger.info("✅ Using default configuration")

    def load_news_data(self) -> pd.DataFrame:
        """
        Load news data using project's data_loader and configuration

        Returns:
            pd.DataFrame: News data with required columns
        """
        try:
            # Load data configuration
            data_config = self.config_loader.load_config("data_sources")
            inputs_config = data_config["data_sources"]["inputs"]
            financial_news_path = inputs_config["financial_news"]

            # Resolve absolute path
            root_financial_news_path = self.project_root / financial_news_path

            logger.info(f"📊 Loading news data from: {root_financial_news_path}")

            # Use project's data_loader
            news_df = self.data_loader.load_tabular_data(root_financial_news_path)

            logger.info(f"✅ News data loaded: {news_df.shape}")
            logger.info(f"   Columns: {news_df.columns.tolist()}")

            return news_df

        except Exception as e:
            logger.error(f"❌ Error loading news data: {e}")
            raise

    def analyze_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sentiment analysis on news headlines

        Args:
            news_df: DataFrame with news data

        Returns:
            DataFrame with sentiment scores
        """
        logger.info("🔍 Performing sentiment analysis...")

        # Create working copy
        sentiment_df = news_df.copy()

        # Ensure date format
        if "date" in sentiment_df.columns:
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.normalize()

        # Apply sentiment analysis
        sentiment_df["sentiment_score"] = sentiment_df["headline"].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0.0
        )

        # Categorize sentiment
        pos_threshold = self.config["sentiment"]["positive_threshold"]
        neg_threshold = self.config["sentiment"]["negative_threshold"]

        sentiment_df["sentiment_category"] = sentiment_df["sentiment_score"].apply(
            lambda x: (
                "positive"
                if x > pos_threshold
                else "negative" if x < neg_threshold else "neutral"
            )
        )

        # Aggregate daily sentiment
        if "stock" in sentiment_df.columns:
            daily_sentiment = (
                sentiment_df.groupby(["date", "stock"])
                .agg(
                    {
                        "sentiment_score": ["mean", "count", "std"],
                        "sentiment_category": lambda x: (
                            x.mode()[0] if len(x.mode()) > 0 else "neutral"
                        ),
                    }
                )
                .round(4)
            )
        else:
            daily_sentiment = (
                sentiment_df.groupby("date")
                .agg(
                    {
                        "sentiment_score": ["mean", "count", "std"],
                        "sentiment_category": lambda x: (
                            x.mode()[0] if len(x.mode()) > 0 else "neutral"
                        ),
                    }
                )
                .round(4)
            )

        # Flatten column names
        daily_sentiment.columns = [
            "avg_sentiment",
            "article_count",
            "sentiment_std",
            "dominant_sentiment",
        ]
        daily_sentiment = daily_sentiment.reset_index()

        logger.info(
            f"✅ Sentiment analysis completed: {len(daily_sentiment)} daily records"
        )

        # Print sentiment distribution
        sentiment_counts = daily_sentiment["dominant_sentiment"].value_counts()
        for sentiment, count in sentiment_counts.items():
            logger.info(
                f"   {sentiment}: {count} days ({count/len(daily_sentiment)*100:.1f}%)"
            )

        self.sentiment_data = daily_sentiment
        return daily_sentiment

    def load_stock_data(self, symbols: List[str], period: str = None) -> Dict:
        """
        Fetch stock data for analysis

        Args:
            symbols: List of stock symbols
            period: Data period (default from config)

        Returns:
            Dictionary of stock DataFrames
        """
        if period is None:
            period = self.config["stock_data"]["default_period"]

        logger.info(
            f"📈 Fetching stock data for {len(symbols)} symbols (period: {period})..."
        )

        stock_data = {}
        successful = 0

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)

                if not data.empty:
                    # Calculate daily returns
                    data["daily_return"] = data["Close"].pct_change() * 100
                    data = data.dropna(subset=["daily_return"])
                    data["date"] = data.index.normalize()

                    stock_data[symbol] = data
                    successful += 1
                    logger.info(f"   ✅ {symbol}: {len(data)} trading days")
                else:
                    logger.warning(f"   ⚠️  {symbol}: No data found")

            except Exception as e:
                logger.warning(f"   ❌ {symbol}: Error - {e}")

        logger.info(
            f"✅ Stock data loaded: {successful}/{len(symbols)} symbols successful"
        )
        self.stock_data = stock_data
        return stock_data

    def align_datasets(self) -> Dict:
        """
        Align sentiment and stock data by date and stock symbol

        Returns:
            Dictionary of merged DataFrames by symbol
        """
        logger.info("🔗 Aligning sentiment and stock datasets...")

        if self.sentiment_data is None:
            raise ValueError("Sentiment data not loaded. Run analyze_sentiment first.")
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Run load_stock_data first.")

        merged_data = {}
        min_sample_size = self.config["correlation"]["min_sample_size"]

        for symbol, stock_df in self.stock_data.items():
            # Filter sentiment data for this stock
            if "stock" in self.sentiment_data.columns:
                symbol_sentiment = self.sentiment_data[
                    self.sentiment_data["stock"] == symbol
                ]
            else:
                symbol_sentiment = self.sentiment_data

            if not symbol_sentiment.empty:
                # Merge on date
                merged_df = pd.merge(
                    symbol_sentiment,
                    stock_df[["date", "daily_return", "Close", "Volume"]],
                    on="date",
                    how="inner",
                )

                if len(merged_df) >= min_sample_size:
                    merged_data[symbol] = merged_df
                    logger.info(f"   ✅ {symbol}: {len(merged_df)} aligned records")
                else:
                    logger.info(
                        f"   ⚠️  {symbol}: Insufficient data ({len(merged_df)} < {min_sample_size})"
                    )
            else:
                logger.info(f"   ⚠️  {symbol}: No sentiment data")

        self.merged_data = merged_data
        logger.info(
            f"✅ Dataset alignment completed: {len(merged_data)} stocks with sufficient data"
        )
        return merged_data

    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calculate comprehensive correlation analysis

        Returns:
            DataFrame with correlation results
        """
        logger.info("📈 Calculating sentiment-return correlations...")

        if not self.merged_data:
            raise ValueError("No merged data available. Run align_datasets first.")

        correlation_results = []
        significance_level = self.config["correlation"]["significance_level"]

        for symbol, data in self.merged_data.items():
            try:
                # Calculate different correlation methods
                pearson_corr, pearson_p = pearsonr(
                    data["avg_sentiment"], data["daily_return"]
                )

                if "spearman" in self.config["correlation"]["methods"]:
                    spearman_corr, spearman_p = spearmanr(
                        data["avg_sentiment"], data["daily_return"]
                    )
                else:
                    spearman_corr, spearman_p = np.nan, np.nan

                # Comprehensive results
                results = {
                    "symbol": symbol,
                    "sample_size": len(data),
                    "pearson_correlation": pearson_corr,
                    "pearson_p_value": pearson_p,
                    "spearman_correlation": spearman_corr,
                    "spearman_p_value": spearman_p,
                    "significant": pearson_p < significance_level,
                    "mean_sentiment": data["avg_sentiment"].mean(),
                    "mean_return": data["daily_return"].mean(),
                    "sentiment_volatility": data["avg_sentiment"].std(),
                    "return_volatility": data["daily_return"].std(),
                    "date_range_days": (data["date"].max() - data["date"].min()).days,
                    "articles_per_day": data["article_count"].mean(),
                    "analysis_date": datetime.now(),
                }

                correlation_results.append(results)
                logger.info(
                    f"   ✅ {symbol}: r = {pearson_corr:.3f} (p = {pearson_p:.3f})"
                )

            except Exception as e:
                logger.error(f"   ❌ {symbol}: Correlation error - {e}")
                continue

        self.correlation_results = pd.DataFrame(correlation_results)
        logger.info(
            f"✅ Correlation analysis completed: {len(correlation_results)} stocks analyzed"
        )
        return self.correlation_results

    def perform_lagged_analysis(self, max_lag: int = 5) -> pd.DataFrame:
        """
        Analyze correlation with different time lags

        Args:
            max_lag: Maximum number of days to lag

        Returns:
            DataFrame with lagged correlation results
        """
        logger.info(
            f"⏰ Performing lagged correlation analysis (0 to {max_lag} days)..."
        )

        lag_results = []
        min_data_points = 5  # Minimum for lagged analysis

        for symbol, data in self.merged_data.items():
            if len(data) < min_data_points:
                continue

            data = data.sort_values("date")

            for lag in range(max_lag + 1):
                try:
                    if lag == 0:
                        sentiment_data = data["avg_sentiment"].values
                        return_data = data["daily_return"].values
                    else:
                        sentiment_data = data["avg_sentiment"].iloc[:-lag].values
                        return_data = data["daily_return"].iloc[lag:].values

                    if (
                        len(sentiment_data) >= min_data_points
                        and len(return_data) >= min_data_points
                        and np.std(sentiment_data) > 0
                        and np.std(return_data) > 0
                    ):

                        corr, p_value = pearsonr(sentiment_data, return_data)

                        lag_results.append(
                            {
                                "symbol": symbol,
                                "lag_days": lag,
                                "correlation": corr,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "sample_size": len(sentiment_data),
                            }
                        )

                except Exception as e:
                    continue

        results_df = pd.DataFrame(lag_results)
        logger.info(
            f"✅ Lagged analysis completed: {len(results_df)} lag correlations calculated"
        )
        return results_df

    def generate_visualizations(self, output_dir: str = None):
        """Generate comprehensive visualizations"""
        logger.info("📊 Generating correlation visualizations...")

        if self.correlation_results is None or self.correlation_results.empty:
            logger.warning("No correlation results available for visualization")
            return

        # Set style
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Correlation coefficients
        plt.subplot(2, 3, 1)
        results_sorted = self.correlation_results.sort_values("pearson_correlation")
        colors = [
            "red" if x < 0 else "green" for x in results_sorted["pearson_correlation"]
        ]
        bars = plt.barh(
            results_sorted["symbol"],
            results_sorted["pearson_correlation"],
            color=colors,
            alpha=0.7,
        )
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.5)
        plt.xlabel("Pearson Correlation Coefficient")
        plt.title("Sentiment-Return Correlation by Stock")
        plt.grid(True, alpha=0.3, axis="x")

        # Plot 2: Statistical significance
        plt.subplot(2, 3, 2)
        plt.bar(
            self.correlation_results["symbol"],
            -np.log10(self.correlation_results["pearson_p_value"] + 1e-10),
            color="orange",
            alpha=0.7,
        )
        plt.axhline(y=-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
        plt.ylabel("-log10(p-value)")
        plt.title("Statistical Significance")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Plot 3: Sample size vs correlation
        plt.subplot(2, 3, 3)
        plt.scatter(
            self.correlation_results["sample_size"],
            self.correlation_results["pearson_correlation"],
            s=100,
            alpha=0.6,
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Correlation Coefficient")
        plt.title("Sample Size vs Correlation Strength")
        plt.grid(True, alpha=0.3)

        # Plot 4: Sentiment distribution
        plt.subplot(2, 3, 4)
        all_sentiments = []
        for data in self.merged_data.values():
            all_sentiments.extend(data["avg_sentiment"].values)
        plt.hist(all_sentiments, bins=20, alpha=0.7, color="blue", edgecolor="black")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Average Daily Sentiment")
        plt.ylabel("Frequency")
        plt.title("Distribution of Sentiment Scores")
        plt.grid(True, alpha=0.3)

        # Plot 5: Returns distribution
        plt.subplot(2, 3, 5)
        all_returns = []
        for data in self.merged_data.values():
            all_returns.extend(data["daily_return"].values)
        plt.hist(all_returns, bins=20, alpha=0.7, color="green", edgecolor="black")
        plt.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Daily Return (%)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Daily Returns")
        plt.grid(True, alpha=0.3)

        # Plot 6: Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis("off")

        summary_text = self._get_summary_text()
        plt.text(
            0.1,
            0.9,
            summary_text,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
            linespacing=1.5,
        )

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(
                output_dir,
                f'correlation_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logger.info(f"✅ Visualization saved to: {plot_path}")

        plt.show()

    def _get_summary_text(self):
        """Generate summary text for visualization"""
        if self.correlation_results.empty:
            return "No correlation results available"

        total_stocks = len(self.correlation_results)
        significant = len(
            self.correlation_results[self.correlation_results["significant"]]
        )
        avg_correlation = self.correlation_results["pearson_correlation"].mean()

        return f"""
CORRELATION ANALYSIS SUMMARY

Total Stocks Analyzed: {total_stocks}
Significant Correlations: {significant}
Average Correlation: {avg_correlation:.4f}

Strongest Positive: {self.correlation_results.loc[self.correlation_results['pearson_correlation'].idxmax()]['symbol']}
Strongest Negative: {self.correlation_results.loc[self.correlation_results['pearson_correlation'].idxmin()]['symbol']}

Average Sample Size: {self.correlation_results['sample_size'].mean():.0f} days
Total Observations: {self.correlation_results['sample_size'].sum()} days
"""

    def get_stock_symbols_from_news(
        self, news_df: pd.DataFrame, max_symbols: int = 20
    ) -> List[str]:
        """
        Extract stock symbols from news data

        Args:
            news_df: News DataFrame
            max_symbols: Maximum number of symbols to return

        Returns:
            List of stock symbols
        """
        if "stock" in news_df.columns:
            symbols = news_df["stock"].value_counts().head(max_symbols).index.tolist()
            logger.info(f"📋 Found {len(symbols)} stocks in news data")
        else:
            # Default stocks if no stock column
            symbols = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
                "NVDA",
                "JPM",
                "JNJ",
                "V",
            ]
            logger.info(f"📋 Using default stocks: {len(symbols)} symbols")

        return symbols


def run_complete_analysis(
    config_loader, data_loader, project_root, period: str = None, max_symbols: int = 15
) -> SentimentCorrelationEngine:
    """
    Run complete sentiment-stock correlation analysis

    Args:
        config_loader: Project config loader
        data_loader: Project data loader
        project_root: Project root path
        period: Stock data period
        max_symbols: Maximum number of stocks to analyze

    Returns:
        Configured SentimentCorrelationEngine instance
    """
    logger.info("🚀 Starting complete sentiment-stock correlation analysis")

    # Initialize engine
    engine = SentimentCorrelationEngine(config_loader, data_loader, project_root)

    try:
        # Step 1: Load news data
        news_df = engine.load_news_data()

        # Step 2: Analyze sentiment
        engine.analyze_sentiment(news_df)

        # Step 3: Get stock symbols and load stock data
        symbols = engine.get_stock_symbols_from_news(news_df, max_symbols=max_symbols)
        engine.load_stock_data(symbols, period=period)

        # Step 4: Align datasets
        engine.align_datasets()

        # Step 5: Calculate correlations
        engine.calculate_correlations()

        logger.info("✅ Complete analysis finished successfully")
        return engine

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise
