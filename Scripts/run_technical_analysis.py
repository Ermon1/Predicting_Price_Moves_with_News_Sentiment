#!/usr/bin/env python3
"""
Script to run technical analysis
"""

import argparse
import logging
import os
import sys
from datetime import datetime

# Import the technical analysis library
from src.techinical.technical_analysis import TechnicalAnalyzer, DataFetcher, AnalysisPlotter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run Technical Analysis on Stocks')
    parser.add_argument('--stocks', nargs='+', 
                       default=['AAPL', 'AMZN', 'GOOG', 'META', 'MSFT', 'NVDA'],
                       help='List of stock symbols to analyze')
    parser.add_argument('--period', default='1y', 
                       help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--output-dir', default='technical_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--use-custom-data', action='store_true',
                       help='Use custom data from dataframe')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Initialize components
        data_fetcher = DataFetcher()
        analyzer = TechnicalAnalyzer()
        plotter = AnalysisPlotter()
        
        # Fetch or prepare data
        if args.use_custom_data:
            # Example: Load your custom data here
            # from src.data.data_loader import data_loader
            # from src.utility.configLoader import config_loader
            # df = data_loader.load_tabular_data(your_data_path)
            # stocks_data = data_fetcher.prepare_custom_data(df, 'stock')
            # analysis_results = analyzer.analyze_stocks(stocks_data)
            
            logger.info("Custom data mode selected but not implemented in this example")
            return
        else:
            # Fetch data from Yahoo Finance
            logger.info(f"Fetching data for {len(args.stocks)} stocks...")
            stocks_data = data_fetcher.fetch_stock_data(args.stocks, args.period)
        
        if not stocks_data:
            logger.error("No data fetched. Exiting.")
            return
        
        # Perform technical analysis
        logger.info("Performing technical analysis...")
        analysis_results = analyzer.analyze_stocks(stocks_data)
        
        if not analysis_results:
            logger.error("No analysis results generated.")
            return
        
        # Generate portfolio summary
        logger.info("Generating portfolio summary...")
        portfolio_summary = analyzer.get_portfolio_summary()
        
        # Save results
        summary_file = os.path.join(args.output_dir, f'portfolio_summary_{timestamp}.csv')
        portfolio_summary.to_csv(summary_file, index=False)
        logger.info(f"Portfolio summary saved to: {summary_file}")
        
        # Save individual stock analysis
        for symbol, data in analysis_results.items():
            stock_file = os.path.join(args.output_dir, f'{symbol}_analysis_{timestamp}.csv')
            data.to_csv(stock_file)
            logger.info(f"Analysis for {symbol} saved to: {stock_file}")
        
        # Generate plots if requested
        if args.generate_plots:
            logger.info("Generating visualizations...")
            
            # Portfolio heatmap
            heatmap_file = os.path.join(args.output_dir, f'portfolio_heatmap_{timestamp}.png')
            plotter.plot_portfolio_heatmap(portfolio_summary, heatmap_file)
            logger.info(f"Portfolio heatmap saved to: {heatmap_file}")
            
            # Individual stock plots
            for symbol in analysis_results.keys():
                try:
                    plot_file = os.path.join(args.output_dir, f'{symbol}_analysis_{timestamp}.png')
                    signals = analyzer.generate_signals(symbol)
                    plotter.plot_stock_analysis(signals, symbol, plot_file)
                    logger.info(f"Plot for {symbol} saved to: {plot_file}")
                except Exception as e:
                    logger.error(f"Error generating plot for {symbol}: {e}")
        
        # Print results to console
        print("\n" + "="*70)
        print("TECHNICAL ANALYSIS RESULTS")
        print("="*70)
        print(portfolio_summary.to_string(index=False))
        
        # Trading recommendations
        print("\n" + "="*70)
        print("TRADING RECOMMENDATIONS")
        print("="*70)
        
        oversold = portfolio_summary[portfolio_summary['RSI_Condition'] == 'OVERSOLD']
        overbought = portfolio_summary[portfolio_summary['RSI_Condition'] == 'OVERBOUGHT']
        
        if not oversold.empty:
            print("\n📈 POTENTIAL BUY OPPORTUNITIES (Oversold):")
            for _, stock in oversold.iterrows():
                print(f"   {stock['Symbol']}: RSI {stock['RSI']:.1f} | Trend: {stock['Trend']} | MACD: {stock['MACD_Condition']}")
        
        if not overbought.empty:
            print("\n📉 POTENTIAL SELL OPPORTUNITIES (Overbought):")
            for _, stock in overbought.iterrows():
                print(f"   {stock['Symbol']}: RSI {stock['RSI']:.1f} | Trend: {stock['Trend']} | MACD: {stock['MACD_Condition']}")
        
        if oversold.empty and overbought.empty:
            print("\n⚡ No strong buy/sell signals detected. Market appears neutral.")
        
        logger.info("Technical analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        raise

if __name__ == "__main__":
    main()