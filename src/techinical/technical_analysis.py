"""
Technical Analysis Library
Complete OOP implementation for technical analysis using TA-Lib
"""

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
import yfinance as yf
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators using TA-Lib"""
    
    def __init__(self):
        self.indicators_calculated = 0
        logger.info("TechnicalIndicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for OHLCV data"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        result_df = df.copy()
        
        # Trend indicators
        result_df = self._calculate_trend_indicators(result_df)
        # Momentum indicators
        result_df = self._calculate_momentum_indicators(result_df)
        # Volatility indicators
        result_df = self._calculate_volatility_indicators(result_df)
        # Volume indicators
        result_df = self._calculate_volume_indicators(result_df)
        
        self.indicators_calculated = len([col for col in result_df.columns if col not in df.columns])
        logger.info(f"Calculated {self.indicators_calculated} technical indicators")
        
        return result_df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-following indicators"""
        # Moving Averages
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # Exponential Moving Averages
        df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
        df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # RSI
        df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
            df['High'], df['Low'], df['Close'],
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # CCI
        df['CCI_14'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Average True Range
        df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Volume SMA
        df['Volume_SMA_20'] = talib.SMA(df['Volume'], timeperiod=20)
        
        return df

class DataFetcher:
    """Fetch and prepare stock data"""
    
    def __init__(self):
        logger.info("DataFetcher initialized")
    
    def fetch_stock_data(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch historical stock data from Yahoo Finance"""
        stocks_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Ensure required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    logger.warning(f"Incomplete data for {symbol}")
                    continue
                
                stocks_data[symbol] = data
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        return stocks_data
    
    def prepare_custom_data(self, df: pd.DataFrame, symbol_column: str = 'stock') -> Dict[str, pd.DataFrame]:
        """Prepare custom dataframe for technical analysis"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if symbol_column not in df.columns:
            raise ValueError(f"Symbol column '{symbol_column}' not found in DataFrame")
        
        stocks_data = {}
        symbols = df[symbol_column].unique()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        for symbol in symbols:
            try:
                symbol_data = df[df[symbol_column] == symbol].copy()
                
                # Ensure data is sorted by date
                date_columns = ['Date', 'datetime', 'date']
                for date_col in date_columns:
                    if date_col in symbol_data.columns:
                        symbol_data = symbol_data.sort_values(date_col)
                        symbol_data = symbol_data.set_index(date_col)
                        break
                
                stocks_data[symbol] = symbol_data
                logger.info(f"Prepared data for {symbol}: {len(symbol_data)} records")
                
            except Exception as e:
                logger.error(f"Error preparing data for {symbol}: {e}")
                continue
        
        return stocks_data

class TechnicalAnalyzer:
    """Main technical analysis engine"""
    
    def __init__(self):
        self.indicators_calculator = TechnicalIndicators()
        self.analysis_results = {}
        self.stocks_data = {}
        logger.info("TechnicalAnalyzer initialized")
    
    def analyze_stocks(self, stocks_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Perform technical analysis on multiple stocks"""
        self.stocks_data = stocks_data
        self.analysis_results = {}
        
        for symbol, data in stocks_data.items():
            try:
                logger.info(f"Analyzing {symbol}")
                analyzed_data = self.indicators_calculator.calculate_all_indicators(data)
                analyzed_data['Symbol'] = symbol
                self.analysis_results[symbol] = analyzed_data
                logger.info(f"Successfully analyzed {symbol}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        return self.analysis_results
    
    def analyze_dataframe(self, df: pd.DataFrame, symbol_column: str = 'stock') -> Dict[str, pd.DataFrame]:
        """Analyze custom dataframe with multiple stocks"""
        data_fetcher = DataFetcher()
        stocks_data = data_fetcher.prepare_custom_data(df, symbol_column)
        return self.analyze_stocks(stocks_data)
    
    def generate_signals(self, symbol: str) -> pd.DataFrame:
        """Generate trading signals for a specific stock"""
        if symbol not in self.analysis_results:
            raise ValueError(f"No analysis data found for {symbol}")
        
        df = self.analysis_results[symbol].copy()
        
        # RSI Signals
        df['RSI_Overbought'] = df['RSI_14'] > 70
        df['RSI_Oversold'] = df['RSI_14'] < 30
        
        # MACD Signals
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'] > 0)
        df['MACD_Bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'] < 0)
        
        # Moving Average Signals
        df['MA_Bullish'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200'])
        df['MA_Bearish'] = (df['SMA_20'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])
        
        # Bollinger Band Signals
        df['BB_Upper_Touch'] = df['Close'] >= df['BB_Upper']
        df['BB_Lower_Touch'] = df['Close'] <= df['BB_Lower']
        
        # Combined Signals
        df['Buy_Signal'] = df['RSI_Oversold'] & df['MACD_Bullish'] & df['BB_Lower_Touch']
        df['Sell_Signal'] = df['RSI_Overbought'] & df['MACD_Bearish'] & df['BB_Upper_Touch']
        
        return df
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Get summary of technical signals for all analyzed stocks"""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_stocks() first.")
        
        summary_data = []
        
        for symbol, data in self.analysis_results.items():
            latest = data.iloc[-1]
            
            # Determine trend
            if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
                trend = "STRONG UPTREND"
            elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
                trend = "STRONG DOWNTREND"
            else:
                trend = "MIXED/TRANSITION"
            
            # RSI condition
            if latest['RSI_14'] > 70:
                rsi_condition = "OVERBOUGHT"
            elif latest['RSI_14'] < 30:
                rsi_condition = "OVERSOLD"
            else:
                rsi_condition = "NEUTRAL"
            
            # MACD condition
            macd_condition = "BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH"
            
            summary_data.append({
                'Symbol': symbol,
                'Price': latest['Close'],
                'RSI': latest['RSI_14'],
                'RSI_Condition': rsi_condition,
                'MACD': latest['MACD'],
                'MACD_Condition': macd_condition,
                'Trend': trend,
                'Volatility': latest['ATR_14'] / latest['Close'],
                'Volume_Ratio': latest['Volume'] / latest['Volume_SMA_20']
            })
        
        return pd.DataFrame(summary_data)
    
    def get_stock_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get analysis results for a specific stock"""
        return self.analysis_results.get(symbol)
    
    def get_available_symbols(self) -> List[str]:
        """Get list of analyzed stock symbols"""
        return list(self.analysis_results.keys())

class AnalysisPlotter:
    """Create visualizations for technical analysis results"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        self.style = style
        plt.style.use(self.style)
        logger.info("AnalysisPlotter initialized")
    
    def plot_stock_analysis(self, analysis_data: pd.DataFrame, symbol: str, save_path: Optional[str] = None):
        """Create comprehensive technical analysis plot for a single stock"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), height_ratios=[3, 1, 1, 1])
        fig.suptitle(f'Technical Analysis - {symbol}', fontsize=16, fontweight='bold')
        
        # Price and Moving Averages
        axes[0].plot(analysis_data.index, analysis_data['Close'], label='Close Price', linewidth=1, color='black')
        axes[0].plot(analysis_data.index, analysis_data['SMA_20'], label='SMA 20', linewidth=1, alpha=0.7)
        axes[0].plot(analysis_data.index, analysis_data['SMA_50'], label='SMA 50', linewidth=1, alpha=0.7)
        axes[0].fill_between(analysis_data.index, analysis_data['BB_Upper'], analysis_data['BB_Lower'], 
                           alpha=0.2, label='Bollinger Bands', color='gray')
        axes[0].set_title(f'{symbol} - Price and Indicators')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        axes[1].plot(analysis_data.index, analysis_data['RSI_14'], label='RSI 14', linewidth=1, color='purple')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[1].fill_between(analysis_data.index, 70, analysis_data['RSI_14'], 
                           where=(analysis_data['RSI_14'] >= 70), alpha=0.3, color='red')
        axes[1].fill_between(analysis_data.index, 30, analysis_data['RSI_14'], 
                           where=(analysis_data['RSI_14'] <= 30), alpha=0.3, color='green')
        axes[1].set_ylim(0, 100)
        axes[1].set_title('RSI Indicator')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MACD
        axes[2].plot(analysis_data.index, analysis_data['MACD'], label='MACD', linewidth=1, color='blue')
        axes[2].plot(analysis_data.index, analysis_data['MACD_Signal'], label='Signal Line', linewidth=1, color='red')
        axes[2].bar(analysis_data.index, analysis_data['MACD_Histogram'], label='Histogram', alpha=0.3, color='gray')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Volume
        axes[3].bar(analysis_data.index, analysis_data['Volume'], alpha=0.7, color='blue', label='Volume')
        axes[3].plot(analysis_data.index, analysis_data['OBV'] / 1000000, label='OBV (millions)', 
                   color='orange', linewidth=1)
        axes[3].set_title('Volume and OBV')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_portfolio_heatmap(self, portfolio_summary: pd.DataFrame, save_path: Optional[str] = None):
        """Create a heatmap of RSI values across the portfolio"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        symbols = portfolio_summary['Symbol'].tolist()
        rsi_values = portfolio_summary['RSI'].tolist()
        
        colors = []
        for rsi in rsi_values:
            if rsi > 70:
                colors.append('red')
            elif rsi < 30:
                colors.append('green')
            else:
                colors.append('yellow')
        
        bars = ax.bar(symbols, rsi_values, color=colors, alpha=0.7)
        
        for bar, value in zip(bars, rsi_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{value:.1f}', ha='center', va='bottom')
        
        ax.axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
        ax.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
        ax.set_ylim(0, 100)
        ax.set_ylabel('RSI Value')
        ax.set_title('Portfolio RSI Heatmap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()