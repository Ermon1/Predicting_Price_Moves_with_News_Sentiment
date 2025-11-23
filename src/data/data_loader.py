import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List

class DataLoader:
    """
    COMPLETELY GENERIC DataLoader - No hardcoded paths, no task knowledge
    No logging - just pure data operations
    """
    
    def load_tabular_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load any tabular data file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_path.suffix == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def load_time_series_data(self, file_path: str, date_column: str = 'date', 
                            index_date: bool = True, **kwargs) -> pd.DataFrame:
        """Load time series data with date handling"""
        df = self.load_tabular_data(file_path, **kwargs)
        
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            if index_date:
                df = df.set_index(date_column)
                df = df.sort_index()
        
        return df
    
    def load_stock_data(self, file_path: str, symbol: str = None, **kwargs) -> pd.DataFrame:
        """Load single stock data file"""
        if symbol is None:
            symbol = Path(file_path).stem
        return self.load_time_series_data(file_path, date_column='Date', index_date=True, **kwargs)
    
    def load_stocks_from_directory(self, directory_path: str, 
                                 file_pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """Load multiple stock files from directory"""
        directory = Path(directory_path)
        stock_data = {}
        
        for file_path in directory.glob(file_pattern):
            symbol = file_path.stem
            try:
                stock_data[symbol] = self.load_stock_data(str(file_path), symbol)
            except Exception:
                # Silently skip files that can't be loaded
                continue
        
        return stock_data
    
    def save_data(self, data: Union[pd.DataFrame, np.ndarray], 
                 file_path: str, **kwargs) -> None:
        """Save data to file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, **kwargs)

# Singleton instance
data_loader = DataLoader()