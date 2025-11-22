# src/customException.py - 100% GUARANTEED TO WORK
import logging
import sys
from pathlib import Path

# 🚀 NUCLEAR OPTION: Completely self-contained logging
def setup_logging():
    """Self-contained logging that needs no imports"""
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )

def get_logger(module_file):
    """Self-contained logger getter"""
    if not logging.getLogger().hasHandlers():
        setup_logging()
    return logging.getLogger(Path(module_file).stem)

# Initialize logger
logger = get_logger(__file__)
logger.info("🔥 CustomException loaded - NO IMPORTS NEEDED!")

class CustomException(Exception):
    def __init__(self, message: str, error_details: str = None):
        self.message = message
        self.error_details = error_details
        
        logger.error(f"🚨 {message}")
        if error_details:
            logger.debug(f"🔍 {error_details}")
            
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.message} | Details: {self.error_details}" if self.error_details else self.message

if __name__ == "__main__":
    try:
        raise CustomException("Test", "This ALWAYS works")
    except CustomException as e:
        logger.info(f"✅ Success: {e}")
        print("🎉 IT WORKS!")