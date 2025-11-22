# test_import.py - Create this in project root and run it
from src.logger import get_logger

logger = get_logger(__file__)
logger.info("✅ Import works perfectly!")
print("SUCCESS! src imports work!")