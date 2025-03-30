# src/RNN3D/pipeline/stage_01_data_ingestion.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.data_ingestion import DataIngestion
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/data_ingestion.log"),
        logging.StreamHandler()
    ]
)

def main():
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.validate_data()
        logging.info("Data ingestion completed successfully")
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e

if __name__ == "__main__":
    main()