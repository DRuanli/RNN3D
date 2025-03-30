# src/RNN3D/pipeline/stage_02_data_preparation.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.data_preparation import DataPreparation
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/data_preparation.log"),
        logging.StreamHandler()
    ]
)

def main():
    try:
        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config=data_preparation_config)
        data_preparation.load_and_process_data()
        logging.info("Data preparation completed successfully")
    except Exception as e:
        logging.error(f"Error in data preparation: {e}")
        raise e

if __name__ == "__main__":
    main()