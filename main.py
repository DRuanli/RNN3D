# main.py
import logging
import os
from src.RNN3D.pipeline.stage_01_data_ingestion import main as data_ingestion_main

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/running_logs.log"),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    try:
        logging.info("\n\n>>>>> Stage 1: Data Ingestion Started <<<<<")
        data_ingestion_main()
        logging.info(">>>>> Stage 1: Data Ingestion Completed <<<<<\n\n")
    except Exception as e:
        logging.error(f">>>>> Stage 1: Data Ingestion Failed with error: {e} <<<<<")
        raise e