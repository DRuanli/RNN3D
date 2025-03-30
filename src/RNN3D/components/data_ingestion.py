# src/RNN3D/components/data_ingestion.py
import os
import zipfile
import gdown
import pandas as pd
import logging
from src.RNN3D.utils.common import get_size
from src.RNN3D.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        '''
        Fetch data from the Google Drive URL
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?export=download&id='
            gdown.download(prefix + file_id, str(zip_download_dir))

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            logging.info(f"File size: {get_size(zip_download_dir)}")

        except Exception as e:
            logging.error(f"Error downloading file: {e}")
            raise e

    def extract_zip_file(self) -> None:
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logging.info(f"Extracted data at {unzip_path}")

    def validate_data(self) -> None:
        """
        Validates that the data directory exists and lists its contents
        """
        try:
            unzip_path = self.config.unzip_dir

            if not os.path.exists(unzip_path):
                raise FileNotFoundError(f"Unzipped directory not found at {unzip_path}")

            # List the contents of the directory
            logging.info(f"Listing contents of {unzip_path}:")
            all_files = []
            for root, dirs, files in os.walk(unzip_path):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), unzip_path)
                    all_files.append(relative_path)

            # Log first 10 files to get an idea of the structure
            for file in all_files[:10]:
                logging.info(f"  - {file}")

            if len(all_files) > 10:
                logging.info(f"  - ... and {len(all_files) - 10} more files")

            logging.info(f"Data extraction successful. Found {len(all_files)} files.")

        except Exception as e:
            logging.error(f"Data validation failed: {e}")
            raise e
