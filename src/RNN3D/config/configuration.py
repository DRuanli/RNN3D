# src/RNN3D/config/configuration.py
from src.RNN3D.constants import *
from src.RNN3D.utils.common import read_yaml, create_directories
from src.RNN3D.entity.config_entity import DataIngestionConfig, DataPreparationConfig
import logging

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

        return data_ingestion_config

    def get_data_preparation_config(self) -> DataPreparationConfig:
        config = self.config.data_preparation
        params = self.params

        create_directories([config.root_dir, config.processed_data_dir])

        data_preparation_config = DataPreparationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            processed_data_dir=Path(config.processed_data_dir),
            max_sequence_length=params.max_sequence_length,
            num_conformations=params.num_conformations,
            train_data_path=Path(config.train_data_path),
            validation_data_path=Path(config.validation_data_path),
            msa_dir=Path(config.msa_dir)
        )

        return data_preparation_config