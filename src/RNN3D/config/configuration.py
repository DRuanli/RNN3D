# src/RNN3D/config/configuration.py
from src.RNN3D.constants import *
from src.RNN3D.utils.common import read_yaml, create_directories
from src.RNN3D.entity.config_entity import (
    DataIngestionConfig,
    DataPreparationConfig,
    ModelConfig,
    SubmissionValidationConfig,
    VisualizationConfig,
    WebInterfaceConfig
)
import logging


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
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

    def get_model_config(self) -> ModelConfig:
        config = self.config.model
        params = self.params

        create_directories([config.root_dir, config.model_dir, config.output_dir])

        model_config = ModelConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            pretrained_structures_path=Path(config.pretrained_structures_path),
            output_dir=Path(config.output_dir),
            num_conformations=params.num_conformations,
            max_sequence_length=params.max_sequence_length
        )

        return model_config

    def get_submission_validation_config(self) -> SubmissionValidationConfig:
        config = self.config.model  # Using model config paths since validation works with model outputs

        # Create the directories if they don't exist
        validation_dir = Path(config.output_dir) / "validation"
        create_directories([validation_dir])

        submission_validation_config = SubmissionValidationConfig(
            root_dir=validation_dir,
            submission_path=Path(config.output_dir) / "submission.csv",
            template_path=Path("submission.csv") if Path("submission.csv").exists() else None,
            metrics_path=validation_dir / "metrics.txt",
            report_path=validation_dir / "performance_report.png",
            generate_report=True
        )

        return submission_validation_config

    def get_visualization_config(self) -> VisualizationConfig:
        config = self.config.visualization if hasattr(self.config, 'visualization') else self.config.model
        params = self.params

        # Create the directories if they don't exist
        vis_dir = Path(config.root_dir) / "visualizations"
        create_directories([vis_dir])

        visualization_config = VisualizationConfig(
            root_dir=Path(config.root_dir),
            submission_path=Path(self.config.model.output_dir) / "submission.csv",
            visualizations_dir=vis_dir,
            num_conformations=params.num_conformations
        )

        return visualization_config

    def get_web_interface_config(self) -> WebInterfaceConfig:
        web_config = self.config.web_interface if hasattr(self.config, 'web_interface') else None
        params = self.params

        # Create the directories if they don't exist
        templates_dir = Path("templates")
        static_dir = Path("static")
        create_directories([templates_dir, static_dir])

        # Get validation metrics path
        validation_dir = Path(self.config.model.output_dir) / "validation"
        metrics_path = validation_dir / "metrics.txt"

        # Get visualizations directory
        vis_dir = self.get_visualization_config().visualizations_dir

        # Determine web interface root directory
        if hasattr(self.config, 'artifacts_root'):
            web_root = Path(self.config.artifacts_root) / "web_interface"
        else:
            web_root = Path("artifacts/web_interface")

        # Get host and port from config if available
        host = web_config.host if web_config and hasattr(web_config, 'host') else "0.0.0.0"
        port = web_config.port if web_config and hasattr(web_config, 'port') else 5000
        debug_mode = web_config.debug_mode if web_config and hasattr(web_config, 'debug_mode') else False
        run_server = web_config.run_server if web_config and hasattr(web_config, 'run_server') else True

        web_interface_config = WebInterfaceConfig(
            root_dir=web_root,
            templates_dir=templates_dir,
            static_dir=static_dir,
            visualizations_dir=vis_dir,
            metrics_path=metrics_path,
            host=host,
            port=port,
            debug_mode=debug_mode,
            run_server=run_server,
            max_sequence_length=params.max_sequence_length
        )

        return web_interface_config