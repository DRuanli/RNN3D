# src/RNN3D/entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    data_dir: Path
    processed_data_dir: Path
    max_sequence_length: int
    num_conformations: int
    train_data_path: Path
    validation_data_path: Path
    msa_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    root_dir: Path
    model_dir: Path
    pretrained_structures_path: Path
    output_dir: Path
    num_conformations: int
    max_sequence_length: int


@dataclass(frozen=True)
class SubmissionValidationConfig:
    root_dir: Path
    submission_path: Path
    template_path: Optional[Path]
    metrics_path: Path
    report_path: Path
    generate_report: bool


@dataclass(frozen=True)
class VisualizationConfig:
    root_dir: Path
    submission_path: Path
    visualizations_dir: Path
    num_conformations: int


@dataclass(frozen=True)
class WebInterfaceConfig:
    root_dir: Path
    templates_dir: Path
    static_dir: Path
    visualizations_dir: Path
    metrics_path: Path
    host: str
    port: int
    debug_mode: bool
    run_server: bool
    max_sequence_length: int