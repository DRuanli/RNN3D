import os
import pandas as pd
import numpy as np
import logging
import torch
from typing import Dict, List, Tuple
from pathlib import Path
from src.RNN3D.entity.config_entity import DataPreparationConfig


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def discover_data_paths(self) -> Dict[str, Path]:
        """
        Discovers the actual paths to data files based on the top-level data directory
        """
        data_paths = {}

        # Log all directories and files for debugging
        logging.info("Exploring directory structure:")
        for root, dirs, files in os.walk(self.config.data_dir):
            rel_path = os.path.relpath(root, self.config.data_dir)
            if rel_path == '.':
                rel_path = 'root'
            logging.info(f"Directory: {rel_path}")
            for file in files:
                logging.info(f"  - File: {file}")

        # Find CSV files in data directory, skip macOS metadata files
        for root, _, files in os.walk(self.config.data_dir):
            for file in files:
                # Skip macOS metadata files
                if file.startswith('._') or '__MACOSX' in root:
                    continue

                if file.endswith('.csv'):
                    if 'train' in file.lower() and 'sequences' in file.lower():
                        data_paths['train_sequences'] = Path(os.path.join(root, file))
                    elif 'train' in file.lower() and 'labels' in file.lower():
                        data_paths['train_labels'] = Path(os.path.join(root, file))
                    elif 'validation' in file.lower() and 'sequences' in file.lower():
                        data_paths['validation_sequences'] = Path(os.path.join(root, file))
                    elif 'validation' in file.lower() and 'labels' in file.lower():
                        data_paths['validation_labels'] = Path(os.path.join(root, file))
                    elif 'test' in file.lower() and 'sequences' in file.lower():
                        data_paths['test_sequences'] = Path(os.path.join(root, file))

        # Find MSA directory, skip macOS metadata directories
        for root, dirs, _ in os.walk(self.config.data_dir):
            if '__MACOSX' in root:
                continue

            if 'MSA' in dirs:
                data_paths['msa_dir'] = Path(os.path.join(root, 'MSA'))
                break

        logging.info(f"Discovered data paths: {data_paths}")
        return data_paths

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Encodes RNA sequence as one-hot vectors
        A -> [1,0,0,0]
        C -> [0,1,0,0]
        G -> [0,0,1,0]
        U -> [0,0,0,1]
        """
        encoding_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        encoded = np.zeros((len(sequence), 4), dtype=np.float32)

        for i, nucleotide in enumerate(sequence):
            if nucleotide in encoding_map:
                encoded[i, encoding_map[nucleotide]] = 1.0

        return encoded

    def parse_msa_file(self, target_id: str) -> np.ndarray:
        """
        Reads and parses MSA file for a given target
        """
        msa_paths = list(self.config.msa_dir.glob(f"*{target_id}*.fasta")) + list(
            self.config.msa_dir.glob(f"*{target_id}*.MSA.fasta"))

        if not msa_paths:
            logging.warning(f"No MSA file found for {target_id}")
            return None

        msa_path = msa_paths[0]
        logging.info(f"Using MSA file: {msa_path}")

        # Read MSA file (in FASTA format)
        sequences = []
        current_seq = ""

        with open(msa_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line

            if current_seq:
                sequences.append(current_seq)

        if not sequences:
            logging.warning(f"No sequences found in MSA file for {target_id}")
            return None

        # Convert sequences to numpy array
        msa_data = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)

        for i, seq in enumerate(sequences):
            msa_data[i, :len(seq)] = self.encode_sequence(seq)

        return msa_data

    def load_and_process_data(self) -> None:
        """
        Loads and processes the RNA data
        """
        # Discover file paths
        data_paths = self.discover_data_paths()

        # Create processed data directory
        os.makedirs(self.config.processed_data_dir, exist_ok=True)

        # Check if any data paths were found
        if not data_paths:
            logging.error("No data files found. Check the data directory structure.")
            return

        # Process training data
        if 'train_sequences' in data_paths and 'train_labels' in data_paths:
            self._process_dataset(
                data_paths['train_sequences'],
                data_paths['train_labels'],
                self.config.train_data_path,
                'training'
            )
        else:
            logging.warning("Training data files not found. Skipping training data processing.")

        # Process validation data
        if 'validation_sequences' in data_paths and 'validation_labels' in data_paths:
            self._process_dataset(
                data_paths['validation_sequences'],
                data_paths['validation_labels'],
                self.config.validation_data_path,
                'validation'
            )
        else:
            logging.warning("Validation data files not found. Skipping validation data processing.")

    def _process_dataset(self, sequences_path: Path, labels_path: Path, output_path: Path, dataset_type: str) -> None:
        """
        Processes a dataset and saves the processed data
        """
        try:
            # Load data with explicit encoding and error handling
            try:
                sequences_df = pd.read_csv(sequences_path, encoding='utf-8')
            except UnicodeDecodeError:
                logging.info(f"UTF-8 encoding failed, trying latin1 for {sequences_path}")
                sequences_df = pd.read_csv(sequences_path, encoding='latin1')

            try:
                labels_df = pd.read_csv(labels_path, encoding='utf-8')
            except UnicodeDecodeError:
                logging.info(f"UTF-8 encoding failed, trying latin1 for {labels_path}")
                labels_df = pd.read_csv(labels_path, encoding='latin1')

            logging.info(f"Processing {dataset_type} dataset with {len(sequences_df)} sequences")

            # Process each RNA sequence
            processed_data = []

            for idx, row in sequences_df.iterrows():
                target_id = row['target_id']
                sequence = row['sequence']

                # Skip sequences longer than max length
                if len(sequence) > self.config.max_sequence_length:
                    logging.warning(
                        f"Skipping {target_id} - sequence length {len(sequence)} exceeds max length {self.config.max_sequence_length}")
                    continue

                # Get labels for this sequence
                target_labels = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]

                if len(target_labels) == 0:
                    logging.warning(f"No labels found for {target_id}")
                    continue

                # Get coordinates
                coords = []

                # Check if we have multiple conformations in the labels
                conf_columns = [col for col in target_labels.columns if
                                col.startswith('x_') or col.startswith('y_') or col.startswith('z_')]
                num_confs = len(conf_columns) // 3

                for i in range(min(num_confs, self.config.num_conformations)):
                    x_col = f'x_{i + 1}'
                    y_col = f'y_{i + 1}'
                    z_col = f'z_{i + 1}'

                    if x_col in target_labels.columns and y_col in target_labels.columns and z_col in target_labels.columns:
                        conf_coords = np.zeros((len(sequence), 3), dtype=np.float32)
                        for j, label_row in target_labels.iterrows():
                            residue_idx = int(label_row['resid']) - 1  # 0-indexed
                            if residue_idx < len(sequence):
                                conf_coords[residue_idx] = [label_row[x_col], label_row[y_col], label_row[z_col]]

                        coords.append(conf_coords)

                # Get MSA data if available
                msa_data = self.parse_msa_file(target_id)

                # Create data entry
                entry = {
                    'target_id': target_id,
                    'sequence': sequence,
                    'encoded_sequence': self.encode_sequence(sequence),
                    'coordinates': coords,
                    'msa_data': msa_data
                }

                processed_data.append(entry)

                if (idx + 1) % 50 == 0:
                    logging.info(f"Processed {idx + 1}/{len(sequences_df)} sequences")

            # Save processed data
            logging.info(f"Saving {len(processed_data)} processed sequences to {output_path}")
            torch.save(processed_data, output_path)

        except Exception as e:
            logging.error(f"Error processing {dataset_type} dataset: {e}")
            raise e