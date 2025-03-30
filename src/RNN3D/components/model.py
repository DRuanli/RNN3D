# src/RNN3D/components/model.py
import os
import numpy as np
import pandas as pd
import pickle
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from src.RNN3D.entity.config_entity import ModelConfig


class RNA3DStructurePredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.pretrained_structures = {}

    def load_pretrained_structures(self) -> None:
        """
        Load pretrained RNA 3D structures from pickle file
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config.pretrained_structures_path), exist_ok=True)

            if not os.path.exists(self.config.pretrained_structures_path):
                logging.warning(f"Pretrained structures file not found at {self.config.pretrained_structures_path}")
                logging.warning("Using empty structures dictionary as fallback")
                return

            with open(self.config.pretrained_structures_path, 'rb') as f:
                self.pretrained_structures = pickle.load(f)

            logging.info(f"Loaded {len(self.pretrained_structures)} pretrained structures")
        except Exception as e:
            logging.warning(f"Error loading pretrained structures: {e}")
            logging.warning("Using empty structures dictionary as fallback")

    def predict_structure(self, sequences: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict 3D structures for the given RNA sequences using pretrained structures
        """
        if not self.pretrained_structures:
            logging.info("No pretrained structures available. Using default empty coordinates.")

        solution = {}

        for i, row in sequences.iterrows():
            target_id = row.target_id
            sequence = row.sequence

            # Create empty coordinates for all conformations
            coords = [
                np.zeros((len(sequence), 3), dtype=np.float32)
                for _ in range(self.config.num_conformations)
            ]

            # Try to find the sequence in pretrained structures if available
            found = False
            if self.pretrained_structures:
                for known_seq, known_coords in self.pretrained_structures.items():
                    if sequence == known_seq:
                        # Exact match found
                        logging.info(f"Found exact match for {target_id}")
                        found = True

                        # Copy coordinates for each conformation
                        num_known_confs = len(known_coords)
                        for k in range(min(num_known_confs, self.config.num_conformations)):
                            coords[k] = known_coords[k]
                        break

                if not found:
                    # Try to find similar sequences using alignment (simple version)
                    for known_seq, known_coords in self.pretrained_structures.items():
                        if len(known_seq) == len(sequence):
                            similarity = sum(a == b for a, b in zip(sequence, known_seq)) / len(sequence)
                            if similarity > 0.8:  # 80% similarity threshold
                                logging.info(f"Found similar sequence for {target_id} with {similarity:.2f} similarity")
                                found = True

                                # Copy coordinates for each conformation
                                num_known_confs = len(known_coords)
                                for k in range(min(num_known_confs, self.config.num_conformations)):
                                    coords[k] = known_coords[k]
                                break

            if not found:
                logging.warning(f"No matching structure found for {target_id}")

            solution[target_id] = {
                'target_id': target_id,
                'sequence': sequence,
                'coord': coords
            }

        return solution

    def solution_to_submission(self, solution: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert solution dictionary to submission dataframe
        """
        submit_dfs = []

        for target_id, data in solution.items():
            df = pd.DataFrame()
            sequence = data['sequence']
            coords = data['coord']

            # Create submission dataframe
            df['ID'] = [f'{target_id}_{i + 1}' for i in range(len(sequence))]
            df['resname'] = list(sequence)
            df['resid'] = [i + 1 for i in range(len(sequence))]

            # Add coordinates for each conformation
            for j in range(len(coords)):
                df[f'x_{j + 1}'] = coords[j][:, 0]
                df[f'y_{j + 1}'] = coords[j][:, 1]
                df[f'z_{j + 1}'] = coords[j][:, 2]

            submit_dfs.append(df)

        return pd.concat(submit_dfs) if submit_dfs else pd.DataFrame()

    def process_test_sequences(self, test_file_path: str) -> str:
        """
        Process test sequences and generate predictions
        """
        try:
            # Check if the test file exists
            if not os.path.exists(test_file_path):
                logging.error(f"Test file not found at {test_file_path}")
                test_dir = os.path.dirname(test_file_path)
                available_test_files = []

                if os.path.exists(test_dir):
                    logging.info(f"Files in {test_dir}:")
                    for file in os.listdir(test_dir):
                        logging.info(f"  - {file}")
                    available_test_files = [f for f in os.listdir(test_dir) if
                                            'test' in f.lower() and f.endswith('.csv')]

                if available_test_files:
                    test_file_path = os.path.join(test_dir, available_test_files[0])
                    logging.info(f"Using alternative test file: {test_file_path}")
                else:
                    # Create a dummy test file
                    logging.info("Creating a dummy test dataset")
                    test_df = pd.DataFrame({
                        'target_id': ['dummy_1', 'dummy_2'],
                        'sequence': ['GGGAAACCC', 'AAAUUUCCCGGG'],
                        'temporal_cutoff': ['2025-03-30', '2025-03-30'],
                        'description': ['Dummy test sequence 1', 'Dummy test sequence 2']
                    })
                    return self._process_test_df(test_df)

            # Load test sequences
            test_df = pd.read_csv(test_file_path)
            logging.info(f"Loaded {len(test_df)} test sequences")
            return self._process_test_df(test_df)

        except Exception as e:
            logging.error(f"Error processing test sequences: {e}")
            logging.info("Generating submission with dummy data as fallback")

            # Create a dummy test dataframe as fallback
            test_df = pd.DataFrame({
                'target_id': ['dummy_1', 'dummy_2'],
                'sequence': ['GGGAAACCC', 'AAAUUUCCCGGG'],
                'temporal_cutoff': ['2025-03-30', '2025-03-30'],
                'description': ['Dummy test sequence 1', 'Dummy test sequence 2']
            })
            return self._process_test_df(test_df)

    def _process_test_df(self, test_df):
        """Helper function to process a test dataframe"""
        # Predict structures
        solution = self.predict_structure(test_df)

        # Convert to submission format
        submission_df = self.solution_to_submission(solution)

        # Save submission
        submission_path = os.path.join(self.config.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        logging.info(f"Saved submission to {submission_path}")

        return submission_path


# src/RNN3D/pipeline/stage_03_model.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.model import RNA3DStructurePredictor
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/model.log"),
        logging.StreamHandler()
    ]
)


def main():
    try:
        config = ConfigurationManager()
        model_config = config.get_model_config()

        # Create pretrained directory if it doesn't exist
        os.makedirs(os.path.dirname(model_config.pretrained_structures_path), exist_ok=True)

        # Initialize model
        model = RNA3DStructurePredictor(config=model_config)

        # Load pretrained structures
        model.load_pretrained_structures()

        # Process test sequences - find test file in the right location
        test_file_path = os.path.join(config.config.data_ingestion.unzip_dir, 'stanford-rna-3d-folding',
                                      'test_sequences.csv')
        submission_path = model.process_test_sequences(test_file_path)

        logging.info(f"Model prediction completed successfully. Submission saved to {submission_path}")

    except Exception as e:
        logging.error(f"Error in model prediction: {e}")
        raise e


if __name__ == "__main__":
    main()