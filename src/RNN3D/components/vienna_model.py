# src/RNN3D/components/vienna_model.py
import os
import numpy as np
import pandas as pd
import pickle
import subprocess
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from src.RNN3D.entity.config_entity import ModelConfig


class ViennaRNAPredictor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.vienna_installed = False

    def setup_tools(self):
        """
        Set up the ViennaRNA tools
        """
        try:
            # Check if ViennaRNA is installed
            result = subprocess.run(["which", "RNAfold"],
                                    capture_output=True, text=True)

            if result.returncode == 0:
                logging.info("ViennaRNA is installed")
                self.vienna_installed = True
            else:
                logging.warning("ViennaRNA is not installed. Will use fallback method.")
                # You could add automatic installation here for some platforms
        except Exception as e:
            logging.warning(f"Error checking ViennaRNA installation: {e}")

    def predict_secondary_structure(self, sequence):
        """
        Predict RNA secondary structure using ViennaRNA or fallback method
        """
        if not hasattr(self, 'vienna_installed'):
            self.setup_tools()

        if self.vienna_installed:
            try:
                # Call RNAfold to predict the structure
                process = subprocess.run(
                    ["RNAfold"],
                    input=sequence,
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Extract the structure from the output
                lines = process.stdout.strip().split('\n')
                if len(lines) >= 2:
                    # Format: sequence\nstructure (energy)
                    structure_line = lines[1]
                    structure = structure_line.split()[0]
                    return structure
                else:
                    logging.warning(f"Unexpected output from RNAfold: {process.stdout}")
                    return self._fallback_structure_prediction(sequence)
            except Exception as e:
                logging.warning(f"Error using RNAfold: {e}")
                return self._fallback_structure_prediction(sequence)
        else:
            return self._fallback_structure_prediction(sequence)

    def _fallback_structure_prediction(self, sequence):
        """
        Simple fallback for secondary structure prediction when ViennaRNA is not available
        """
        # Very simplified model based on basic RNA folding principles
        # This is not accurate but serves as a placeholder
        seq_length = len(sequence)
        structure = ["." for _ in range(seq_length)]

        # Look for simple complementary regions
        stack = []
        pairs = {"A": "U", "U": "A", "G": "C", "C": "G"}

        for i, base in enumerate(sequence):
            # Skip if this position is already paired
            if structure[i] != ".":
                continue

            # Try to find a matching base
            if stack:
                last_idx, last_base = stack[-1]
                if pairs.get(last_base) == base:
                    # Found a pair
                    structure[last_idx] = "("
                    structure[i] = ")"
                    stack.pop()
                    continue

            # No pair found, add to stack
            stack.append((i, base))

        return "".join(structure)

    def secondary_to_3d_coordinates(self, sequence, secondary_structure, num_conformations=5):
        """
        Convert secondary structure to 3D coordinates
        """
        # Simple placeholder implementation - in a real scenario, you'd use a proper 3D structure generator
        seq_length = len(sequence)
        coords = []

        for i in range(num_conformations):
            # Create random but consistent 3D coordinates based on sequence and secondary structure
            np.random.seed(i + hash(sequence) % 10000)  # For reproducibility

            # Create a backbone trace with structure-based positioning
            coord = np.zeros((seq_length, 3), dtype=np.float32)

            # Stack of paired positions for helix formation
            paired_positions = []
            for j, char in enumerate(secondary_structure):
                if char == "(":
                    paired_positions.append(j)
                elif char == ")":
                    if paired_positions:
                        # Create a helical structure for paired bases
                        pair_idx = paired_positions.pop()
                        helix_angle = (j - pair_idx) * 0.3
                        coord[j] = np.array([np.cos(helix_angle), np.sin(helix_angle), j * 0.2]) * 10
                        coord[pair_idx] = np.array(
                            [np.cos(helix_angle + np.pi), np.sin(helix_angle + np.pi), pair_idx * 0.2]) * 10
                else:  # Unpaired base, loop region
                    coord[j] = np.array([np.cos(j * 0.2) * j * 0.5, np.sin(j * 0.2) * j * 0.5, j * 0.3]) * 5

            # Fix any remaining unassigned coordinates
            for j in range(seq_length):
                if np.all(coord[j] == 0):
                    coord[j] = np.array([np.random.normal(0, 5), np.random.normal(0, 5), j * 0.3])

                    # Add some random noise for different conformations
                coord[j] += np.random.normal(0, 2, 3)

            coords.append(coord)

        return coords

    def predict_structure(self, sequences_df):
        """
        Predict 3D structures for all sequences in the dataframe
        """
        solution = {}

        for i, row in sequences_df.iterrows():
            target_id = row.target_id
            sequence = row.sequence

            # Skip sequences that are too long
            if len(sequence) > self.config.max_sequence_length:
                logging.warning(f"Skipping {target_id} - sequence too long: {len(sequence)}")
                # Create empty coordinates
                coords = [np.zeros((len(sequence), 3)) for _ in range(self.config.num_conformations)]
            else:
                # Predict secondary structure
                secondary_structure = self.predict_secondary_structure(sequence)
                logging.info(f"Predicted structure for {target_id}: {secondary_structure[:20]}...")

                # Convert to 3D coordinates
                coords = self.secondary_to_3d_coordinates(
                    sequence,
                    secondary_structure,
                    self.config.num_conformations
                )

            solution[target_id] = {
                'target_id': target_id,
                'sequence': sequence,
                'coord': coords
            }

            if (i + 1) % 5 == 0:
                logging.info(f"Processed {i + 1}/{len(sequences_df)} sequences")

        return solution

    def solution_to_submission(self, solution):
        """
        Convert solution to submission format
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

    def process_test_sequences(self, test_file_path):
        """
        Process test sequences and generate predictions
        """
        try:
            # Check if test file exists
            if not os.path.exists(test_file_path):
                logging.error(f"Test file not found: {test_file_path}")
                # Try to find it in a different location
                test_dir = os.path.dirname(test_file_path)
                if os.path.exists(test_dir):
                    test_files = [f for f in os.listdir(test_dir) if 'test' in f.lower() and f.endswith('.csv')]
                    if test_files:
                        test_file_path = os.path.join(test_dir, test_files[0])
                        logging.info(f"Using alternative test file: {test_file_path}")
                    else:
                        # Create a dummy test file for demonstration
                        logging.warning("No test files found. Creating dummy test dataset.")
                        test_df = pd.DataFrame({
                            'target_id': ['dummy_1', 'dummy_2'],
                            'sequence': ['GGGAAACCC', 'AAAUUUCCCGGG'],
                            'temporal_cutoff': ['2025-03-30', '2025-03-30'],
                            'description': ['Dummy test sequence 1', 'Dummy test sequence 2']
                        })
                        return self._process_test_df(test_df)

            # Load test sequences
            test_df = pd.read_csv(test_file_path)
            logging.info(f"Processing {len(test_df)} test sequences")

            return self._process_test_df(test_df)

        except Exception as e:
            logging.error(f"Error processing test sequences: {e}")
            raise e

    def _process_test_df(self, test_df):
        """Helper function to process a test dataframe"""
        # Predict structures
        solution = self.predict_structure(test_df)

        # Convert to submission format
        submission_df = self.solution_to_submission(solution)

        # Save submission
        os.makedirs(self.config.output_dir, exist_ok=True)
        submission_path = os.path.join(self.config.output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        logging.info(f"Saved submission to {submission_path}")

        return submission_path