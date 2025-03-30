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
        Convert secondary structure to more realistic 3D coordinates
        """
        import numpy as np

        # Constants for RNA geometry
        P_C4_DIST = 6.0  # Approx distance between phosphate and C4' in RNA
        HELIX_RISE = 2.8  # Rise per base pair in A-form RNA helix
        HELIX_TURN = 32.7  # Degrees of turn per base pair in A-form RNA
        BACKBONE_DIST = 4.0  # Approx distance between consecutive phosphates

        seq_length = len(sequence)
        coords = []

        for i in range(num_conformations):
            # Use a different seed for each conformation but consistent for reproducibility
            np.random.seed(i + hash(sequence) % 10000)

            # Initialize coordinates array
            coord = np.zeros((seq_length, 3), dtype=np.float32)

            # Stack of paired positions for helix formation
            paired_positions = []
            paired_bases = set()

            # First pass: identify all paired bases
            for j, char in enumerate(secondary_structure):
                if char == '(':
                    paired_positions.append(j)
                elif char == ')':
                    if paired_positions:
                        pair_idx = paired_positions.pop()
                        paired_bases.add(j)
                        paired_bases.add(pair_idx)

            # Second pass: lay out the RNA backbone
            current_pos = np.array([0.0, 0.0, 0.0])
            current_vec = np.array([1.0, 0.0, 0.0])  # Initial direction

            # Stack structure for helix formation
            helix_stack = []
            in_helix = False
            helix_dir = None

            for j, char in enumerate(secondary_structure):
                # Basic position based on backbone trace
                coord[j] = current_pos.copy()

                if char == '(':
                    # Start or continue helix
                    helix_stack.append(j)
                    if not in_helix:
                        in_helix = True
                        # Pick a random direction for helix axis that's perpendicular to current_vec
                        helix_dir = np.cross(current_vec, np.random.randn(3))
                        helix_dir = helix_dir / np.linalg.norm(helix_dir)

                    # Move along the helix
                    angle = np.radians(HELIX_TURN)
                    rot_matrix = self._rotation_matrix(helix_dir, angle)
                    current_vec = np.dot(rot_matrix, current_vec)
                    current_pos += current_vec * BACKBONE_DIST

                elif char == ')':
                    # End or continue helix
                    if helix_stack:
                        helix_stack.pop()

                        # Move along the helix
                        angle = np.radians(HELIX_TURN)
                        rot_matrix = self._rotation_matrix(helix_dir, angle)
                        current_vec = np.dot(rot_matrix, current_vec)
                        current_pos += current_vec * BACKBONE_DIST

                        if not helix_stack:
                            in_helix = False
                else:
                    # Loop region - move in a random direction that's somewhat smooth
                    if j > 0:
                        # Generate a random perturbation to the current direction
                        perturb = np.random.randn(3) * 0.3
                        current_vec = current_vec + perturb
                        current_vec = current_vec / np.linalg.norm(current_vec)
                        current_pos += current_vec * BACKBONE_DIST

            # Third pass: adjust paired bases to be across from each other in helices
            for j in range(seq_length):
                if j in paired_bases:
                    # Find the paired base
                    paired_j = None
                    for k in range(seq_length):
                        if k != j and k in paired_bases:
                            if (secondary_structure[j] == '(' and secondary_structure[k] == ')') or \
                                    (secondary_structure[j] == ')' and secondary_structure[k] == '('):
                                paired_j = k
                                break

                    if paired_j is not None:
                        # Adjust the positions to be across from each other
                        center = (coord[j] + coord[paired_j]) / 2
                        direction = coord[j] - coord[paired_j]
                        direction = direction / np.linalg.norm(direction) * P_C4_DIST

                        coord[j] = center + direction / 2
                        coord[paired_j] = center - direction / 2

            # Add random noise for each conformation
            for j in range(seq_length):
                coord[j] += np.random.normal(0, 1, 3)

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
            sequence = data['sequence']
            coords = data['coord']

            # Ensure we have exactly 5 conformations
            while len(coords) < 5:
                # If we have fewer than 5, duplicate the last one with small variations
                if coords:
                    new_coords = coords[-1].copy()
                    # Add small random variations
                    new_coords += np.random.normal(0, 1, new_coords.shape)
                    coords.append(new_coords)
                else:
                    # Create random coordinates if none exist
                    new_coords = np.random.normal(0, 10, (len(sequence), 3))
                    coords.append(new_coords)

            # Limit to 5 conformations if we have more
            coords = coords[:5]

            # Create submission dataframe
            df = pd.DataFrame()
            df['ID'] = [f'{target_id}_{i + 1}' for i in range(len(sequence))]
            df['resname'] = list(sequence)
            df['resid'] = [i + 1 for i in range(len(sequence))]

            # Add coordinates for each conformation
            for j in range(5):  # Always use exactly 5 conformations
                if j < len(coords):
                    conf_coords = coords[j]
                    # Ensure the shape matches
                    if len(conf_coords) < len(sequence):
                        # Pad with zeros if needed
                        padding = np.zeros((len(sequence) - len(conf_coords), 3))
                        conf_coords = np.vstack([conf_coords, padding])
                    elif len(conf_coords) > len(sequence):
                        # Truncate if needed
                        conf_coords = conf_coords[:len(sequence)]

                    df[f'x_{j + 1}'] = conf_coords[:, 0]
                    df[f'y_{j + 1}'] = conf_coords[:, 1]
                    df[f'z_{j + 1}'] = conf_coords[:, 2]
                else:
                    # Fill with zeros if missing
                    df[f'x_{j + 1}'] = [0.0] * len(sequence)
                    df[f'y_{j + 1}'] = [0.0] * len(sequence)
                    df[f'z_{j + 1}'] = [0.0] * len(sequence)

            submit_dfs.append(df)

        # Combine all dataframes
        result = pd.concat(submit_dfs) if submit_dfs else pd.DataFrame()

        # Ensure no NaN values
        if result.isna().any().any():
            # Fill NaN values with 0
            result = result.fillna(0.0)

        return result
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

    def _rotation_matrix(self, axis, angle):
        """
        Return the rotation matrix for rotation around axis by angle (radians)
        """
        import numpy as np

        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2)
        b, c, d = -axis * np.sin(angle / 2)

        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]
        ])