# src/RNN3D/components/visualization.py
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from src.RNN3D.entity.config_entity import VisualizationConfig


class RNAVisualizer:
    def __init__(self, config: VisualizationConfig):
        """
        Initialize the RNA Visualizer

        Args:
            config (VisualizationConfig): Configuration for visualization
        """
        self.config = config

    def load_submission_data(self):
        """
        Loads the submission data for visualization

        Returns:
            DataFrame or None: The loaded dataframe if successful, None otherwise
        """
        try:
            submission_path = self.config.submission_path
            if not os.path.exists(submission_path):
                logging.error(f"Submission file not found: {submission_path}")
                return None

            df = pd.read_csv(submission_path)
            logging.info(f"Loaded submission data with {len(df)} rows")

            # Process data to extract unique RNA sequences
            rna_ids = df['ID'].str.split('_').str[0].unique()
            logging.info(f"Found {len(rna_ids)} unique RNA sequences")

            return df

        except Exception as e:
            logging.error(f"Error loading submission data: {e}")
            return None

    def generate_static_visualizations(self):
        """
        Generates static 3D visualizations of RNA structures

        Returns:
            dict: Paths to generated visualization files
        """
        df = self.load_submission_data()
        if df is None:
            return {}

        # Create output directory
        vis_dir = Path(self.config.visualizations_dir)
        os.makedirs(vis_dir, exist_ok=True)

        visualization_paths = {}

        # Get unique RNA IDs
        rna_ids = df['ID'].str.split('_').str[0].unique()

        # For each RNA sequence
        for rna_id in rna_ids:
            # Filter rows for this RNA
            rna_data = df[df['ID'].str.startswith(f"{rna_id}_")]

            # Generate visualizations for each conformation
            for conf_idx in range(1, self.config.num_conformations + 1):
                # Get coordinates for this conformation
                coord_cols = [f'x_{conf_idx}', f'y_{conf_idx}', f'z_{conf_idx}']

                # Skip if all coordinates are zero
                if (rna_data[coord_cols] == 0).all().all():
                    continue

                coords = rna_data[coord_cols].values
                resnames = rna_data['resname'].values

                # Create a 3D plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Plot the backbone
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', alpha=0.7)

                # Plot each nucleotide
                for i, (x, y, z) in enumerate(coords):
                    nucleotide = resnames[i]
                    color = self._get_nucleotide_color(nucleotide)
                    ax.scatter(x, y, z, color=color, s=50)

                # Set labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f"RNA {rna_id} - Conformation {conf_idx}")

                # Save the plot
                out_path = vis_dir / f"{rna_id}_conf{conf_idx}.png"
                plt.savefig(out_path)
                plt.close()

                # Store the path
                if rna_id not in visualization_paths:
                    visualization_paths[rna_id] = []
                visualization_paths[rna_id].append(str(out_path))

                logging.info(f"Generated visualization for {rna_id}, conformation {conf_idx}")

        # Save the paths to a JSON file
        vis_paths_file = vis_dir / "visualization_paths.json"
        with open(vis_paths_file, 'w') as f:
            json.dump(visualization_paths, f, indent=2)

        logging.info(f"Generated {sum(len(paths) for paths in visualization_paths.values())} visualizations")

        return visualization_paths

    def prepare_interactive_data(self):
        """
        Prepares data for interactive 3D visualizations in the web interface

        Returns:
            dict: Visualization data for the web interface
        """
        df = self.load_submission_data()
        if df is None:
            return {}

        # Create output directory
        vis_dir = Path(self.config.visualizations_dir)
        os.makedirs(vis_dir, exist_ok=True)

        visualization_data = {}

        # Get unique RNA IDs
        rna_ids = df['ID'].str.split('_').str[0].unique()

        # For each RNA sequence
        for rna_id in rna_ids:
            # Filter rows for this RNA
            rna_data = df[df['ID'].str.startswith(f"{rna_id}_")]

            # Get sequence
            sequence = ''.join(rna_data['resname'].values)

            # Generate data for each conformation
            conformations = []
            for conf_idx in range(1, self.config.num_conformations + 1):
                # Get coordinates for this conformation
                coord_cols = [f'x_{conf_idx}', f'y_{conf_idx}', f'z_{conf_idx}']

                # Skip if all coordinates are zero
                if (rna_data[coord_cols] == 0).all().all():
                    continue

                coords = rna_data[coord_cols].values

                # Prepare data for this conformation
                conf_data = {
                    'conformation_id': conf_idx,
                    'coordinates': coords.tolist(),
                    'nucleotides': rna_data['resname'].tolist(),
                    'residue_ids': rna_data['resid'].tolist()
                }

                conformations.append(conf_data)

            # Prepare data for this RNA
            rna_data = {
                'rna_id': rna_id,
                'sequence': sequence,
                'length': len(sequence),
                'conformations': conformations
            }

            visualization_data[rna_id] = rna_data

        # Save the data to a JSON file
        vis_data_file = vis_dir / "visualization_data.json"
        with open(vis_data_file, 'w') as f:
            json.dump(visualization_data, f, indent=2)

        logging.info(f"Prepared interactive visualization data for {len(rna_ids)} RNA sequences")

        return visualization_data

    def _get_nucleotide_color(self, nucleotide):
        """
        Returns a color for each nucleotide type

        Args:
            nucleotide (str): Nucleotide letter (A, C, G, U)

        Returns:
            str: Color code
        """
        color_map = {
            'A': 'red',
            'C': 'blue',
            'G': 'green',
            'U': 'orange'
        }

        return color_map.get(nucleotide, 'gray')

    def run(self):
        """
        Runs all visualization tasks

        Returns:
            dict: Results of the visualization process
        """
        results = {}

        # Generate static visualizations
        static_results = self.generate_static_visualizations()
        results['static_visualizations'] = static_results

        # Prepare interactive data
        interactive_data = self.prepare_interactive_data()
        results['interactive_data'] = bool(interactive_data)

        return results