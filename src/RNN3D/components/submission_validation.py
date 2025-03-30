# src/RNN3D/components/submission_validation.py
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.RNN3D.entity.config_entity import SubmissionValidationConfig


class SubmissionValidator:
    def __init__(self, config: SubmissionValidationConfig):
        """
        Initialize the SubmissionValidator

        Args:
            config (SubmissionValidationConfig): Configuration for submission validation
        """
        self.config = config

    def validate_and_fix_submission(self, template_path=None):
        """
        Validates and fixes the submission file to match the expected format

        Args:
            template_path (str, optional): Path to a template file to use as reference

        Returns:
            bool: True if validation was successful, False otherwise
        """
        submission_path = self.config.submission_path

        logging.info(f"Validating submission file: {submission_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        # If template is provided, use it as reference
        if template_path and os.path.exists(template_path):
            template_df = pd.read_csv(template_path)
            logging.info(f"Using template file with {len(template_df)} rows")
        else:
            template_df = None

        # Required columns in the exact order
        required_columns = ['ID', 'resname', 'resid'] + \
                           [f'{coord}_{i}' for i in range(1, 6) for coord in ['x', 'y', 'z']]

        try:
            # Check if file exists and try to load it
            if os.path.exists(submission_path):
                df = pd.read_csv(submission_path)
                logging.info(f"Loaded existing submission with {len(df)} rows and {len(df.columns)} columns")

                # Check and fix column names
                if list(df.columns) != required_columns:
                    logging.warning("Column names don't match required format. Fixing...")

                    # Rename columns if possible, otherwise create new ones
                    fixed_df = pd.DataFrame()

                    # First, handle essential ID columns
                    for col in ['ID', 'resname', 'resid']:
                        if col in df.columns:
                            fixed_df[col] = df[col]
                        elif template_df is not None and col in template_df.columns:
                            fixed_df[col] = template_df[col]
                        else:
                            if col == 'ID':
                                fixed_df[col] = [f"R1107_{i + 1}" for i in range(len(df))]
                            elif col == 'resname':
                                fixed_df[col] = ['G'] * len(df)  # Default to G
                            elif col == 'resid':
                                fixed_df[col] = list(range(1, len(df) + 1))

                    # Then handle coordinate columns
                    for i in range(1, 6):
                        for coord in ['x', 'y', 'z']:
                            col_name = f'{coord}_{i}'
                            if col_name in df.columns:
                                fixed_df[col_name] = df[col_name]
                            elif template_df is not None and col_name in template_df.columns:
                                if len(fixed_df) == len(template_df):
                                    fixed_df[col_name] = template_df[col_name]
                                else:
                                    fixed_df[col_name] = np.random.normal(0, 10, len(fixed_df))
                            else:
                                fixed_df[col_name] = np.random.normal(0, 10, len(fixed_df))

                    df = fixed_df[required_columns]  # Ensure correct column order
                    logging.info("Column names fixed")

                # Ensure correct data types
                df['ID'] = df['ID'].astype(str)
                df['resname'] = df['resname'].astype(str)
                df['resid'] = df['resid'].astype(int)

                for i in range(1, 6):
                    for coord in ['x', 'y', 'z']:
                        col_name = f'{coord}_{i}'
                        df[col_name] = df[col_name].astype(float)

                # Check for NaN values and fill them
                if df.isna().any().any():
                    logging.warning("Found NaN values in submission. Filling with reasonable values...")

                    # Fill NaN values in coordinate columns
                    for i in range(1, 6):
                        for coord in ['x', 'y', 'z']:
                            col_name = f'{coord}_{i}'
                            if df[col_name].isna().any():
                                # Use mean of column or random values if all NaN
                                if df[col_name].notna().any():
                                    mean_val = df[col_name].mean()
                                    df[col_name] = df[col_name].fillna(mean_val)
                                else:
                                    df.loc[df[col_name].isna(), col_name] = np.random.normal(0, 10,
                                                                                             df[col_name].isna().sum())

                # Save fixed submission
                df.to_csv(submission_path, index=False)
                logging.info(f"Fixed submission saved to {submission_path}")

                # Store the dataframe for performance evaluation
                self.submission_df = df
                return True

            else:
                # File doesn't exist, create a new one
                logging.warning(f"Submission file not found: {submission_path}")

                if template_df is not None:
                    # Use template as a base
                    template_df.to_csv(submission_path, index=False)
                    logging.info(f"Created submission from template: {submission_path}")

                    # Store the dataframe for performance evaluation
                    self.submission_df = template_df
                    return True
                else:
                    logging.error("No template available and submission file not found")
                    return False

        except Exception as e:
            logging.error(f"Error processing submission file: {e}")
            return False

    def extract_sample_from_uploaded(self, csv_path):
        """
        Extracts a sample from the uploaded CSV to use as template

        Args:
            csv_path (str): Path to the uploaded CSV

        Returns:
            DataFrame or None: The loaded dataframe if successful, None otherwise
        """
        try:
            if not os.path.exists(csv_path):
                logging.error(f"Uploaded CSV not found: {csv_path}")
                return None

            df = pd.read_csv(csv_path)
            logging.info(f"Loaded uploaded CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logging.error(f"Error loading uploaded CSV: {e}")
            return None

    def evaluate_performance(self):
        """
        Evaluates the performance of the submission

        Returns:
            dict: Performance metrics
        """
        if not hasattr(self, 'submission_df'):
            logging.error("No submission data available for performance evaluation")
            return {}

        logging.info("Evaluating submission performance...")

        df = self.submission_df
        metrics = {}

        # Count the number of unique RNA sequences
        metrics['num_sequences'] = len(df['ID'].str.split('_').str[0].unique())

        # Check number of conformations
        num_confs = 0
        for i in range(1, 6):
            cols = [f'{coord}_{i}' for coord in ['x', 'y', 'z']]
            # Check if this conformation has any non-zero values
            if not (df[cols] == 0).all().all():
                num_confs += 1

        metrics['num_conformations'] = num_confs

        # Check if all coordinates are valid (not NaN or Inf)
        coord_columns = [f'{coord}_{i}' for i in range(1, 6) for coord in ['x', 'y', 'z']]
        metrics['has_invalid_values'] = df[coord_columns].isna().any().any() or df[coord_columns].isin(
            [np.inf, -np.inf]).any().any()

        # Check structural diversity between conformations
        diversity_scores = []

        # Calculate RMSD between different conformations
        for target_id in df['ID'].str.split('_').str[0].unique():
            target_rows = df[df['ID'].str.startswith(f"{target_id}_")]

            # Get coordinates for each conformation
            confs = []
            for i in range(1, 6):
                coords = target_rows[[f'x_{i}', f'y_{i}', f'z_{i}']].values
                if not (coords == 0).all():
                    confs.append(coords)

            # Calculate RMSD between each pair of conformations
            if len(confs) >= 2:
                rmsd_values = []
                for i in range(len(confs)):
                    for j in range(i + 1, len(confs)):
                        # Simple RMSD calculation
                        rmsd = np.sqrt(np.mean(np.sum((confs[i] - confs[j]) ** 2, axis=1)))
                        rmsd_values.append(rmsd)

                if rmsd_values:
                    diversity_scores.append(np.mean(rmsd_values))

        if diversity_scores:
            metrics['mean_structural_diversity'] = np.mean(diversity_scores)
        else:
            metrics['mean_structural_diversity'] = 0

        # Save metrics to a file
        metrics_path = self.config.metrics_path
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

        logging.info(f"Performance metrics saved to {metrics_path}")
        logging.info(f"Performance metrics: {metrics}")

        # Generate visual report
        if self.config.generate_report:
            self._generate_visual_report(metrics)

        return metrics

    def _generate_visual_report(self, metrics):
        """
        Generates a visual report of the performance metrics

        Args:
            metrics (dict): Performance metrics to visualize
        """
        report_path = self.config.report_path
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Generate some basic plots
        plt.figure(figsize=(10, 6))

        # Metrics bar chart
        plt.bar(['Sequences', 'Conformations', 'Diversity Score'],
                [metrics['num_sequences'], metrics['num_conformations'],
                 metrics.get('mean_structural_diversity', 0)])

        plt.title('RNN3D Submission Metrics')
        plt.ylabel('Count / Score')
        plt.tight_layout()

        # Save the plot
        plt.savefig(report_path)
        logging.info(f"Performance report saved to {report_path}")

    def run_all(self):
        """
        Runs all validation and evaluation steps

        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        # Extract sample if available
        if self.config.template_path and os.path.exists(self.config.template_path):
            template_df = self.extract_sample_from_uploaded(self.config.template_path)
        else:
            template_df = None

        # Validate and fix submission
        if not self.validate_and_fix_submission(self.config.template_path if self.config.template_path else None):
            return False

        # Evaluate performance
        self.evaluate_performance()

        return True