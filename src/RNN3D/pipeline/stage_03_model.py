# src/RNN3D/pipeline/stage_03_model.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.vienna_model import ViennaRNAPredictor
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

        # Initialize ViennaRNA predictor
        model = ViennaRNAPredictor(config=model_config)

        # Process test sequences
        test_file_path = os.path.join(config.config.data_ingestion.unzip_dir, 'stanford-rna-3d-folding',
                                      'test_sequences.csv')
        submission_path = model.process_test_sequences(test_file_path)

        logging.info(f"Model prediction completed successfully. Submission saved to {submission_path}")

    except Exception as e:
        logging.error(f"Error in model prediction: {e}")
        raise e


if __name__ == "__main__":
    main()