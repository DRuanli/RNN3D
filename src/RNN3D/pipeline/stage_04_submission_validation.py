# src/RNN3D/pipeline/stage_04_submission_validation.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.submission_validation import SubmissionValidator
import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/submission_validation.log"),
        logging.StreamHandler()
    ]
)


def main():
    try:
        # Create configuration
        config = ConfigurationManager()
        submission_validation_config = config.get_submission_validation_config()

        # Initialize validator
        validator = SubmissionValidator(config=submission_validation_config)

        # Run validation and evaluation
        result = validator.run_all()

        if result:
            logging.info("Submission validation and evaluation completed successfully")
            return True
        else:
            logging.error("Submission validation and evaluation failed")
            return False

    except Exception as e:
        logging.error(f"Error in submission validation: {e}")
        raise e


if __name__ == "__main__":
    main()