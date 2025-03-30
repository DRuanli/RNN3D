# main.py
import logging
import os
from src.RNN3D.pipeline.stage_01_data_ingestion import main as data_ingestion_main
from src.RNN3D.pipeline.stage_02_data_preparation import main as data_preparation_main
from src.RNN3D.pipeline.stage_03_model import main as model_main
import subprocess

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/running_logs.log"),
        logging.StreamHandler()
    ]
)


def validate_submission():
    """Run the submission validation script"""
    try:
        logging.info("\n\n>>>>> Validating Submission Format <<<<<")
        subprocess.run(["python", "ensure_submission.py"], check=True)
        logging.info(">>>>> Submission Validation Completed <<<<<\n\n")
        return True
    except Exception as e:
        logging.error(f">>>>> Submission Validation Failed: {e} <<<<<")
        return False


if __name__ == "__main__":
    try:
        logging.info("\n\n>>>>> Stage 1: Data Ingestion Started <<<<<")
        # data_ingestion_main()
        logging.info(">>>>> Stage 1: Data Ingestion Completed <<<<<\n\n")

        logging.info("\n\n>>>>> Stage 2: Data Preparation Started <<<<<")
        # data_preparation_main()
        logging.info(">>>>> Stage 2: Data Preparation Completed <<<<<\n\n")

        logging.info("\n\n>>>>> Stage 3: Model Prediction Started <<<<<")
        model_main()
        logging.info(">>>>> Stage 3: Model Prediction Completed <<<<<\n\n")

        # Add validation step
        validate_submission()

    except Exception as e:
        logging.error(f">>>>> Pipeline Failed with error: {e} <<<<<")
        raise e