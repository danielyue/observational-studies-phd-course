"""
Data Pipeline Orchestrator
Creates timestamped output directories and executes pipeline stages.
Supports separate download and processing modes via command line arguments.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stage_a_code import process_data
from downloadModelsData import download_models_data
from clean_models import clean_models_data


def setup_logging(log_dir):
    """
    Configure logging to both file and console.

    Args:
        log_dir: Directory to save log file
    """
    log_file = log_dir / 'pipeline.log'

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("="*60)
    logging.info("Pipeline execution started")
    logging.info(f"Log file: {log_file}")
    logging.info("="*60)


def run_download(project_root):
    """
    Run download stages only.

    Args:
        project_root: Root directory of the project
    """
    # Create timestamped log directory for downloads
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = project_root / 'data' / 'raw' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to raw/logs directory
    log_file = log_dir / f'download_{timestamp}.log'
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info("="*60)
    logging.info("Download mode: Starting data downloads")
    logging.info(f"Log file: {log_file}")
    logging.info("="*60)

    try:
        # Download data from HuggingFace
        raw_data_dir = project_root / 'data' / 'raw'
        download_models_data(raw_data_dir)

        # Add more download stages here as needed
        # from other_download_script import download_other_data
        # download_other_data(raw_data_dir)

        logging.info("="*60)
        logging.info("Download completed successfully!")
        logging.info(f"Data saved to: {raw_data_dir}")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Download failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


def run_processing(project_root):
    """
    Run data processing stages only.

    Args:
        project_root: Root directory of the project
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'data' / 'clean' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir)

    try:
        # Define paths
        raw_data_path = project_root / 'data' / 'raw' / 'sample_data.csv'
        clean_data_path = output_dir / 'cleaned_data.csv'
        models_parquet_path = project_root / 'data' / 'raw' / 'models.parquet'
        models_csv_path = output_dir / 'models.csv'

        # Execute processing stages
        logging.info("Starting data processing stages...")

        # Stage A: Clean models data
        clean_models_data(models_parquet_path, models_csv_path)

        # Stage B: Sample data cleaning (original pipeline)
        process_data(raw_data_path, clean_data_path)

        # Add more processing stages here as needed
        # from stage_c_code import process_stage_c
        # process_stage_c(models_csv_path, output_dir / 'stage_c_output.csv')

        logging.info("="*60)
        logging.info("Processing completed successfully!")
        logging.info(f"Output directory: {output_dir}")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Processing failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main pipeline orchestration function.
    Parses command line arguments and executes appropriate mode.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Data pipeline orchestrator with separate download and processing modes'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Run download mode to fetch data from external sources'
    )

    args = parser.parse_args()

    # Get project root (parent of pipeline directory)
    project_root = Path(__file__).parent.parent

    # Execute appropriate mode
    if args.download:
        run_download(project_root)
    else:
        run_processing(project_root)


if __name__ == '__main__':
    main()
