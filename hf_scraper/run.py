"""
Data Pipeline Orchestrator for HuggingFace Scraper

This script orchestrates the HuggingFace data pipeline with two modes:
1. Download mode (--download): Fetch data from HuggingFace
2. Clean mode (default): Process and clean downloaded data

The pipeline supports:
- Downloading current and historical models data
- Scraping HuggingFace profiles
- Cleaning and processing data into Parquet format (compressed)
"""
import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from download_models_data import download_models_data, download_historical_models_data
from download_hf_profiles import download_hf_profiles
from clean_models_data import clean_models_parquet, process_historical_data
from clean_hf_profiles import clean_hf_profiles


def setup_logging(log_dir=None, mode="pipeline"):
    """
    Configure logging to both file and console.

    Args:
        log_dir: Directory to save log file (optional)
        mode: Mode name for log file naming
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Add file handler if log directory is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{mode}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logging.info("="*60)
        logging.info(f"Pipeline execution started - {mode.upper()} mode")
        logging.info(f"Log file: {log_file}")
        logging.info("="*60)


def run_download(project_root, historical=False, top_n=1000):
    """
    Run download mode to fetch data from HuggingFace.

    Args:
        project_root: Root directory of the project
        historical: If True, download historical model data
        top_n: Number of top profiles to scrape (default: 1000)
    """
    raw_data_dir = project_root / 'data' / 'raw'
    log_dir = raw_data_dir / 'logs'

    # Setup logging
    setup_logging(log_dir, mode="download")

    try:
        logging.info("="*60)
        logging.info("DOWNLOAD MODE: Fetching data from HuggingFace")
        logging.info("="*60)

        # Step 1: Download models data
        if historical:
            logging.info("\nStep 1: Downloading historical models data...")
            logging.info("-" * 60)
            downloaded_files = download_historical_models_data(
                raw_data_dir,
                include_current=True
            )
            logging.info(f"Downloaded {len(downloaded_files)} files")
        else:
            logging.info("\nStep 1: Downloading current models data...")
            logging.info("-" * 60)
            models_path = download_models_data(raw_data_dir)
            logging.info(f"Downloaded to: {models_path}")

        # Step 2: Download HuggingFace profiles
        logging.info(f"\nStep 2: Downloading top {top_n} HuggingFace profiles...")
        logging.info("-" * 60)

        models_parquet_path = raw_data_dir / 'models.parquet'

        if not models_parquet_path.exists():
            logging.error(f"Error: models.parquet not found at {models_parquet_path}")
            logging.error("Cannot proceed with profile downloads")
            sys.exit(1)

        results = download_hf_profiles(
            parquet_path=models_parquet_path,
            output_dir=raw_data_dir,
            top_n=top_n,
            delay=0.5
        )

        logging.info("="*60)
        logging.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
        logging.info(f"Successfully downloaded {len(results['success'])} profiles")
        if results['retry']:
            logging.info(f"Profiles to retry: {len(results['retry'])}")
        logging.info(f"Data saved to: {raw_data_dir}")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"Download failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


def check_required_data(raw_data_dir):
    """
    Check for required data files and provide helpful error messages.

    Args:
        raw_data_dir: Path to raw data directory

    Returns:
        Dictionary with paths to required files
    """
    raw_data_dir = Path(raw_data_dir)
    errors = []
    warnings = []

    # Check for models.parquet
    models_path = raw_data_dir / 'models.parquet'
    if not models_path.exists():
        errors.append(
            f"Missing: {models_path}\n"
            f"  → Run with --download to download current models data"
        )

    # Check for historical data
    historical_dir = raw_data_dir / 'historical'
    has_historical = False
    if historical_dir.exists():
        historical_files = list(historical_dir.glob('models-*.parquet'))
        if historical_files:
            has_historical = True
        else:
            warnings.append(
                f"Historical directory exists but is empty: {historical_dir}\n"
                f"  → Run with --download --historical to download historical data"
            )
    else:
        warnings.append(
            f"No historical data found\n"
            f"  → Run with --download --historical to download historical data"
        )

    # Check for profile data
    profile_files = list(raw_data_dir.glob('profiles_top*.jsonl'))
    if not profile_files:
        errors.append(
            f"Missing: Profile data (profiles_top*.jsonl)\n"
            f"  → Run with --download to download profile data"
        )

    # Report errors
    if errors:
        logging.error("="*60)
        logging.error("MISSING REQUIRED DATA FILES")
        logging.error("="*60)
        for error in errors:
            logging.error(error)
        logging.error("="*60)
        sys.exit(1)

    # Report warnings
    if warnings:
        logging.warning("="*60)
        logging.warning("OPTIONAL DATA MISSING")
        logging.warning("="*60)
        for warning in warnings:
            logging.warning(warning)
        logging.warning("="*60)

    return {
        'models_path': models_path,
        'historical_dir': historical_dir if has_historical else None,
        'profile_files': profile_files
    }


def run_clean(project_root):
    """
    Run clean mode to process and clean downloaded data.

    Args:
        project_root: Root directory of the project
    """
    raw_data_dir = project_root / 'data' / 'raw'

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = project_root / 'data' / 'clean' / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(output_dir, mode="clean")

    try:
        logging.info("="*60)
        logging.info("CLEAN MODE: Processing raw data")
        logging.info("="*60)

        # Check for required data
        logging.info("\nChecking for required data files...")
        logging.info("-" * 60)
        data_paths = check_required_data(raw_data_dir)

        logging.info("All required data files found!")

        # Step 1: Clean models data
        logging.info("\nStep 1: Cleaning models data...")
        logging.info("-" * 60)

        models_parquet_path = output_dir / 'models.parquet'
        clean_models_parquet(data_paths['models_path'], models_parquet_path)

        # Step 2: Process historical data (if available)
        if data_paths['historical_dir']:
            logging.info("\nStep 2: Processing historical models data...")
            logging.info("-" * 60)

            author_output = output_dir / 'author_daily_downloads.parquet'
            model_output = output_dir / 'daily_model_downloads.parquet'

            process_historical_data(
                data_paths['historical_dir'],
                author_output,
                model_output
            )
        else:
            logging.info("\nStep 2: Skipping historical processing (no data)")
            logging.info("-" * 60)

        # Step 3: Clean profile data
        logging.info("\nStep 3: Cleaning profile data...")
        logging.info("-" * 60)

        created_files = clean_hf_profiles(
            input_files=data_paths['profile_files'],
            output_dir=output_dir
        )

        logging.info(f"Created {len(created_files)} CSV files:")
        for file in created_files:
            logging.info(f"  - {file.name}")

        logging.info("="*60)
        logging.info("CLEANING COMPLETED SUCCESSFULLY!")
        logging.info(f"Output directory: {output_dir}")
        logging.info("="*60)

        # List all output files (parquet and csv)
        all_files = list(output_dir.glob('*.parquet')) + list(output_dir.glob('*.csv'))
        if all_files:
            logging.info("\nGenerated files:")
            for file in sorted(all_files):
                size_mb = file.stat().st_size / (1024 * 1024)
                logging.info(f"  - {file.name} ({size_mb:.2f} MB)")

    except Exception as e:
        logging.error(f"Cleaning failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main pipeline orchestration function.
    Parses command line arguments and executes appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description='HuggingFace data pipeline orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download current data
  python run.py --download

  # Download with historical data
  python run.py --download --historical

  # Download top 500 profiles
  python run.py --download --top-n 500

  # Clean downloaded data (default mode)
  python run.py --clean

  # Clean is the default, so this is equivalent:
  python run.py
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--download',
        action='store_true',
        help='Run download mode to fetch data from HuggingFace'
    )
    mode_group.add_argument(
        '--clean',
        action='store_true',
        help='Run clean mode to process downloaded data (default)'
    )

    # Download mode options
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Download historical model data (only with --download)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=1000,
        help='Number of top profiles to scrape (default: 1000, only with --download)'
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).parent

    # Default to clean mode if no mode specified
    if not args.download and not args.clean:
        args.clean = True

    # Validate historical flag
    if args.historical and not args.download:
        parser.error("--historical can only be used with --download")

    # Execute appropriate mode
    if args.download:
        run_download(project_root, historical=args.historical, top_n=args.top_n)
    elif args.clean:
        run_clean(project_root)


if __name__ == '__main__':
    main()
