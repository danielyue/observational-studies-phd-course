"""
Run HuggingFace Data Collection Pipeline

This script orchestrates the entire data collection and analysis pipeline:
1. Fetch top N models from HuggingFace
2. Extract unique organizations from models
3. Scrape organization metadata
4. Analyze and report statistics

All logs are saved to a timestamped log file.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Set
import json

from get_hf_models import get_hf_models
from get_hf_organizations import get_hf_organizations
from analyze_data import analyze_data


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Set up logging to both console and timestamped file

    Args:
        log_dir: Directory to store log files

    Returns:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{timestamp}.log"

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    return logger


def extract_organizations_from_models(models_file: Path) -> Set[str]:
    """
    Extract unique organization names from models data

    Args:
        models_file: Path to models JSONL file

    Returns:
        Set of organization names
    """
    orgs = set()

    if not models_file.exists():
        logging.warning(f"Models file not found: {models_file}")
        return orgs

    with open(models_file, 'r') as f:
        for line in f:
            try:
                model = json.loads(line)
                model_id = model.get('id', '')

                # Extract author from model ID (format: author/model-name)
                if '/' in model_id:
                    author = model_id.split('/')[0]
                    orgs.add(author)

            except json.JSONDecodeError:
                continue

    return orgs


def run_pipeline(
    n_models: int = 100,
    requests_per_second: int = 5,
    sort: str = "downloads",
    direction: int = -1
):
    """
    Run the complete data collection and analysis pipeline

    Args:
        n_models: Number of top models to fetch (default: 100)
        requests_per_second: API rate limit (default: 5)
        sort: Sort field for models (default: downloads)
        direction: Sort direction (-1 for descending, 1 for ascending)
    """
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    log_dir = script_dir / "logs"

    models_file = data_dir / "models.jsonl"
    orgs_file = data_dir / "organizations.jsonl"

    # Setup logging
    logger = setup_logging(log_dir)

    try:
        logger.info("="*80)
        logger.info("HUGGINGFACE DATA COLLECTION PIPELINE")
        logger.info("="*80)
        logger.info(f"Configuration:")
        logger.info(f"  Models to fetch: {n_models}")
        logger.info(f"  Sort by: {sort} ({direction})")
        logger.info(f"  Rate limit: {requests_per_second} req/s")
        logger.info(f"  Data directory: {data_dir}")

        # Step 1: Fetch models
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Fetching HuggingFace Models")
        logger.info("="*80)

        models = get_hf_models(
            n=n_models,
            sort=sort,
            direction=direction,
            requests_per_second=requests_per_second,
            output_file=models_file
        )

        logger.info(f"✓ Fetched {len(models)} models")

        # Step 2: Extract organizations
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Extracting Organizations from Models")
        logger.info("="*80)

        orgs = extract_organizations_from_models(models_file)
        logger.info(f"✓ Found {len(orgs)} unique organizations")

        # Step 3: Scrape organizations
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Scraping Organization Metadata")
        logger.info("="*80)

        org_profiles = get_hf_organizations(
            org_names=sorted(orgs),  # Sort for consistent ordering
            requests_per_second=requests_per_second,
            output_file=orgs_file
        )

        logger.info(f"✓ Scraped {len(org_profiles)} organizations")

        # Step 4: Analyze data
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Analyzing Data")
        logger.info("="*80)

        analyze_data(
            models_file=models_file,
            organizations_file=orgs_file
        )

        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"✓ Models data: {models_file}")
        logger.info(f"✓ Organizations data: {orgs_file}")
        logger.info(f"✓ Log file: {log_dir}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HuggingFace data collection pipeline"
    )
    parser.add_argument(
        "-n", "--n-models",
        type=int,
        default=100,
        help="Number of top models to fetch (default: 100)"
    )
    parser.add_argument(
        "-r", "--rate-limit",
        type=int,
        default=5,
        help="API rate limit in requests per second (default: 5)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="downloads",
        choices=["downloads", "likes", "trending", "updated"],
        help="Sort field for models (default: downloads)"
    )
    parser.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=[1, -1],
        help="Sort direction: -1 for descending, 1 for ascending (default: -1)"
    )

    args = parser.parse_args()

    run_pipeline(
        n_models=args.n_models,
        requests_per_second=args.rate_limit,
        sort=args.sort,
        direction=args.direction
    )
