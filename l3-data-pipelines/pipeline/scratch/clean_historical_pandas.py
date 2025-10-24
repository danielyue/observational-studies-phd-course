#!/usr/bin/env python3
"""
Clean Historical HuggingFace Hub Data using Pandas

This script processes daily snapshots of HuggingFace Hub model data from parquet files,
extracting key metrics (id, likes, downloads, downloadsAllTime) and creating a
historical time series dataset.

Usage:
    uv run python src/clean_historical_pandas.py
    uv run python src/clean_historical_pandas.py --input-dir ../data/raw/historical --output-file ../data/clean/model_historical_downloads.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime


def parse_snapshot_date(filename: str) -> str:
    """
    Extract snapshot date from filename.

    Format: models-YYYYMMDD-{sha}.parquet

    Args:
        filename: Name of the parquet file

    Returns:
        Date string in YYYY-MM-DD format

    Examples:
        >>> parse_snapshot_date("models-20250825-510957d.parquet")
        '2025-08-25'
    """
    # Extract YYYYMMDD from filename
    match = re.match(r'models-(\d{8})-[a-f0-9]+\.parquet', filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected format")

    date_str = match.group(1)
    # Convert YYYYMMDD to YYYY-MM-DD
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    return date_obj.strftime("%Y-%m-%d")


def process_snapshot_file(file_path: Path, snapshot_date: str) -> pd.DataFrame:
    """
    Process a single snapshot file and extract relevant fields.

    Args:
        file_path: Path to the parquet file
        snapshot_date: Date string for this snapshot (YYYY-MM-DD)

    Returns:
        DataFrame with columns: id, likes, downloads, downloadsAllTime, snapshot_date
    """
    # Read parquet file
    df = pd.read_parquet(file_path)

    # Select relevant columns
    # Note: We need to check which columns exist in the data
    columns_to_extract = []

    # Required column
    if 'id' not in df.columns:
        raise ValueError(f"File {file_path} missing required 'id' column")
    columns_to_extract.append('id')

    # Optional columns with defaults
    if 'likes' in df.columns:
        columns_to_extract.append('likes')

    if 'downloads' in df.columns:
        columns_to_extract.append('downloads')

    if 'downloadsAllTime' in df.columns:
        columns_to_extract.append('downloadsAllTime')

    # Extract columns
    result_df = df[columns_to_extract].copy()

    # Add missing columns with NaN if they don't exist
    for col in ['likes', 'downloads', 'downloadsAllTime']:
        if col not in result_df.columns:
            result_df[col] = pd.NA

    # Add snapshot date
    result_df['snapshot_date'] = snapshot_date

    return result_df


def clean_historical_data(input_dir: Path, output_file: Path, verbose: bool = False):
    """
    Process all historical snapshot files and combine into a single CSV.

    Args:
        input_dir: Directory containing historical parquet files
        output_file: Path to output CSV file
        verbose: Whether to show detailed logging
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Processing historical data from: {input_dir}")
    logging.info(f"Output file: {output_file}")

    # Ensure input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all parquet files
    parquet_files = sorted(input_dir.glob("models-*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")

    logging.info(f"Found {len(parquet_files)} snapshot files")

    # Process each file
    all_data = []

    for file_path in tqdm(parquet_files, desc="Processing snapshots", unit="file"):
        try:
            # Extract date from filename
            snapshot_date = parse_snapshot_date(file_path.name)

            # Process the file
            df = process_snapshot_file(file_path, snapshot_date)

            all_data.append(df)

            if verbose:
                logging.debug(f"Processed {file_path.name}: {len(df)} records, date={snapshot_date}")

        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            raise

    # Combine all dataframes
    logging.info("Combining all snapshots...")
    combined_df = pd.concat(all_data, ignore_index=True)

    logging.info(f"Total records: {len(combined_df):,}")
    logging.info(f"Date range: {combined_df['snapshot_date'].min()} to {combined_df['snapshot_date'].max()}")
    logging.info(f"Unique models: {combined_df['id'].nunique():,}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    logging.info(f"Writing to {output_file}...")
    combined_df.to_csv(output_file, index=False)

    # Print summary statistics
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logging.info(f"Output file size: {file_size_mb:.2f} MB")
    logging.info("✓ Processing complete!")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Clean historical HuggingFace Hub model data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default paths
    uv run python src/clean_historical_pandas.py

    # Specify custom paths
    uv run python src/clean_historical_pandas.py \\
        --input-dir ../data/raw/historical \\
        --output-file ../data/processed/model_historical_downloads.csv

    # Verbose output
    uv run python src/clean_historical_pandas.py --verbose
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'raw' / 'historical',
        help='Directory containing historical parquet files (default: ../data/raw/historical)'
    )

    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'data' / 'processed' / 'model_historical_downloads.csv',
        help='Output CSV file path (default: ../data/processed/model_historical_downloads.csv)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    try:
        clean_historical_data(
            input_dir=args.input_dir,
            output_file=args.output_file,
            verbose=args.verbose
        )
    except Exception as e:
        logging.error(f"Failed to process historical data: {e}")
        raise


if __name__ == '__main__':
    main()
