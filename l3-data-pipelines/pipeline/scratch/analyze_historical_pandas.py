#!/usr/bin/env python3
"""
Analyze Historical HuggingFace Downloads using Pandas

Computes daily downloads by author using pure Pandas operations (in-memory).

The task:
- Load historical download data from CSV
- Compute daily downloads (difference in downloadsAllTime between consecutive days)
- Aggregate by author to get total daily downloads per author
- Output results to CSV

Usage:
    uv run python scratch/analyze_historical_pandas.py
    uv run python scratch/analyze_historical_pandas.py --limit 1000000  # For testing
"""

import argparse
import logging
import time
from pathlib import Path
import pandas as pd


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def extract_author(model_id: str) -> str:
    """
    Extract author from model ID.

    Args:
        model_id: Model ID in format "author/model-name"

    Returns:
        Author name

    Examples:
        >>> extract_author("Qwen/Qwen-Image-Edit")
        'Qwen'
        >>> extract_author("no-slash-model")
        'no-slash-model'
    """
    if '/' in model_id:
        return model_id.split('/')[0]
    return model_id


def analyze_historical_data(csv_path: Path, output_path: Path, limit: int = None):
    """
    Compute author daily downloads using pure Pandas.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file
        limit: Optional row limit for testing
    """
    logging.info("=" * 60)
    logging.info("PANDAS ANALYSIS: Author Daily Downloads")
    logging.info("=" * 60)

    # Load data
    logging.info(f"Loading data from {csv_path}...")
    start_load = time.time()

    if limit:
        logging.info(f"Loading first {limit:,} rows for testing...")
        df = pd.read_csv(csv_path, nrows=limit)
    else:
        logging.info("Loading full dataset (this may take a while)...")
        df = pd.read_csv(csv_path)

    load_time = time.time() - start_load
    logging.info(f"✓ Data loaded: {len(df):,} rows in {load_time:.2f}s")
    logging.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Compute daily downloads
    logging.info("Computing daily downloads by author...")
    start_compute = time.time()

    # Select required columns
    df_subset = df[['id', 'downloadsAllTime', 'snapshot_date']].copy()

    # Extract author from id
    logging.info("  Extracting authors from model IDs...")
    df_subset['author'] = df_subset['id'].apply(extract_author)

    # Sort by id and snapshot_date to ensure correct ordering
    logging.info("  Sorting data by model ID and date...")
    df_subset = df_subset.sort_values(['id', 'snapshot_date'])

    # Compute daily downloads as difference between consecutive days for same model
    logging.info("  Computing daily download differences...")
    df_subset['daily_downloads'] = df_subset.groupby('id')['downloadsAllTime'].diff()

    # Fill NaN (first day for each model) with the downloadsAllTime value
    df_subset['daily_downloads'] = df_subset['daily_downloads'].fillna(df_subset['downloadsAllTime'])

    # Handle negative values (could happen if data is noisy or models are deleted/reset)
    # Set negative values to 0
    df_subset.loc[df_subset['daily_downloads'] < 0, 'daily_downloads'] = 0

    # Aggregate by author and snapshot_date
    logging.info("  Aggregating by author and date...")
    author_daily = df_subset.groupby(['author', 'snapshot_date'])['daily_downloads'].sum().reset_index()
    author_daily = author_daily.rename(columns={'daily_downloads': 'total_daily_downloads'})

    compute_time = time.time() - start_compute
    logging.info(f"✓ Computation complete in {compute_time:.2f}s")
    logging.info(f"  Result: {len(author_daily):,} author-date combinations")
    logging.info(f"  Unique authors: {author_daily['author'].nunique():,}")
    logging.info(f"  Date range: {author_daily['snapshot_date'].min()} to {author_daily['snapshot_date'].max()}")

    # Save results
    logging.info(f"Saving results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    author_daily.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"✓ Results saved: {file_size_mb:.2f} MB")

    # Show sample results
    logging.info("\nSample results (top 10 by total downloads):")
    sample = author_daily.nlargest(10, 'total_daily_downloads')
    for _, row in sample.iterrows():
        logging.info(f"  {row['author']}: {row['total_daily_downloads']:,.0f} downloads on {row['snapshot_date']}")

    # Print summary
    total_time = load_time + compute_time
    logging.info("\n" + "=" * 60)
    logging.info("PERFORMANCE SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Load Time:    {load_time:>8.2f}s ({load_time/total_time*100:>5.1f}%)")
    logging.info(f"Compute Time: {compute_time:>8.2f}s ({compute_time/total_time*100:>5.1f}%)")
    logging.info(f"Total Time:   {total_time:>8.2f}s")
    logging.info("\n✓ Analysis complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze historical HuggingFace downloads using Pandas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full dataset
  uv run python scratch/analyze_historical_pandas.py

  # Process first 1M rows only
  uv run python scratch/analyze_historical_pandas.py --limit 1000000

  # Custom input/output paths
  uv run python scratch/analyze_historical_pandas.py \\
    --input-file data/processed/model_historical_downloads.csv \\
    --output-file data/processed/author_daily_downloads.csv
        """
    )

    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'processed' / 'model_historical_downloads.csv',
        help='Input CSV file path (default: ../data/processed/model_historical_downloads.csv)'
    )

    parser.add_argument(
        '--output-file',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'processed' / 'author_daily_downloads_pandas.csv',
        help='Output CSV file path (default: ../data/processed/author_daily_downloads_pandas.csv)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of rows to process (for testing)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        analyze_historical_data(
            csv_path=args.input_file,
            output_path=args.output_file,
            limit=args.limit
        )
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()
