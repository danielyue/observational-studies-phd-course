"""
Get Top HuggingFace Profiles by Downloads

This script loads the models.parquet file, identifies the top N profiles by total
downloads across all their models, and scrapes profile data using scrape_hf_profile.
Results are saved to a JSONL file, with failed profiles (429 errors) logged separately.
"""

import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from scrape_hf_profile import scrape_hf_profile

logger = logging.getLogger(__name__)


def load_and_rank_authors(
    parquet_path: Path,
    top_n: int
) -> pd.DataFrame:
    """
    Load models.parquet and rank authors by total downloads.

    Args:
        parquet_path: Path to models.parquet file
        top_n: Number of top profiles to return

    Returns:
        DataFrame with top N authors and their total downloads
    """
    logger.info(f"Loading parquet file from {parquet_path}")

    # Read only the columns we need for efficiency
    df = pd.read_parquet(parquet_path, columns=['author', 'downloadsAllTime'])

    logger.info(f"Loaded {len(df):,} models")

    # Group by author and sum downloads
    author_downloads = df.groupby('author')['downloadsAllTime'].sum().reset_index()
    author_downloads.columns = ['author', 'total_downloads']

    # Sort by total downloads descending
    author_downloads = author_downloads.sort_values('total_downloads', ascending=False)

    # Get top N
    top_authors = author_downloads.head(top_n)

    logger.info(f"Top {top_n} authors by downloads:")
    for idx, row in top_authors.head(10).iterrows():
        logger.info(f"  {row['author']}: {row['total_downloads']:,} downloads")
    if top_n > 10:
        logger.info(f"  ... and {top_n - 10} more")

    return top_authors


def scrape_profiles(
    authors: pd.DataFrame,
    output_dir: Path,
    run_timestamp: str,
    top_n: int,
    delay: float = 0.5
) -> Dict[str, List[str]]:
    """
    Scrape profile data for each author.

    Args:
        authors: DataFrame with author names and total downloads
        output_dir: Directory to save output files
        run_timestamp: Timestamp string for filenames
        top_n: Number of profiles being scraped (for filename)
        delay: Delay between requests in seconds (default: 0.5)

    Returns:
        Dictionary with 'success' and 'retry' lists of profile names
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file paths
    output_file = output_dir / f"profiles_top{top_n}_{run_timestamp}.jsonl"
    retry_file = output_dir / f"profiles_retry_top{top_n}_{run_timestamp}.txt"

    results = {
        'success': [],
        'retry': []
    }

    logger.info(f"Scraping {len(authors)} profiles...")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Retry file: {retry_file}")

    with open(output_file, 'w') as f_out:
        for i, (idx, row) in enumerate(authors.iterrows()):
            author = row['author']
            total_downloads = row['total_downloads']

            logger.info(f"[{i + 1}/{len(authors)}] Scraping {author} ({total_downloads:,} downloads)")

            try:
                # Scrape profile
                result = scrape_hf_profile(author)

                # Check for errors
                if result.get('error'):
                    error_msg = result['error']

                    # Check if it's a 429 error
                    if '429' in str(error_msg):
                        logger.warning(f"Rate limited on {author} - adding to retry list")
                        results['retry'].append(author)

                        # Write to retry file immediately
                        with open(retry_file, 'a') as f_retry:
                            f_retry.write(f"{author}\n")

                        # Wait longer before next request
                        logger.info(f"Waiting 60 seconds before continuing...")
                        time.sleep(60)
                    else:
                        logger.error(f"Error scraping {author}: {error_msg}")
                        results['retry'].append(author)

                        # Write to retry file
                        with open(retry_file, 'a') as f_retry:
                            f_retry.write(f"{author}\n")
                else:
                    # Add download stats to result
                    result['total_downloads'] = int(total_downloads)

                    # Write to JSONL file
                    f_out.write(json.dumps(result) + '\n')
                    f_out.flush()  # Ensure written to disk

                    results['success'].append(author)
                    logger.info(f"Successfully scraped {author}")

            except Exception as e:
                logger.error(f"Unexpected error scraping {author}: {e}", exc_info=True)
                results['retry'].append(author)

                # Write to retry file
                with open(retry_file, 'a') as f_retry:
                    f_retry.write(f"{author}\n")

            # Rate limiting delay
            if i < len(authors) - 1:  # Don't wait after last request
                time.sleep(delay)

    logger.info(f"\nScraping complete!")
    logger.info(f"Successfully scraped: {len(results['success'])} profiles")
    logger.info(f"Profiles to retry: {len(results['retry'])} profiles")

    if results['retry']:
        logger.info(f"Retry list saved to: {retry_file}")

    return results


def retry_failed_profiles(
    retry_file_path: Path,
    output_file_path: Path,
    delay: float = 0.5
) -> Dict[str, List[str]]:
    """
    Retry scraping profiles from a retry file and append to output file.

    Args:
        retry_file_path: Path to retry file with list of profile names
        output_file_path: Path to JSONL file to append results to
        delay: Delay between requests in seconds

    Returns:
        Dictionary with 'success' and 'retry' lists of profile names
    """
    # Read profiles from retry file
    with open(retry_file_path, 'r') as f:
        profiles = [line.strip() for line in f if line.strip()]

    logger.info(f"Retrying {len(profiles)} profiles from {retry_file_path}")
    logger.info(f"Appending results to {output_file_path}")

    # Create new retry file for any remaining failures
    retry_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_retry_file = retry_file_path.parent / f"{retry_file_path.stem}_retry_{retry_timestamp}.txt"

    results = {
        'success': [],
        'retry': []
    }

    # Open output file in append mode
    with open(output_file_path, 'a') as f_out:
        for i, author in enumerate(profiles):
            logger.info(f"[{i + 1}/{len(profiles)}] Retrying {author}")

            try:
                # Scrape profile
                result = scrape_hf_profile(author)

                # Check for errors
                if result.get('error'):
                    error_msg = result['error']

                    # Check if it's a 429 error
                    if '429' in str(error_msg):
                        logger.warning(f"Rate limited on {author} - adding to new retry list")
                        results['retry'].append(author)

                        # Write to new retry file
                        with open(new_retry_file, 'a') as f_retry:
                            f_retry.write(f"{author}\n")

                        # Wait longer before next request
                        logger.info(f"Waiting 60 seconds before continuing...")
                        time.sleep(60)
                    else:
                        logger.error(f"Error scraping {author}: {error_msg}")
                        results['retry'].append(author)

                        # Write to new retry file
                        with open(new_retry_file, 'a') as f_retry:
                            f_retry.write(f"{author}\n")
                else:
                    # Write to JSONL file (append mode)
                    f_out.write(json.dumps(result) + '\n')
                    f_out.flush()

                    results['success'].append(author)
                    logger.info(f"Successfully scraped {author}")

            except Exception as e:
                logger.error(f"Unexpected error scraping {author}: {e}", exc_info=True)
                results['retry'].append(author)

                # Write to new retry file
                with open(new_retry_file, 'a') as f_retry:
                    f_retry.write(f"{author}\n")

            # Rate limiting delay
            if i < len(profiles) - 1:
                time.sleep(delay)

    logger.info(f"\nRetry complete!")
    logger.info(f"Successfully scraped: {len(results['success'])} profiles")
    logger.info(f"Profiles still failing: {len(results['retry'])} profiles")

    if results['retry']:
        logger.info(f"New retry list saved to: {new_retry_file}")

    return results


def download_hf_profiles(parquet_path: Path, output_dir: Path, top_n: int = 1000, delay: float = 0.5):
    """
    Download HuggingFace profiles for top N authors by downloads.

    Args:
        parquet_path: Path to models.parquet file
        output_dir: Directory to save output files
        top_n: Number of top profiles to scrape (default: 1000)
        delay: Delay between requests in seconds (default: 0.5)

    Returns:
        Dictionary with 'success' and 'retry' lists of profile names
    """
    # Generate timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load and rank authors
    top_authors = load_and_rank_authors(parquet_path, top_n)

    # Scrape profiles
    results = scrape_profiles(
        top_authors,
        output_dir,
        run_timestamp,
        top_n,
        delay=delay
    )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scrape top HuggingFace profiles by downloads'
    )
    parser.add_argument(
        'top_n',
        type=int,
        nargs='?',
        help='Number of top profiles to scrape (not required if using --retry-file)'
    )
    parser.add_argument(
        '--parquet-file',
        type=str,
        default='hf_scraper/data/raw/models.parquet',
        help='Path to models.parquet file (default: hf_scraper/data/raw/models.parquet)'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default='hf_scraper/data/raw',
        help='Output directory for JSONL file (default: hf_scraper/data/raw)'
    )
    parser.add_argument(
        '--retry-file',
        '-r',
        type=str,
        help='Path to retry file with failed profiles to re-scrape'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to output JSONL file (required when using --retry-file if cannot be inferred)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between requests in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Retry mode
    if args.retry_file:
        retry_file_path = Path(args.retry_file)

        if not retry_file_path.exists():
            logger.error(f"Retry file not found: {retry_file_path}")
            return 1

        # Try to infer output file from retry file name if not specified
        if args.output_file:
            output_file_path = Path(args.output_file)
        else:
            # Convert profiles_retry_topN_timestamp.txt -> profiles_topN_timestamp.jsonl
            output_filename = retry_file_path.stem.replace('_retry', '') + '.jsonl'
            output_file_path = retry_file_path.parent / output_filename

        if not output_file_path.exists():
            logger.error(f"Output file not found: {output_file_path}")
            logger.error("Please specify --output-file explicitly")
            return 1

        logger.info("=== RETRY MODE ===")
        results = retry_failed_profiles(
            retry_file_path,
            output_file_path,
            delay=args.delay
        )

        logger.info("Done!")
        return 0

    # Normal mode - scrape top N
    if not args.top_n:
        logger.error("Error: top_n is required unless using --retry-file")
        parser.print_help()
        return 1

    # Generate timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Convert paths
    parquet_path = Path(args.parquet_file)
    output_dir = Path(args.output_dir)

    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return 1

    # Load and rank authors
    top_authors = load_and_rank_authors(parquet_path, args.top_n)

    # Scrape profiles
    results = scrape_profiles(
        top_authors,
        output_dir,
        run_timestamp,
        args.top_n,
        delay=args.delay
    )

    logger.info("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
