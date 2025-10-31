"""
Clean HuggingFace Models Data with DuckDB

This script processes HuggingFace models data in two modes:
1. Simple mode: Clean a single models.parquet file
2. Historical mode: Process daily snapshots to calculate daily downloads per model and author

Input: Parquet files in data/raw/ and data/raw/historical/
Output: Cleaned Parquet files (compressed with ZSTD)
"""
import duckdb
from pathlib import Path
import re
import time
import logging

logger = logging.getLogger(__name__)


def clean_models_parquet(parquet_path: Path, output_path: Path):
    """
    Clean a single models.parquet file and export to Parquet.

    Args:
        parquet_path: Path to models.parquet file
        output_path: Path to save cleaned Parquet file

    Returns:
        Path to the output Parquet file
    """
    logger.info(f"Cleaning models data from: {parquet_path}")

    # Initialize DuckDB connection
    conn = duckdb.connect()

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read parquet and export to Parquet with basic cleaning
    logger.info("Processing models.parquet...")
    conn.execute(f"""
        COPY (
            SELECT
                id,
                SPLIT_PART(id, '/', 1) as author,
                SPLIT_PART(id, '/', 2) as model_name,
                downloadsAllTime,
                likes,
                trendingScore,
                downloads as downloads30,
                createdAt,
                lastModified,
                tags,
                pipeline_tag,
                library_name
            FROM '{parquet_path}'
            WHERE id IS NOT NULL
              AND SPLIT_PART(id, '/', 1) != ''
            ORDER BY downloadsAllTime DESC
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved cleaned models data to: {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def extract_date_from_filename(filename):
    """
    Extract date from filename like 'models-20250825-510957d.parquet'.

    Args:
        filename: Name of the parquet file

    Returns:
        Date string in YYYY-MM-DD format
    """
    match = re.search(r'models-(\d{8})-', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return None


def process_historical_data(data_dir, output_path=None, model_output_path=None):
    """
    Process historical parquet files to compute daily downloads by author.

    Args:
        data_dir: Path to directory containing historical parquet files
        output_path: Optional path to save author-level results as Parquet
        model_output_path: Optional path to save model-level results as Parquet

    Returns:
        DuckDB relation with daily downloads by author
    """
    # Track overall timing
    start_time = time.time()
    timings = {}

    data_dir = Path(data_dir)

    # Initialize DuckDB connection
    conn = duckdb.connect()

    # Configure DuckDB for large datasets
    conn.execute("SET preserve_insertion_order=false")
    conn.execute("SET threads=4")

    print(f"Processing historical data from: {data_dir}")

    # Get all parquet files sorted by date
    step_start = time.time()
    parquet_files = sorted(data_dir.glob("models-*.parquet"))
    timings['file_discovery'] = time.time() - step_start
    print(f"Found {len(parquet_files)} parquet files (took {timings['file_discovery']:.2f}s)")

    if len(parquet_files) == 0:
        raise ValueError(f"No parquet files found in {data_dir}")

    # Step 1: Load all historical data with dates
    print("\nStep 1: Loading historical data...")
    step_start = time.time()

    # Create a union of all files with their respective dates
    union_queries = []
    for file_path in parquet_files:
        date = extract_date_from_filename(file_path.name)
        if date:
            # Extract author from id (format: "author/model-name")
            query = f"""
                SELECT
                    '{date}'::DATE as snapshot_date,
                    id,
                    SPLIT_PART(id, '/', 1) as author,
                    downloadsAllTime,
                    likes,
                    trendingScore,
                    downloads as downloads30
                FROM '{file_path}'
                WHERE id IS NOT NULL
                  AND downloadsAllTime IS NOT NULL
                  AND SPLIT_PART(id, '/', 1) != ''
            """
            union_queries.append(query)

    if not union_queries:
        raise ValueError("No valid dates extracted from filenames")

    # Combine all snapshots
    full_union_query = " UNION ALL ".join(union_queries)

    conn.execute(f"CREATE OR REPLACE TEMP TABLE all_snapshots AS {full_union_query}")

    # Check the data
    result = conn.execute("SELECT COUNT(*) as total_rows FROM all_snapshots").fetchdf()
    timings['load_data'] = time.time() - step_start
    print(f"Loaded {result['total_rows'][0]:,} total rows across all snapshots (took {timings['load_data']:.2f}s)")

    # Step 2: Calculate daily downloads per model
    print("\nStep 2: Calculating daily downloads per model...")
    step_start = time.time()

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE daily_model_downloads AS
        SELECT
            snapshot_date,
            id,
            author,
            downloadsAllTime,
            likes,
            trendingScore,
            downloads30,
            downloadsAllTime - LAG(downloadsAllTime) OVER (
                PARTITION BY id
                ORDER BY snapshot_date
            ) as daily_downloads,
            likes - LAG(likes) OVER (
                PARTITION BY id
                ORDER BY snapshot_date
            ) as daily_likes,
            trendingScore - LAG(trendingScore) OVER (
                PARTITION BY id
                ORDER BY snapshot_date
            ) as daily_trending_score,
            downloads30 - LAG(downloads30) OVER (
                PARTITION BY id
                ORDER BY snapshot_date
            ) as daily_downloads30
        FROM all_snapshots
        ORDER BY id, snapshot_date
    """)
    timings['calculate_daily_downloads'] = time.time() - step_start
    print(f"Calculated daily downloads per model (took {timings['calculate_daily_downloads']:.2f}s)")

    # Save model-level results if output path is provided
    if model_output_path:
        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving model-level results to: {model_output_path}")
        step_start = time.time()
        conn.execute(f"""
            COPY daily_model_downloads
            TO '{model_output_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        timings['save_model_output'] = time.time() - step_start

        file_size_mb = model_output_path.stat().st_size / (1024 * 1024)
        print(f"Saved {model_output_path.name} ({file_size_mb:.2f} MB) (took {timings['save_model_output']:.2f}s)")

    # Step 3: Aggregate by author and date
    print("\nStep 3: Aggregating by author and date...")
    step_start = time.time()

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE author_daily_downloads AS
        SELECT
            snapshot_date,
            author,
            COUNT(DISTINCT id) as n_models,
            SUM(daily_downloads) as total_daily_downloads,
            AVG(daily_downloads) as avg_daily_downloads_per_model,
            SUM(downloadsAllTime) as total_cumulative_downloads,
            SUM(daily_likes) as total_daily_likes,
            AVG(daily_likes) as avg_daily_likes_per_model,
            SUM(likes) as total_cumulative_likes,
            SUM(daily_trending_score) as total_daily_trending_score,
            AVG(daily_trending_score) as avg_daily_trending_score_per_model,
            SUM(trendingScore) as total_cumulative_trending_score,
            SUM(daily_downloads30) as total_daily_downloads30,
            AVG(daily_downloads30) as avg_daily_downloads30_per_model,
            SUM(downloads30) as total_cumulative_downloads30
        FROM daily_model_downloads
        WHERE daily_downloads IS NOT NULL
          AND daily_downloads >= 0  -- Filter out negative values (data anomalies)
        GROUP BY snapshot_date, author
        ORDER BY snapshot_date, total_daily_downloads DESC
    """)
    timings['aggregate_by_author'] = time.time() - step_start
    print(f"Aggregated by author and date (took {timings['aggregate_by_author']:.2f}s)")

    # Show summary statistics
    print("\nSummary Statistics:")
    summary = conn.execute("""
        SELECT
            COUNT(DISTINCT snapshot_date) as n_days,
            COUNT(DISTINCT author) as n_authors,
            MIN(snapshot_date) as first_date,
            MAX(snapshot_date) as last_date,
            SUM(total_daily_downloads) as grand_total_downloads,
            AVG(total_daily_downloads) as avg_daily_downloads
        FROM author_daily_downloads
    """).fetchdf()
    print(summary.to_string())

    # Show top authors by total daily downloads
    print("\nTop 10 Authors by Total Daily Downloads:")
    top_authors = conn.execute("""
        SELECT
            author,
            COUNT(DISTINCT snapshot_date) as n_days_active,
            SUM(total_daily_downloads) as total_downloads,
            AVG(total_daily_downloads) as avg_daily_downloads,
            MAX(n_models) as max_models
        FROM author_daily_downloads
        GROUP BY author
        ORDER BY total_downloads DESC
        LIMIT 10
    """).fetchdf()
    print(top_authors.to_string(index=False))

    # Save results if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to: {output_path}")
        step_start = time.time()
        conn.execute(f"""
            COPY author_daily_downloads
            TO '{output_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        timings['save_output'] = time.time() - step_start

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved {output_path.name} ({file_size_mb:.2f} MB) (took {timings['save_output']:.2f}s)")

    # Calculate total time
    total_time = time.time() - start_time
    timings['total'] = total_time

    # Print timing summary
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARKS")
    print("="*80)
    print(f"File discovery:              {timings['file_discovery']:>8.2f}s ({timings['file_discovery']/total_time*100:>5.1f}%)")
    print(f"Load data:                   {timings['load_data']:>8.2f}s ({timings['load_data']/total_time*100:>5.1f}%)")
    print(f"Calculate daily downloads:   {timings['calculate_daily_downloads']:>8.2f}s ({timings['calculate_daily_downloads']/total_time*100:>5.1f}%)")
    if 'save_model_output' in timings:
        print(f"Save model output:           {timings['save_model_output']:>8.2f}s ({timings['save_model_output']/total_time*100:>5.1f}%)")
    print(f"Aggregate by author:         {timings['aggregate_by_author']:>8.2f}s ({timings['aggregate_by_author']/total_time*100:>5.1f}%)")
    if 'save_output' in timings:
        print(f"Save author output:          {timings['save_output']:>8.2f}s ({timings['save_output']/total_time*100:>5.1f}%)")
    print("-" * 80)
    print(f"TOTAL TIME:                  {total_time:>8.2f}s")
    print("="*80)

    # Return the result relation for further analysis
    return conn.execute("SELECT * FROM author_daily_downloads")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean HuggingFace models data'
    )
    parser.add_argument(
        '--parquet-file',
        type=str,
        help='Path to models.parquet file (for simple mode)'
    )
    parser.add_argument(
        '--output-file',
        '-o',
        type=str,
        help='Path to output Parquet file (for simple mode)'
    )
    parser.add_argument(
        '--historical',
        action='store_true',
        help='Process historical data instead of single file'
    )
    parser.add_argument(
        '--historical-dir',
        type=str,
        help='Path to historical data directory (default: hf_scraper/data/raw/historical)'
    )
    parser.add_argument(
        '--author-output',
        type=str,
        help='Path for author-level aggregated output (historical mode only)'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        help='Path for model-level output (historical mode only)'
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
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if args.historical:
        # Historical processing mode
        print("="*80)
        print("HISTORICAL DATA PROCESSING - DUCKDB")
        print("="*80)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Set paths
        script_dir = Path(__file__).parent
        data_dir = Path(args.historical_dir) if args.historical_dir else script_dir.parent / 'data' / 'raw' / 'historical'
        author_output = Path(args.author_output) if args.author_output else script_dir.parent / 'data' / 'processed' / 'author_daily_downloads.parquet'
        model_output = Path(args.model_output) if args.model_output else script_dir.parent / 'data' / 'processed' / 'daily_model_downloads.parquet'

        # Process the data
        result = process_historical_data(data_dir, author_output, model_output)

        print("\n" + "="*80)
        print("Processing complete!")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Show a sample of the final results
        print("\nSample of final results (first 20 rows):")
        sample = result.df()
        print(sample.head(20).to_string(index=False))

    else:
        # Simple cleaning mode
        if not args.parquet_file or not args.output_file:
            parser.error("--parquet-file and --output-file are required for simple mode")

        print("="*80)
        print("SIMPLE MODELS DATA CLEANING")
        print("="*80)

        parquet_path = Path(args.parquet_file)
        output_path = Path(args.output_file)

        if not parquet_path.exists():
            print(f"Error: File not found: {parquet_path}")
            return 1

        clean_models_parquet(parquet_path, output_path)

        print("="*80)
        print("Cleaning complete!")
        print("="*80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
