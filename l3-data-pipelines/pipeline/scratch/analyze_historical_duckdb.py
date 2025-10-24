"""
Clean Historical Data with DuckDB

This script processes daily snapshots of HuggingFace models data to calculate:
1. Daily downloads per model (by taking differences in downloadsAllTime)
2. Total daily downloads aggregated by author

Input: Parquet files in l3-data-pipelines/data/raw/historical/
Output: Aggregated daily downloads by author
"""
import duckdb
from pathlib import Path
import re
import time


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


def process_historical_data(data_dir, output_path=None):
    """
    Process historical parquet files to compute daily downloads by author.

    Args:
        data_dir: Path to directory containing historical parquet files
        output_path: Optional path to save results as CSV

    Returns:
        DuckDB relation with daily downloads by author
    """
    # Track overall timing
    start_time = time.time()
    timings = {}

    data_dir = Path(data_dir)

    # Initialize DuckDB connection
    conn = duckdb.connect()

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
                    downloadsAllTime
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
            downloadsAllTime - LAG(downloadsAllTime) OVER (
                PARTITION BY id
                ORDER BY snapshot_date
            ) as daily_downloads
        FROM all_snapshots
        ORDER BY id, snapshot_date
    """)
    timings['calculate_daily_downloads'] = time.time() - step_start
    print(f"Calculated daily downloads per model (took {timings['calculate_daily_downloads']:.2f}s)")

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
            SUM(downloadsAllTime) as total_cumulative_downloads
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
            (HEADER, DELIMITER ',')
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
    print(f"Aggregate by author:         {timings['aggregate_by_author']:>8.2f}s ({timings['aggregate_by_author']/total_time*100:>5.1f}%)")
    if 'save_output' in timings:
        print(f"Save output:                 {timings['save_output']:>8.2f}s ({timings['save_output']/total_time*100:>5.1f}%)")
    print("-" * 80)
    print(f"TOTAL TIME:                  {total_time:>8.2f}s")
    print("="*80)

    # Return the result relation for further analysis
    return conn.execute("SELECT * FROM author_daily_downloads")


def main():
    """Main execution function."""
    print("="*80)
    print("HISTORICAL DATA PROCESSING - DUCKDB")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Set paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'raw' / 'historical'
    output_path = script_dir.parent / 'data' / 'processed' / 'author_daily_downloads.csv'

    # Process the data
    result = process_historical_data(data_dir, output_path)

    print("\n" + "="*80)
    print("Processing complete!")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Show a sample of the final results
    print("\nSample of final results (first 20 rows):")
    sample = result.df()
    print(sample.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
