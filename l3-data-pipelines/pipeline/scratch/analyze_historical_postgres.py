#!/usr/bin/env python3
"""
Analyze Historical HuggingFace Downloads using PostgreSQL

Computes daily downloads by author using PostgreSQL with optimized COPY loading.

The task:
- Load historical download data into PostgreSQL using native COPY command
- Create indexes for optimal query performance
- Compute daily downloads using SQL window functions (LAG)
- Aggregate by author to get total daily downloads per author
- Output results to CSV

Usage:
    uv run python scratch/analyze_historical_postgres.py --db-user danielyue
    uv run python scratch/analyze_historical_postgres.py --limit 1000000  # For testing
"""

import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2 import sql
import tempfile
import os


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def analyze_historical_data(
    csv_path: Path,
    output_path: Path,
    db_name: str = "hf_models",
    db_user: str = "postgres",
    db_password: str = "postgres",
    db_host: str = "localhost",
    db_port: int = 5432,
    limit: int = None
):
    """
    Compute author daily downloads using PostgreSQL.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file
        db_name: PostgreSQL database name
        db_user: PostgreSQL username
        db_password: PostgreSQL password
        db_host: PostgreSQL host
        db_port: PostgreSQL port
        limit: Optional row limit for testing
    """
    logging.info("=" * 60)
    logging.info("POSTGRESQL ANALYSIS: Author Daily Downloads")
    logging.info("=" * 60)

    # Connect to PostgreSQL
    logging.info(f"Connecting to PostgreSQL at {db_host}:{db_port}/{db_name}...")
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.autocommit = False
        cur = conn.cursor()
        logging.info("✓ Connected to PostgreSQL")
    except psycopg2.OperationalError as e:
        logging.error(f"Failed to connect to PostgreSQL: {e}")
        logging.error("Make sure PostgreSQL is running and credentials are correct")
        logging.error("\nTo set up PostgreSQL:")
        logging.error("  # Using Docker:")
        logging.error("  docker run --name postgres-hf -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres")
        logging.error("  docker exec -it postgres-hf createdb -U postgres hf_models")
        logging.error("\n  # Or using local PostgreSQL:")
        logging.error("  createdb hf_models")
        raise

    try:
        # Load data
        logging.info(f"Loading data from {csv_path}...")
        start_load = time.time()

        # Drop and recreate table
        logging.info("  Creating table...")
        cur.execute("""
            DROP TABLE IF EXISTS model_downloads;
            CREATE TABLE model_downloads (
                id TEXT,
                likes BIGINT,
                downloads BIGINT,
                downloads_all_time BIGINT,
                snapshot_date DATE
            );
        """)

        # Load CSV data using PostgreSQL's native COPY command (MUCH faster!)
        logging.info("  Loading CSV into PostgreSQL using COPY command...")

        if limit:
            # For limited rows, we need to create a temp CSV file
            logging.info(f"  Creating temporary CSV with first {limit:,} rows...")
            df = pd.read_csv(csv_path, nrows=limit)
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            csv_to_load = temp_file.name
        else:
            # Use the original CSV file directly
            csv_to_load = csv_path

        try:
            # Use PostgreSQL's COPY command for fast bulk loading
            with open(csv_to_load, 'r') as f:
                cur.copy_expert(
                    """
                    COPY model_downloads (id, likes, downloads, downloads_all_time, snapshot_date)
                    FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '')
                    """,
                    f
                )
            conn.commit()
            logging.info("  ✓ Data loaded successfully")

            # Clean up temp file if created
            if limit:
                os.unlink(csv_to_load)

        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            conn.rollback()
            raise

        # Create indexes
        logging.info("  Creating indexes...")
        cur.execute("CREATE INDEX idx_id_date ON model_downloads(id, snapshot_date);")
        cur.execute("CREATE INDEX idx_snapshot_date ON model_downloads(snapshot_date);")
        cur.execute("ANALYZE model_downloads;")
        conn.commit()
        logging.info("  ✓ Indexes created")

        load_time = time.time() - start_load
        logging.info(f"✓ Data loaded and indexed in {load_time:.2f}s")

        # Get table size
        cur.execute("""
            SELECT pg_size_pretty(pg_total_relation_size('model_downloads'));
        """)
        table_size = cur.fetchone()[0]
        logging.info(f"  Table size: {table_size}")

        # Get row count
        cur.execute("SELECT COUNT(*) FROM model_downloads;")
        row_count = cur.fetchone()[0]
        logging.info(f"  Rows loaded: {row_count:,}")

        # Compute daily downloads
        logging.info("Computing daily downloads by author...")
        start_compute = time.time()

        query = """
        WITH daily_diffs AS (
            SELECT
                SPLIT_PART(id, '/', 1) AS author,
                id AS model_id,
                snapshot_date,
                downloads_all_time,
                downloads_all_time - LAG(downloads_all_time) OVER (
                    PARTITION BY id
                    ORDER BY snapshot_date
                ) AS daily_downloads
            FROM model_downloads
        ),
        cleaned_diffs AS (
            SELECT
                author,
                snapshot_date,
                CASE
                    WHEN daily_downloads IS NULL THEN downloads_all_time
                    WHEN daily_downloads < 0 THEN 0
                    ELSE daily_downloads
                END AS daily_downloads
            FROM daily_diffs
        )
        SELECT
            author,
            snapshot_date,
            SUM(daily_downloads) AS total_daily_downloads
        FROM cleaned_diffs
        GROUP BY author, snapshot_date
        ORDER BY author, snapshot_date;
        """

        cur.execute(query)
        results = cur.fetchall()

        compute_time = time.time() - start_compute
        logging.info(f"✓ Computation complete in {compute_time:.2f}s")
        logging.info(f"  Result: {len(results):,} author-date combinations")

        # Convert to DataFrame and save
        logging.info(f"Saving results to {output_path}...")
        result_df = pd.DataFrame(results, columns=['author', 'snapshot_date', 'total_daily_downloads'])

        # Convert total_daily_downloads to numeric (PostgreSQL returns Decimal objects)
        result_df['total_daily_downloads'] = pd.to_numeric(result_df['total_daily_downloads'])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_path, index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logging.info(f"✓ Results saved: {file_size_mb:.2f} MB")

        # Show sample results
        logging.info("\nSample results (top 10 by total downloads):")
        sample = result_df.nlargest(10, 'total_daily_downloads')
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

    finally:
        # Cleanup
        logging.info("\nCleaning up...")
        cur.execute("DROP TABLE IF EXISTS model_downloads;")
        conn.commit()
        cur.close()
        conn.close()
        logging.info("✓ PostgreSQL connection closed")
        logging.info("\n✓ Analysis complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze historical HuggingFace downloads using PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full dataset
  uv run python scratch/analyze_historical_postgres.py --db-user danielyue

  # Process first 1M rows only
  uv run python scratch/analyze_historical_postgres.py --db-user danielyue --limit 1000000

  # Custom database settings
  uv run python scratch/analyze_historical_postgres.py \\
    --db-host localhost \\
    --db-port 5432 \\
    --db-name hf_models \\
    --db-user postgres \\
    --db-password mypassword

Note: Requires psycopg2-binary package (included in project dependencies)
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
        default=Path(__file__).parent.parent / 'data' / 'processed' / 'author_daily_downloads_postgres.csv',
        help='Output CSV file path (default: ../data/processed/author_daily_downloads_postgres.csv)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of rows to process (for testing)'
    )

    parser.add_argument(
        '--db-name',
        type=str,
        default='hf_models',
        help='PostgreSQL database name (default: hf_models)'
    )

    parser.add_argument(
        '--db-user',
        type=str,
        default='postgres',
        help='PostgreSQL username (default: postgres)'
    )

    parser.add_argument(
        '--db-password',
        type=str,
        default='postgres',
        help='PostgreSQL password (default: postgres)'
    )

    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )

    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
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
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            db_host=args.db_host,
            db_port=args.db_port,
            limit=args.limit
        )
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()
