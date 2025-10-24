"""
Stage A: Data Cleaning and Filtering
Reads raw CSV data and performs basic cleaning operations.
"""
import pandas as pd
import logging


def process_data(input_path, output_path):
    """
    Process raw data: filter active records and calculate metrics.

    Args:
        input_path: Path to raw CSV file
        output_path: Path to save cleaned CSV file
    """
    logging.info(f"Stage A: Reading data from {input_path}")

    # Read raw data
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} records")

    # Filter for active status only
    df_clean = df[df['status'] == 'active'].copy()
    logging.info(f"Filtered to {len(df_clean)} active records")

    # Add computed column
    df_clean['value_doubled'] = df_clean['value'] * 2

    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    logging.info(f"Stage A: Saved cleaned data to {output_path}")

    return df_clean
