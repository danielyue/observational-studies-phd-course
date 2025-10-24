"""
Clean Models Data
Processes the raw models.parquet file and creates a cleaned CSV with essential columns.
"""
import pandas as pd
import logging
from pathlib import Path


def extract_model_name(model_id):
    """
    Extract model name from the full model ID.

    Args:
        model_id: Full model ID in format "author/model-name"

    Returns:
        Model name or None if invalid format
    """
    if pd.isna(model_id):
        return None

    parts = str(model_id).split('/')
    if len(parts) >= 2:
        return '/'.join(parts[1:])  # Everything after the first slash
    return model_id  # Return as-is if no slash found


def extract_base_model_info(base_models_dict):
    """
    Extract base model ID and relation from baseModels field.

    Args:
        base_models_dict: Dictionary with 'models' list and 'relation' string

    Returns:
        Tuple of (base_model_id, relation) or (None, None)
    """
    if pd.isna(base_models_dict) or not isinstance(base_models_dict, dict):
        return None, None

    # Extract relation
    relation = base_models_dict.get('relation', None)

    # Extract first base model ID
    models_list = base_models_dict.get('models', [])

    # Check if models_list exists and is not empty
    if models_list is not None and len(models_list) > 0:
        first_model = models_list[0]
        if isinstance(first_model, dict):
            base_model_id = first_model.get('id', None)
            return base_model_id, relation

    return None, relation


def extract_parameters(safetensors_dict):
    """
    Extract total parameter count from safetensors field.

    Args:
        safetensors_dict: Dictionary with 'total' and 'parameters' fields

    Returns:
        Total parameter count or None
    """
    if pd.isna(safetensors_dict) or not isinstance(safetensors_dict, dict):
        return None

    total = safetensors_dict.get('total', None)

    # Ensure it's a valid number
    if total is not None and total > 0:
        return total

    return None


def clean_models_data(input_path, output_path):
    """
    Clean and transform models.parquet into a simplified CSV.

    Args:
        input_path: Path to raw models.parquet file
        output_path: Path to save cleaned models.csv file
    """
    logging.info(f"Starting models data cleaning from {input_path}")

    # Read the parquet file
    logging.info("Loading models.parquet...")
    df = pd.read_parquet(input_path)
    logging.info(f"Loaded {len(df):,} models")

    # Create the cleaned dataframe
    logging.info("Extracting and transforming columns...")

    cleaned_df = pd.DataFrame({
        '_id': df['_id'],
        'author': df['author'],
        'model_name': df['id'].apply(extract_model_name),
        'created_at': df['createdAt'],
        'likes': df['likes'],
        'downloads_last30': df['downloads'],
        'downloads_all_time': df['downloadsAllTime'],
        'n_parameters': df['safetensors'].apply(extract_parameters),
        'pipeline_tag': df['pipeline_tag'],
    })

    # Extract base model information (requires two columns)
    logging.info("Extracting base model relationships...")
    base_model_info = df['baseModels'].apply(extract_base_model_info)
    cleaned_df['base_model_id'] = base_model_info.apply(lambda x: x[0])
    cleaned_df['base_model_relation'] = base_model_info.apply(lambda x: x[1])

    # Log statistics about the cleaned data
    logging.info("Cleaning complete. Data summary:")
    logging.info(f"  Total models: {len(cleaned_df):,}")
    logging.info(f"  Models with author: {cleaned_df['author'].notna().sum():,}")
    logging.info(f"  Models with likes: {(cleaned_df['likes'] > 0).sum():,}")
    logging.info(f"  Models with recent downloads: {(cleaned_df['downloads_last30'] > 0).sum():,}")
    logging.info(f"  Models with parameter counts: {cleaned_df['n_parameters'].notna().sum():,}")
    logging.info(f"  Models with base model info: {cleaned_df['base_model_id'].notna().sum():,}")
    logging.info(f"  Models with pipeline tags: {cleaned_df['pipeline_tag'].notna().sum():,}")

    # Save to CSV
    logging.info(f"Saving cleaned data to {output_path}")
    cleaned_df.to_csv(output_path, index=False)

    # Log file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Saved models.csv ({file_size_mb:.2f} MB)")

    # Generate author-year aggregated statistics
    author_year_stats_path = output_path.parent / 'author_year_stats.csv'
    create_author_year_stats(cleaned_df, author_year_stats_path)

    return cleaned_df


def create_author_year_stats(cleaned_df, output_path):
    """
    Create aggregated statistics by author and year.

    Args:
        cleaned_df: Cleaned models DataFrame
        output_path: Path to save author_year_stats.csv file
    """
    logging.info("Creating author-year aggregated statistics...")

    # Extract year from created_at
    cleaned_df['year'] = pd.to_datetime(cleaned_df['created_at']).dt.year

    # Group by author and year
    grouped = cleaned_df.groupby(['author', 'year']).agg(
        n_models=('_id', 'count'),
        total_likes=('likes', 'sum'),
        avg_likes=('likes', 'mean'),
        total_downloads_last30=('downloads_last30', 'sum'),
        avg_downloads_last30=('downloads_last30', 'mean'),
        total_downloads_all_time=('downloads_all_time', 'sum'),
        avg_downloads_all_time=('downloads_all_time', 'mean')
    ).reset_index()

    # Sort by year and total downloads (descending)
    grouped = grouped.sort_values(['year', 'total_downloads_all_time'],
                                   ascending=[True, False])

    # Log statistics
    logging.info(f"  Total author-year combinations: {len(grouped):,}")
    logging.info(f"  Years covered: {grouped['year'].min():.0f} - {grouped['year'].max():.0f}")
    logging.info(f"  Unique authors: {grouped['author'].nunique():,}")

    # Save to CSV
    logging.info(f"Saving author-year statistics to {output_path}")
    grouped.to_csv(output_path, index=False)

    # Log file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"Saved author_year_stats.csv ({file_size_mb:.2f} MB)")

    return grouped
