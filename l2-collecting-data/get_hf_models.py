"""
Get HuggingFace Models

This script fetches model metadata from the HuggingFace Models API and saves to JSONL.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List
from hf_client import get_client

logger = logging.getLogger(__name__)


def get_hf_models(
    n: int = 100,
    search: Optional[str] = None,
    author: Optional[str] = None,
    filter: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
    library: Optional[str] = None,
    language: Optional[str] = None,
    tags: Optional[List[str]] = None,
    sort: str = "downloads",
    direction: int = -1,
    full: bool = False,
    requests_per_second: int = 5,
    output_file: Optional[str] = None
) -> List[dict]:
    """
    Fetch top N models from HuggingFace API

    Args:
        n: Number of models to fetch (default: 100)
        search: Search by model name
        author: Filter by author/organization
        filter: Filter by task (e.g., "text-classification")
        pipeline_tag: Filter by pipeline tag
        library: Filter by library (pytorch, tensorflow, etc.)
        language: Filter by language
        tags: Filter by specific tags (list)
        sort: Sort field (trending, downloads, likes, updated) - default: downloads
        direction: Sort direction (1 ascending, -1 descending) - default: -1
        full: Include full model details (default: False)
        requests_per_second: Rate limit for API requests (default: 5)
        output_file: Path to output JSONL file (default: data/models.jsonl)

    Returns:
        List of model metadata dictionaries
    """
    # Set default output file
    if output_file is None:
        output_file = Path(__file__).parent / "data" / "models.jsonl"
    else:
        output_file = Path(output_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Build API parameters
    params = {
        'sort': sort,
        'direction': direction,
    }

    if search is not None:
        params['search'] = search
    if author is not None:
        params['author'] = author
    if filter is not None:
        params['filter'] = filter
    if pipeline_tag is not None:
        params['pipeline_tag'] = pipeline_tag
    if library is not None:
        params['library'] = library
    if language is not None:
        params['language'] = language
    if tags is not None:
        params['tags'] = ','.join(tags) if isinstance(tags, list) else tags
    if full:
        params['full'] = 'true'

    # Fetch models in batches (API max is 1000 per request)
    models = []
    remaining = n
    offset = 0
    batch_size = min(1000, n)  # Max 1000 per request

    with get_client(requests_per_second=requests_per_second) as client:
        logger.info(f"Fetching top {n} models from HuggingFace API...")
        logger.info(f"Parameters: {params}")

        while remaining > 0:
            # Adjust limit for last batch
            current_limit = min(batch_size, remaining)
            params['limit'] = current_limit

            # Make API request
            url = "https://huggingface.co/api/models"
            response = client.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                break

            batch = response.json()

            if not batch:
                logger.warning(f"No more models returned. Got {len(models)} total.")
                break

            models.extend(batch)
            remaining -= len(batch)
            offset += len(batch)

            logger.info(f"Fetched {len(models)}/{n} models...")

            # If we got fewer models than requested, we've reached the end
            if len(batch) < current_limit:
                logger.info(f"Reached end of available models. Got {len(models)} total.")
                break

    # Save to JSONL file
    logger.info(f"Saving {len(models)} models to {output_file}")
    with open(output_file, 'w') as f:
        for model in models:
            f.write(json.dumps(model) + '\n')

    logger.info(f"Successfully saved {len(models)} models to {output_file}")

    return models


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Get top 100 models by downloads
    models = get_hf_models(n=100)
    print(f"\nFetched {len(models)} models")
    if models:
        print(f"\nTop model: {models[0].get('id', 'Unknown')}")
        print(f"Downloads: {models[0].get('downloads', 0):,}")
