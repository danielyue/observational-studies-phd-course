"""
Get HuggingFace Organizations

This script scrapes metadata for multiple HuggingFace organizations and saves to JSONL.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from scrape_hf_organization import scrape_hf_organization

logger = logging.getLogger(__name__)


def get_hf_organizations(
    org_names: List[str],
    requests_per_second: int = 5,
    output_file: Optional[str] = None
) -> List[dict]:
    """
    Scrape metadata for multiple HuggingFace organizations

    Args:
        org_names: List of organization usernames to scrape
        requests_per_second: Rate limit for API requests (default: 5)
        output_file: Path to output JSONL file (default: data/organizations.jsonl)

    Returns:
        List of organization profile dictionaries
    """
    # Set default output file
    if output_file is None:
        output_file = Path(__file__).parent / "data" / "organizations.jsonl"
    else:
        output_file = Path(output_file)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scraping {len(org_names)} organizations...")

    profiles = []

    # Open file for writing (append mode so we don't lose data if interrupted)
    with open(output_file, 'w') as f:
        for i, org_name in enumerate(org_names, 1):
            logger.info(f"[{i}/{len(org_names)}] Scraping {org_name}...")

            try:
                profile = scrape_hf_organization(
                    org_name=org_name,
                    requests_per_second=requests_per_second
                )
                profiles.append(profile)

                # Write immediately to file (in case of interruption)
                f.write(json.dumps(profile) + '\n')
                f.flush()  # Ensure it's written to disk

                logger.info(f"✓ Successfully scraped {org_name}")

            except Exception as e:
                logger.error(f"✗ Failed to scrape {org_name}: {e}")
                # Still write the error record
                error_profile = {
                    'organization': org_name,
                    'error': str(e)
                }
                profiles.append(error_profile)
                f.write(json.dumps(error_profile) + '\n')
                f.flush()

    logger.info(f"Successfully scraped {len(profiles)} organizations to {output_file}")

    return profiles


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Scrape a few popular organizations
    test_orgs = ['meta-llama', 'huggingface', 'openai', 'google', 'microsoft']

    profiles = get_hf_organizations(test_orgs)
    print(f"\nScraped {len(profiles)} organizations")
    for profile in profiles:
        org_name = profile.get('organization')
        if profile.get('error'):
            print(f"  ✗ {org_name}: {profile['error']}")
        else:
            model_count = profile.get('api_data', {}).get('models', {}).get('count', 0)
            print(f"  ✓ {org_name}: {model_count} models")
