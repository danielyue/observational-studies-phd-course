"""
Scrape HuggingFace Organization Metadata

This script scrapes organization metadata from HuggingFace organization pages.
Since there is no dedicated organizations API, we must scrape HTML and combine with API data.
"""

import json
import logging
import html
import requests
from typing import Dict, Any
from bs4 import BeautifulSoup
from hf_client import get_client

logger = logging.getLogger(__name__)


def scrape_hf_organization(
    org_name: str,
    requests_per_second: int = 5
) -> Dict[str, Any]:
    """
    Scrape metadata for a single HuggingFace organization

    Args:
        org_name: Organization username/identifier
        requests_per_second: Rate limit for API requests (default: 5)

    Returns:
        Dictionary containing organization metadata
    """
    logger.info(f"Scraping organization: {org_name}")

    profile = {
        'organization': org_name,
        'basic_info': {},
        'follower_info': {},
        'social_links': {},
        'content': {},
        'api_data': {},
        'error': None
    }

    # Note: HuggingFace rejects Authorization headers for HTML pages,
    # so we use plain requests for HTML scraping but authenticated client for API calls
    try:
        # Scrape organization page HTML (without auth)
        org_url = f"https://huggingface.co/{org_name}"
        logger.debug(f"Fetching {org_url}")
        response = requests.get(org_url, headers={'User-Agent': 'HF-Data-Collection-Script/1.0'})

        if response.status_code != 200:
            logger.error(f"Failed to fetch organization page: {response.status_code}")
            profile['error'] = f"HTTP {response.status_code}"
            return profile

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract JSON data from data-props attributes
        data_props_elements = soup.find_all(attrs={'data-props': True})

        for element in data_props_elements:
            try:
                # Decode HTML entities and parse JSON
                props_json = html.unescape(element['data-props'])
                props_data = json.loads(props_json)

                # Extract basic info
                if 'userProfile' in props_data:
                    user_profile = props_data['userProfile']
                    profile['basic_info'] = {
                        'name': user_profile.get('name'),
                        'fullname': user_profile.get('fullname'),
                        'email': user_profile.get('email'),
                        'plan': user_profile.get('plan'),
                        'is_enterprise': user_profile.get('isEnterprise', False),
                        'is_pro': user_profile.get('isPro', False),
                        'created_at': user_profile.get('createdAt'),
                    }

                # Extract follower info
                if 'followers' in props_data:
                    followers_data = props_data['followers']
                    profile['follower_info'] = {
                        'follower_count': followers_data.get('count', 0),
                        'sample_followers': [
                            {
                                'user': f.get('user'),
                                'fullname': f.get('fullname'),
                                'avatarUrl': f.get('avatarUrl')
                            }
                            for f in followers_data.get('followers', [])[:10]  # First 10
                        ]
                    }

                # Extract card/README data
                if 'cardData' in props_data:
                    profile['content']['card'] = props_data['cardData']

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Could not parse data-props: {e}")
                continue

        # Extract social links from HTML
        social_links = {}

        # Look for Twitter/X link
        twitter_link = soup.find('a', href=lambda x: x and ('twitter.com' in x or 'x.com' in x))
        if twitter_link:
            social_links['twitter'] = twitter_link.get('href')

        # Look for GitHub link
        github_link = soup.find('a', href=lambda x: x and 'github.com' in x)
        if github_link:
            social_links['github'] = github_link.get('href')

        # Look for website link
        website_link = soup.find('a', attrs={'rel': 'noopener nofollow'})
        if website_link:
            href = website_link.get('href', '')
            if 'http' in href and 'huggingface.co' not in href:
                social_links['website'] = href

        profile['social_links'] = social_links

        # Use authenticated client for API calls
        with get_client(requests_per_second=requests_per_second) as client:
            # Get models via API
            logger.debug(f"Fetching models for {org_name}")
            models_response = client.get(
                "https://huggingface.co/api/models",
                params={'author': org_name, 'limit': 1000}
            )

            if models_response.status_code == 200:
                models = models_response.json()
                profile['api_data']['models'] = {
                    'count': len(models),
                    'total_likes': sum(m.get('likes', 0) for m in models),
                    'total_downloads': sum(m.get('downloads', 0) for m in models),
                    'models': [
                        {
                            'id': m.get('id'),
                            'likes': m.get('likes', 0),
                            'downloads': m.get('downloads', 0),
                            'pipeline_tag': m.get('pipeline_tag'),
                            'tags': m.get('tags', [])
                        }
                        for m in models
                    ]
                }
            else:
                logger.warning(f"Failed to fetch models: {models_response.status_code}")

            # Get datasets via API
            logger.debug(f"Fetching datasets for {org_name}")
            datasets_response = client.get(
                "https://huggingface.co/api/datasets",
                params={'author': org_name, 'limit': 1000}
            )

            if datasets_response.status_code == 200:
                datasets = datasets_response.json()
                profile['api_data']['datasets'] = {
                    'count': len(datasets),
                    'total_likes': sum(d.get('likes', 0) for d in datasets),
                    'datasets': [
                        {
                            'id': d.get('id'),
                            'likes': d.get('likes', 0),
                            'downloads': d.get('downloads', 0) if 'downloads' in d else None
                        }
                        for d in datasets
                    ]
                }
            else:
                logger.warning(f"Failed to fetch datasets: {datasets_response.status_code}")

            # Get spaces via API
            logger.debug(f"Fetching spaces for {org_name}")
            spaces_response = client.get(
                "https://huggingface.co/api/spaces",
                params={'author': org_name, 'limit': 1000}
            )

            if spaces_response.status_code == 200:
                spaces = spaces_response.json()
                profile['api_data']['spaces'] = {
                    'count': len(spaces),
                    'total_likes': sum(s.get('likes', 0) for s in spaces),
                    'spaces': [
                        {
                            'id': s.get('id'),
                            'likes': s.get('likes', 0)
                        }
                        for s in spaces
                    ]
                }
            else:
                logger.warning(f"Failed to fetch spaces: {spaces_response.status_code}")

        logger.info(f"Successfully scraped {org_name}")

    except Exception as e:
        logger.error(f"Error scraping {org_name}: {e}", exc_info=True)
        profile['error'] = str(e)

    return profile


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example: Scrape meta-llama organization
    profile = scrape_hf_organization('meta-llama')
    print(json.dumps(profile, indent=2))
