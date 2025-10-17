"""
HuggingFace Organization Metadata Scraper

This script demonstrates how to scrape organization metadata from HuggingFace
organization profile pages. The data is embedded in HTML as JSON in data-props attributes.
"""

import requests
from bs4 import BeautifulSoup
import json
import html
import re
from typing import Dict, List, Optional


def scrape_organization_metadata(org_name: str) -> Dict:
    """
    Scrape metadata from a HuggingFace organization profile page.

    Args:
        org_name: Organization username (e.g., 'huggingface', 'meta-llama')

    Returns:
        Dictionary containing organization metadata
    """
    url = f"https://huggingface.co/{org_name}"

    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all elements with data-props attribute
    data_props_elements = soup.find_all(attrs={'data-props': True})

    metadata = {
        'organization': org_name,
        'url': url,
        'basic_info': {},
        'follower_info': {},
        'organization_card': {},
        'social_links': {},
        'raw_data': []
    }

    # Extract data from each data-props element
    for element in data_props_elements:
        props_raw = element.get('data-props', '')

        # Decode HTML entities
        props_decoded = html.unescape(props_raw)

        try:
            props_json = json.loads(props_decoded)
            metadata['raw_data'].append(props_json)

            # Extract organization basic info
            if 'org' in props_json:
                org_data = props_json['org']
                if 'name' in org_data and org_data.get('name') == org_name:
                    metadata['basic_info'] = {
                        'name': org_data.get('name'),
                        'fullname': org_data.get('fullname'),
                        'email': org_data.get('email'),
                        'avatar_url': org_data.get('avatarUrl'),
                        'type': org_data.get('type'),
                        'plan': org_data.get('plan'),
                        'is_enterprise': org_data.get('isEnterprise', False),
                        'is_hf': org_data.get('isHf', False),
                        'details': org_data.get('details', '')
                    }

            # Extract follower information
            if 'followerCount' in props_json:
                metadata['follower_info'] = {
                    'follower_count': props_json.get('followerCount'),
                    'is_following': props_json.get('isFollowing', False),
                    'can_quick_join': props_json.get('canQuickJoin', False),
                    'request_sent': props_json.get('requestSent', False),
                    'sample_followers': props_json.get('sampleFollowers', [])
                }

            # Extract organization card/README
            if 'organizationCard' in props_json:
                card = props_json['organizationCard']
                metadata['organization_card'] = {
                    'metadata': card.get('metadata', {}),
                    'contents': card.get('contents', ''),
                    'html': card.get('html', '')
                }

        except json.JSONDecodeError:
            continue

    # Extract social links from page
    social_links = extract_social_links(soup)
    metadata['social_links'] = social_links

    # Extract team member count from page
    team_count = extract_team_count(soup)
    if team_count:
        metadata['team_member_count'] = team_count

    # Extract content counts from page
    content_counts = extract_content_counts(soup)
    metadata['content_counts'] = content_counts

    return metadata


def extract_social_links(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract social media links from organization page."""
    social_links = {}

    # Find Twitter link
    twitter_link = soup.find('a', href=re.compile(r'twitter\.com/'))
    if twitter_link:
        social_links['twitter'] = twitter_link['href']

    # Find GitHub link
    github_link = soup.find('a', href=re.compile(r'github\.com/'))
    if github_link:
        social_links['github'] = github_link['href']

    # Find website link (look for https:// links that aren't social media)
    website_links = soup.find_all('a', href=re.compile(r'^https?://'))
    for link in website_links:
        href = link['href']
        if not any(domain in href for domain in ['twitter.com', 'github.com', 'huggingface.co']):
            social_links['website'] = href
            break

    return social_links


def extract_team_count(soup: BeautifulSoup) -> Optional[int]:
    """Extract team member count from page."""
    # Look for team members heading
    team_heading = soup.find(string=re.compile(r'Team members \d+'))
    if team_heading:
        match = re.search(r'Team members (\d+)', team_heading)
        if match:
            return int(match.group(1))
    return None


def extract_content_counts(soup: BeautifulSoup) -> Dict[str, int]:
    """Extract counts of models, datasets, and spaces."""
    counts = {}

    # Look for models count
    models_link = soup.find('a', href=re.compile(r'/models$'))
    if models_link:
        text = models_link.get_text()
        match = re.search(r'models (\d+)', text, re.IGNORECASE)
        if match:
            counts['models'] = int(match.group(1))

    # Look for datasets count
    datasets_link = soup.find('a', href=re.compile(r'/datasets$'))
    if datasets_link:
        text = datasets_link.get_text()
        match = re.search(r'datasets (\d+)', text, re.IGNORECASE)
        if match:
            counts['datasets'] = int(match.group(1))

    # Look for spaces count
    spaces_link = soup.find('a', href=re.compile(r'/spaces$'))
    if spaces_link:
        text = spaces_link.get_text()
        match = re.search(r'spaces (\d+)', text, re.IGNORECASE)
        if match:
            counts['spaces'] = int(match.group(1))

    return counts


def get_full_organization_profile(org_name: str) -> Dict:
    """
    Get complete organization profile combining scraped metadata and API data.

    Args:
        org_name: Organization username

    Returns:
        Complete organization profile with both scraped and API data
    """
    # Scrape metadata from HTML
    scraped_data = scrape_organization_metadata(org_name)

    # Get content from APIs
    base_url = "https://huggingface.co/api"

    try:
        # Fetch models
        models_response = requests.get(f"{base_url}/models", params={
            'author': org_name,
            'limit': 1000
        })
        models = models_response.json() if models_response.status_code == 200 else []

        # Fetch datasets
        datasets_response = requests.get(f"{base_url}/datasets", params={
            'author': org_name,
            'limit': 1000
        })
        datasets = datasets_response.json() if datasets_response.status_code == 200 else []

        # Fetch spaces
        spaces_response = requests.get(f"{base_url}/spaces", params={
            'author': org_name,
            'limit': 1000
        })
        spaces = spaces_response.json() if spaces_response.status_code == 200 else []

        # Combine all data
        profile = {
            **scraped_data,
            'api_data': {
                'models': {
                    'count': len(models),
                    'total_likes': sum(m.get('likes', 0) for m in models),
                    'total_downloads': sum(m.get('downloads', 0) for m in models),
                    'items': models
                },
                'datasets': {
                    'count': len(datasets),
                    'total_likes': sum(d.get('likes', 0) for d in datasets),
                    'total_downloads': sum(d.get('downloads', 0) for d in datasets),
                    'items': datasets
                },
                'spaces': {
                    'count': len(spaces),
                    'total_likes': sum(s.get('likes', 0) for s in spaces),
                    'items': spaces
                }
            }
        }

        return profile

    except Exception as e:
        print(f"Error fetching API data: {e}")
        return scraped_data


# Example usage
if __name__ == "__main__":
    # Example 1: Scrape basic metadata
    print("=" * 80)
    print("Example 1: Scrape HuggingFace organization metadata")
    print("=" * 80)

    metadata = scrape_organization_metadata('huggingface')

    print(f"\nOrganization: {metadata['basic_info'].get('fullname')}")
    print(f"Name: {metadata['basic_info'].get('name')}")
    print(f"Email: {metadata['basic_info'].get('email')}")
    print(f"Plan: {metadata['basic_info'].get('plan')}")
    print(f"Enterprise: {metadata['basic_info'].get('is_enterprise')}")
    print(f"Type: {metadata['basic_info'].get('type')}")
    print(f"Follower Count: {metadata['follower_info'].get('follower_count')}")
    print(f"Team Members: {metadata.get('team_member_count')}")

    print("\nSocial Links:")
    for platform, url in metadata['social_links'].items():
        print(f"  {platform}: {url}")

    print("\nContent Counts:")
    for content_type, count in metadata['content_counts'].items():
        print(f"  {content_type}: {count}")

    # Example 2: Get full profile with API data
    print("\n" + "=" * 80)
    print("Example 2: Get full organization profile (metadata + API data)")
    print("=" * 80)

    full_profile = get_full_organization_profile('meta-llama')

    print(f"\nOrganization: {full_profile['basic_info'].get('fullname')}")
    print(f"Follower Count: {full_profile['follower_info'].get('follower_count')}")
    print(f"Models: {full_profile['api_data']['models']['count']}")
    print(f"Total Model Likes: {full_profile['api_data']['models']['total_likes']}")
    print(f"Total Model Downloads: {full_profile['api_data']['models']['total_downloads']}")
    print(f"Datasets: {full_profile['api_data']['datasets']['count']}")
    print(f"Spaces: {full_profile['api_data']['spaces']['count']}")

    # Save to JSON file
    output_file = f"{full_profile['basic_info'].get('name')}_profile.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Remove the full items list to keep file smaller
        output_data = {**full_profile}
        output_data['api_data']['models']['items'] = output_data['api_data']['models']['items'][:5]
        output_data['api_data']['datasets']['items'] = output_data['api_data']['datasets']['items'][:5]
        output_data['api_data']['spaces']['items'] = output_data['api_data']['spaces']['items'][:5]
        output_data['raw_data'] = []  # Remove raw data

        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nFull profile saved to: {output_file}")
