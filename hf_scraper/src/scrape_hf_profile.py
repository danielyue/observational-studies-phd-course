"""
Scrape HuggingFace Profile Raw Data

This script scrapes raw data-props elements from HuggingFace profile pages
(users or organizations) and merges them into a single dictionary.
"""

import json
import logging
import html
import requests
import argparse
from typing import Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def scrape_hf_profile(
    profile_name: str,
) -> Dict[str, Any]:
    """
    Scrape raw data-props elements from a HuggingFace profile page

    Args:
        profile_name: Profile username/identifier (user or organization)

    Returns:
        Dictionary containing merged data-props elements
    """
    logger.info(f"Scraping profile: {profile_name}")

    result = {
        'profile': profile_name,
        'data': {},
        'error': None
    }

    try:
        # Fetch profile page HTML
        profile_url = f"https://huggingface.co/{profile_name}"
        logger.debug(f"Fetching {profile_url}")
        response = requests.get(profile_url, headers={'User-Agent': 'HF-Data-Collection-Script/1.0'})

        if response.status_code != 200:
            logger.error(f"Failed to fetch profile page: {response.status_code}")
            result['error'] = f"HTTP {response.status_code}"
            return result

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all JSON data from data-props attributes
        data_props_elements = soup.find_all(attrs={'data-props': True})
        logger.info(f"Found {len(data_props_elements)} data-props elements")

        # Merge all data-props dictionaries together
        # Strategy: Keep the most complete version of each key (the one with most data)
        merged_data = {}

        for i, element in enumerate(data_props_elements):
            try:
                # Decode HTML entities and parse JSON
                props_json = html.unescape(element['data-props'])
                props_data = json.loads(props_json)

                logger.debug(f"Element {i}: {len(props_data)} keys = {list(props_data.keys())}")

                # Merge this element's data into the main dictionary
                # For each key, keep the value with more content (prefer non-empty, larger structures)
                for key, value in props_data.items():
                    if key not in merged_data:
                        # New key, just add it
                        merged_data[key] = value
                    else:
                        # Key exists - keep the more complete version
                        existing_value = merged_data[key]

                        # If new value is "better" (more complete), use it
                        # Heuristic: prefer non-None, non-empty, or larger data structures
                        if existing_value is None and value is not None:
                            merged_data[key] = value
                        elif isinstance(value, (dict, list)) and isinstance(existing_value, (dict, list)):
                            # For dicts/lists, keep the one with more items
                            if len(value) > len(existing_value):
                                merged_data[key] = value
                        elif value and not existing_value:
                            # New value is truthy, old is falsy
                            merged_data[key] = value

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse data-props element {i}: {e}")
                continue

        result['data'] = merged_data
        logger.info(f"Successfully scraped {profile_name} - merged {len(data_props_elements)} elements into {len(merged_data)} keys")

        # If this is an organization page, scrape additional header metadata
        if merged_data.get('org'):
            logger.info("Detected organization profile - scraping header metadata")
            header_metadata = scrape_org_header(soup, profile_name)
            result['header_metadata'] = header_metadata
            logger.info(f"Scraped {len(header_metadata)} header metadata items")

    except Exception as e:
        logger.error(f"Error scraping {profile_name}: {e}", exc_info=True)
        result['error'] = str(e)

    return result


def scrape_org_header(soup: BeautifulSoup, profile_name: str) -> Dict[str, Any]:
    """
    Scrape header metadata from organization page.

    This captures information not in data-props like:
    - Organization display name
    - Tags/badges
    - Links (website, social media)

    Targets the specific container 2 levels down from <header> element.

    Args:
        soup: BeautifulSoup object of the page
        profile_name: Name of the profile being scraped

    Returns:
        Dictionary with header metadata
    """
    header_data = {
        'org_display_name': None,
        'tags': [],
        'links': []
    }

    try:
        # Find the profile <header> element (not the navigation header)
        # Profile header typically has bg-linear-to-t or from-gray classes
        header_element = None
        for header in soup.find_all('header'):
            classes = ' '.join(header.get('class', []))
            # Profile header has gradient background classes
            if 'bg-linear-to' in classes or 'from-gray' in classes:
                header_element = header
                break

        if not header_element:
            logger.warning("Could not find profile <header> element")
            return header_data

        logger.debug(f"Found profile header element with classes: {header_element.get('class', [])}")

        # Navigate down to find the target container with org info
        # Structure: <header> -> <div class="container..."> -> <div class="mb-4 items-center...">
        #            -> <div class="overflow-hidden"> -> <div class="mb-3.5 items-center...">
        #            -> <div class="flex items-center space-x-2"> (this contains tags)

        # Level 1: Find container div
        container_div = header_element.find('div', class_=lambda x: x and 'container' in x)
        if not container_div:
            logger.warning("Could not find container div (level 1)")
            return header_data

        logger.debug(f"Found container div with classes: {container_div.get('class', [])}")

        # Find the div containing overflow-hidden which has the org info
        overflow_div = container_div.find('div', class_=lambda x: x and 'overflow-hidden' in x)
        if not overflow_div:
            # Try alternative: just find div with items-center
            overflow_div = container_div.find('div', class_=lambda x: x and 'items-center' in x)

        if not overflow_div:
            logger.warning("Could not find overflow/items-center div")
            return header_data

        logger.debug(f"Found overflow div with classes: {overflow_div.get('class', [])}")

        # Find the inner div with the actual content (mb-3 or items-center)
        target_div = overflow_div.find('div', class_=lambda x: x and any(cls.startswith('mb-3') for cls in x))
        if not target_div:
            # Try alternative
            target_div = overflow_div.find('div', class_=lambda x: x and 'items-center' in x)

        if not target_div:
            logger.warning("Could not find target div with content")
            return header_data

        logger.debug(f"Found target div with classes: {target_div.get('class', [])}")

        # Now extract all content from this specific div only

        # 1. Extract organization display name (h1 within this div)
        h1 = target_div.find('h1')
        if h1:
            header_data['org_display_name'] = h1.get_text(strip=True)
            logger.debug(f"Found org display name: {header_data['org_display_name']}")

        # 2. Extract tags - look for divs/spans with tag-like styling within this container
        for element in target_div.find_all(['div', 'span']):
            classes = ' '.join(element.get('class', []))
            text = element.get_text(strip=True)

            # Skip if it's the h1 element or empty
            if element.name == 'h1' or not text:
                continue

            # Skip if already captured
            if any(t['text'] == text for t in header_data['tags']):
                continue

            # Tag patterns: rounded, inline-flex, border, background colors
            # Look for elements that have these styling patterns
            is_tag_like = any(pattern in classes for pattern in [
                'rounded-', 'inline-flex', 'inline-block', 'border', 'bg-', 'px-', 'py-'
            ])

            # Must be reasonably short and have no deep nesting
            if is_tag_like and len(text) < 30 and len(element.find_all(recursive=True)) <= 2:
                tag_info = {
                    'text': text,
                    'classes': element.get('class', [])
                }
                header_data['tags'].append(tag_info)
                logger.debug(f"Found tag: {text}")

        # 3. Extract links - search in overflow_div for broader scope
        # Links may appear outside the immediate target_div
        for link in overflow_div.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            title = link.get('title', '')

            # Identify external links (social media, website, etc.)
            external_patterns = ['http://', 'https://']

            if any(pattern in href for pattern in external_patterns):
                # Skip if it's the profile's own HuggingFace link
                if 'huggingface.co' not in href or profile_name not in href:
                    # Skip internal links like /enterprise
                    if not href.startswith('/'):
                        link_info = {
                            'url': href,
                            'text': text if text else title,
                            'title': title,
                            'type': identify_link_type(href)
                        }

                        # Avoid duplicates
                        if not any(l['url'] == href for l in header_data['links']):
                            header_data['links'].append(link_info)
                            logger.debug(f"Found link: {href}")

    except Exception as e:
        logger.warning(f"Error scraping organization header: {e}")

    return header_data


def identify_link_type(url: str) -> str:
    """Identify the type of external link."""
    url_lower = url.lower()

    if 'github.com' in url_lower:
        return 'github'
    elif 'twitter.com' in url_lower or 'x.com' in url_lower:
        return 'twitter'
    elif 'linkedin.com' in url_lower:
        return 'linkedin'
    elif 'facebook.com' in url_lower:
        return 'facebook'
    elif 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'youtube'
    elif 'discord' in url_lower:
        return 'discord'
    else:
        return 'website'


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Scrape raw data-props from a HuggingFace profile'
    )
    parser.add_argument(
        'profile',
        type=str,
        help='HuggingFace profile name to scrape (user or organization)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output JSON file path (if not specified, prints to console only)'
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

    # Scrape the profile
    result = scrape_hf_profile(args.profile)

    # Output result to console
    print(json.dumps(result, indent=2))

    # Save to file if output path specified
    if args.output:
        from pathlib import Path
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved output to {output_path}")
