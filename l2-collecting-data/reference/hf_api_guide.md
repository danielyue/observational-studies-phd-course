# HuggingFace API & Scraping Documentation

This repository contains comprehensive documentation and code examples for working with HuggingFace APIs and scraping organization metadata.

## Contents

### 1. Models API Documentation
- **API Endpoint**: `https://huggingface.co/api/models`
- **Features**: Search, filter, and sort models by author, tags, libraries, etc.
- **Documentation**: See inline curl examples below

### 2. Organization Metadata Scraping
- **Scraper**: `hf_org_scraper.py` - Complete Python scraper for organization profiles
- **Guide**: `SCRAPING_GUIDE.md` - Detailed documentation on what can be scraped and how

## Quick Start

### Using Models API

```bash
# Get trending models
curl "https://huggingface.co/api/models?limit=10"

# Filter by author
curl "https://huggingface.co/api/models?author=meta-llama&limit=10"

# Filter by task type
curl "https://huggingface.co/api/models?pipeline_tag=text-generation&limit=10"

# Sort by downloads
curl "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=10"

# Get full model details
curl "https://huggingface.co/api/models?limit=1&full=true"
```

### Scraping Organization Metadata

```bash
# Run the scraper with uv (manages Python dependencies automatically)
uv run --with requests --with beautifulsoup4 python hf_org_scraper.py
```

The scraper will:
1. Extract basic organization info (name, email, plan, verification status)
2. Get follower counts and sample followers
3. Extract organization README/card content
4. Find social media links (Twitter, GitHub, website)
5. Count team members
6. Fetch all models, datasets, and spaces via API
7. Calculate aggregate statistics
8. Save results to a JSON file

## What Organization Metadata Can You Collect?

### From HTML Scraping (not available via API)
- Organization name and full name
- Contact email
- Subscription plan level (enterprise, team, pro)
- Verification status
- Follower count
- Sample followers with profiles
- Organization README/card content
- Social media links
- Team member count

### From API (combined with scraping)
- Complete list of models with stats
- Complete list of datasets with stats
- Complete list of spaces with stats
- Total likes across all content
- Total downloads

## Key Findings

### Models API

**Available Parameters:**
- `search` - Search by model name
- `author` - Filter by author/organization
- `filter` - Filter by task (e.g., "text-classification")
- `pipeline_tag` - Filter by pipeline tag
- `library` - Filter by library (pytorch, tensorflow, etc.)
- `language` - Filter by language
- `tags` - Filter by specific tags
- `sort` - Sort field (trending, downloads, likes, updated)
- `direction` - Sort direction (1 ascending, -1 descending)
- `limit` - Number of results (default 20, max 1000)
- `full` - Include full model details (true/false)

**Response Fields:**
- `_id` - Internal ID
- `id` - Model identifier (author/model-name)
- `likes` - Number of likes
- `downloads` - Download count
- `trendingScore` - Trending score
- `private` - Is private
- `tags` - Array of tags
- `pipeline_tag` - Primary task
- `library_name` - ML library
- `createdAt` - Creation timestamp
- `modelId` - Model ID

### Organizations

**No Dedicated API Exists**
- There is NO `/api/organizations` endpoint
- Must scrape HTML for organization metadata
- Can use `/api/models?author={org}` to get content by organization

**Rich Metadata in HTML:**
- Organization pages embed JSON data in `data-props` attributes
- Contains follower counts, plan info, team details, etc.
- Must parse HTML and decode HTML entities to extract

## Example Output

```json
{
  "organization": "meta-llama",
  "basic_info": {
    "name": "meta-llama",
    "fullname": "Meta Llama",
    "plan": "enterprise",
    "is_enterprise": true
  },
  "follower_info": {
    "follower_count": 64321,
    "sample_followers": [...]
  },
  "api_data": {
    "models": {
      "count": 70,
      "total_likes": 55664,
      "total_downloads": 27270020
    },
    "datasets": {
      "count": 11,
      "total_likes": 213
    }
  }
}
```

## Use Cases

1. **Organization Analytics** - Track growth, engagement, and content production
2. **Competitive Analysis** - Compare organizations by followers, models, downloads
3. **Discovery** - Find enterprise organizations, high-engagement models
4. **Monitoring** - Track new models, datasets, or changes over time
5. **Research** - Analyze trends in ML model development and adoption

## Technical Notes

### Scraping Considerations
- **HTML Entities**: JSON in `data-props` uses HTML entities - must decode
- **Multiple Elements**: Check all `data-props` elements on the page
- **Rate Limiting**: Be respectful with request frequency
- **Error Handling**: Some orgs have incomplete data - check field existence

### API Considerations
- **No Official Rate Limits**: Published, but be reasonable
- **Pagination**: Use `limit` parameter (max 1000 per request)
- **Caching**: Results change infrequently - cache when possible
- **Trending Sort**: Only available on web, not via API parameter

## Files

- `README.md` - This file
- `hf_org_scraper.py` - Complete organization scraper implementation
- `SCRAPING_GUIDE.md` - Detailed guide on scraping techniques
- `meta-llama_profile.json` - Example output file

## Requirements

```bash
# Using uv (recommended - manages dependencies automatically)
uv run --with requests --with beautifulsoup4 python hf_org_scraper.py

# Or install manually
pip install requests beautifulsoup4
python hf_org_scraper.py
```

## Complete API Examples

### Models API

```bash
# Basic search
curl "https://huggingface.co/api/models?search=bert&limit=5"

# Filter by author and pipeline
curl "https://huggingface.co/api/models?author=openai&pipeline_tag=text-generation"

# Sort by likes
curl "https://huggingface.co/api/models?sort=likes&direction=-1&limit=10"

# Multiple filters
curl "https://huggingface.co/api/models?library=pytorch&pipeline_tag=text-generation&language=en&limit=20"

# Get full details
curl "https://huggingface.co/api/models?author=meta-llama&full=true&limit=1" | jq '.'
```

### Datasets API

```bash
# List datasets
curl "https://huggingface.co/api/datasets?limit=10"

# Filter by author
curl "https://huggingface.co/api/datasets?author=huggingface&limit=20"

# Search datasets
curl "https://huggingface.co/api/datasets?search=squad&limit=5"
```

### Spaces API

```bash
# List spaces
curl "https://huggingface.co/api/spaces?limit=10"

# Filter by author
curl "https://huggingface.co/api/spaces?author=huggingface&limit=20"
```

## Python Examples

### Get All Models for an Organization

```python
import requests

def get_all_org_models(org_name):
    url = "https://huggingface.co/api/models"
    models = []

    response = requests.get(url, params={
        'author': org_name,
        'limit': 1000
    })

    if response.status_code == 200:
        models = response.json()

    return models

# Usage
models = get_all_org_models('meta-llama')
print(f"Found {len(models)} models")
print(f"Total downloads: {sum(m.get('downloads', 0) for m in models)}")
```

### Get Top Models by Downloads

```python
import requests

def get_top_models(limit=10):
    url = "https://huggingface.co/api/models"

    response = requests.get(url, params={
        'sort': 'downloads',
        'direction': -1,
        'limit': limit
    })

    if response.status_code == 200:
        return response.json()
    return []

# Usage
top_models = get_top_models(10)
for model in top_models:
    print(f"{model['id']}: {model['downloads']:,} downloads")
```

### Scrape Organization Profile

```python
from hf_org_scraper import get_full_organization_profile

# Get complete profile
profile = get_full_organization_profile('meta-llama')

print(f"Organization: {profile['basic_info']['fullname']}")
print(f"Followers: {profile['follower_info']['follower_count']:,}")
print(f"Models: {profile['api_data']['models']['count']}")
print(f"Total Downloads: {profile['api_data']['models']['total_downloads']:,}")
```

## Summary

This repository provides everything you need to:
1. ✅ Access HuggingFace Models, Datasets, and Spaces APIs
2. ✅ Scrape organization metadata from HTML pages
3. ✅ Combine API and scraped data for complete profiles
4. ✅ Analyze and compare organizations
5. ✅ Track growth and engagement over time

All code is provided with working examples and comprehensive documentation.
