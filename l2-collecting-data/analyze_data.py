"""
Analyze HuggingFace Data

This script loads collected model and organization data and reports summary statistics.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> List[Dict[Any, Any]]:
    """Load data from a JSONL file"""
    data = []
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return data

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse line: {e}")
                continue

    return data


def analyze_models(models: List[dict]) -> Dict[str, Any]:
    """Analyze model data and return summary statistics"""
    if not models:
        return {'error': 'No models to analyze'}

    stats = {
        'total_models': len(models),
        'total_downloads': sum(m.get('downloads', 0) for m in models),
        'total_likes': sum(m.get('likes', 0) for m in models),
        'avg_downloads': 0,
        'avg_likes': 0,
        'top_models_by_downloads': [],
        'top_models_by_likes': [],
        'models_by_pipeline_tag': {},
        'models_by_library': {},
        'models_by_author': {},
    }

    # Calculate averages
    stats['avg_downloads'] = stats['total_downloads'] / len(models)
    stats['avg_likes'] = stats['total_likes'] / len(models)

    # Top models by downloads
    sorted_by_downloads = sorted(models, key=lambda x: x.get('downloads', 0), reverse=True)
    stats['top_models_by_downloads'] = [
        {
            'id': m.get('id'),
            'downloads': m.get('downloads', 0),
            'likes': m.get('likes', 0)
        }
        for m in sorted_by_downloads[:10]
    ]

    # Top models by likes
    sorted_by_likes = sorted(models, key=lambda x: x.get('likes', 0), reverse=True)
    stats['top_models_by_likes'] = [
        {
            'id': m.get('id'),
            'likes': m.get('likes', 0),
            'downloads': m.get('downloads', 0)
        }
        for m in sorted_by_likes[:10]
    ]

    # Group by pipeline tag
    pipeline_counter = Counter()
    for m in models:
        tag = m.get('pipeline_tag', 'unknown')
        pipeline_counter[tag] += 1

    stats['models_by_pipeline_tag'] = dict(pipeline_counter.most_common(20))

    # Group by library
    library_counter = Counter()
    for m in models:
        lib = m.get('library_name', 'unknown')
        library_counter[lib] += 1

    stats['models_by_library'] = dict(library_counter.most_common(20))

    # Group by author
    author_counter = Counter()
    for m in models:
        # Extract author from model ID (format: author/model-name)
        model_id = m.get('id', '')
        if '/' in model_id:
            author = model_id.split('/')[0]
            author_counter[author] += 1

    stats['models_by_author'] = dict(author_counter.most_common(20))

    return stats


def analyze_organizations(organizations: List[dict]) -> Dict[str, Any]:
    """Analyze organization data and return summary statistics"""
    if not organizations:
        return {'error': 'No organizations to analyze'}

    # Filter out error records
    valid_orgs = [o for o in organizations if not o.get('error')]
    error_orgs = [o for o in organizations if o.get('error')]

    stats = {
        'total_organizations': len(organizations),
        'valid_organizations': len(valid_orgs),
        'failed_organizations': len(error_orgs),
        'total_models': 0,
        'total_datasets': 0,
        'total_spaces': 0,
        'total_followers': 0,
        'top_orgs_by_models': [],
        'top_orgs_by_downloads': [],
        'top_orgs_by_followers': [],
        'orgs_by_plan': {},
    }

    if not valid_orgs:
        return stats

    # Aggregate statistics
    for org in valid_orgs:
        # Models
        models_data = org.get('api_data', {}).get('models', {})
        stats['total_models'] += models_data.get('count', 0)

        # Datasets
        datasets_data = org.get('api_data', {}).get('datasets', {})
        stats['total_datasets'] += datasets_data.get('count', 0)

        # Spaces
        spaces_data = org.get('api_data', {}).get('spaces', {})
        stats['total_spaces'] += spaces_data.get('count', 0)

        # Followers
        follower_data = org.get('follower_info', {})
        stats['total_followers'] += follower_data.get('follower_count', 0)

    # Top orgs by model count
    orgs_with_models = [
        {
            'name': o.get('organization'),
            'model_count': o.get('api_data', {}).get('models', {}).get('count', 0),
            'total_downloads': o.get('api_data', {}).get('models', {}).get('total_downloads', 0)
        }
        for o in valid_orgs
    ]
    stats['top_orgs_by_models'] = sorted(
        orgs_with_models,
        key=lambda x: x['model_count'],
        reverse=True
    )[:10]

    # Top orgs by downloads
    stats['top_orgs_by_downloads'] = sorted(
        orgs_with_models,
        key=lambda x: x['total_downloads'],
        reverse=True
    )[:10]

    # Top orgs by followers
    orgs_with_followers = [
        {
            'name': o.get('organization'),
            'followers': o.get('follower_info', {}).get('follower_count', 0)
        }
        for o in valid_orgs
    ]
    stats['top_orgs_by_followers'] = sorted(
        orgs_with_followers,
        key=lambda x: x['followers'],
        reverse=True
    )[:10]

    # Group by plan
    plan_counter = Counter()
    for o in valid_orgs:
        plan = o.get('basic_info', {}).get('plan', 'unknown')
        plan_counter[plan] += 1

    stats['orgs_by_plan'] = dict(plan_counter)

    return stats


def print_model_stats(stats: Dict[str, Any]):
    """Pretty print model statistics"""
    if 'error' in stats:
        logger.info(f"\n❌ {stats['error']}")
        return

    logger.info("\n" + "="*80)
    logger.info("MODEL STATISTICS")
    logger.info("="*80)

    logger.info(f"\n📊 Overview:")
    logger.info(f"  Total Models: {stats['total_models']:,}")
    logger.info(f"  Total Downloads: {stats['total_downloads']:,}")
    logger.info(f"  Total Likes: {stats['total_likes']:,}")
    logger.info(f"  Avg Downloads per Model: {stats['avg_downloads']:,.0f}")
    logger.info(f"  Avg Likes per Model: {stats['avg_likes']:,.1f}")

    logger.info(f"\n🏆 Top 10 Models by Downloads:")
    for i, model in enumerate(stats['top_models_by_downloads'], 1):
        logger.info(f"  {i:2}. {model['id']:50} {model['downloads']:12,} downloads")

    logger.info(f"\n❤️  Top 10 Models by Likes:")
    for i, model in enumerate(stats['top_models_by_likes'], 1):
        logger.info(f"  {i:2}. {model['id']:50} {model['likes']:8,} likes")

    logger.info(f"\n🏷️  Top Pipeline Tags:")
    for tag, count in list(stats['models_by_pipeline_tag'].items())[:10]:
        logger.info(f"  {tag:30} {count:5,} models")

    logger.info(f"\n📚 Top Libraries:")
    for lib, count in list(stats['models_by_library'].items())[:10]:
        logger.info(f"  {lib:30} {count:5,} models")

    logger.info(f"\n👥 Top Authors:")
    for author, count in list(stats['models_by_author'].items())[:10]:
        logger.info(f"  {author:30} {count:5,} models")


def print_org_stats(stats: Dict[str, Any]):
    """Pretty print organization statistics"""
    if 'error' in stats:
        logger.info(f"\n❌ {stats['error']}")
        return

    logger.info("\n" + "="*80)
    logger.info("ORGANIZATION STATISTICS")
    logger.info("="*80)

    logger.info(f"\n📊 Overview:")
    logger.info(f"  Total Organizations: {stats['total_organizations']:,}")
    logger.info(f"  Valid Organizations: {stats['valid_organizations']:,}")
    logger.info(f"  Failed Organizations: {stats['failed_organizations']:,}")
    logger.info(f"  Total Models: {stats['total_models']:,}")
    logger.info(f"  Total Datasets: {stats['total_datasets']:,}")
    logger.info(f"  Total Spaces: {stats['total_spaces']:,}")
    logger.info(f"  Total Followers: {stats['total_followers']:,}")

    if stats['top_orgs_by_models']:
        logger.info(f"\n🏆 Top 10 Organizations by Model Count:")
        for i, org in enumerate(stats['top_orgs_by_models'], 1):
            logger.info(f"  {i:2}. {org['name']:30} {org['model_count']:5,} models")

    if stats['top_orgs_by_downloads']:
        logger.info(f"\n⬇️  Top 10 Organizations by Downloads:")
        for i, org in enumerate(stats['top_orgs_by_downloads'], 1):
            logger.info(f"  {i:2}. {org['name']:30} {org['total_downloads']:15,} downloads")

    if stats['top_orgs_by_followers']:
        logger.info(f"\n👥 Top 10 Organizations by Followers:")
        for i, org in enumerate(stats['top_orgs_by_followers'], 1):
            logger.info(f"  {i:2}. {org['name']:30} {org['followers']:10,} followers")

    if stats['orgs_by_plan']:
        logger.info(f"\n💼 Organizations by Plan:")
        for plan, count in stats['orgs_by_plan'].items():
            logger.info(f"  {plan:30} {count:5,} orgs")


def analyze_data(
    models_file: Path = None,
    organizations_file: Path = None
):
    """
    Analyze collected data and print summary statistics

    Args:
        models_file: Path to models JSONL file
        organizations_file: Path to organizations JSONL file
    """
    data_dir = Path(__file__).parent / "data"

    if models_file is None:
        models_file = data_dir / "models.jsonl"
    if organizations_file is None:
        organizations_file = data_dir / "organizations.jsonl"

    # Load models
    logger.info("Loading models data...")
    models = load_jsonl(models_file)
    logger.info(f"Loaded {len(models)} models")

    # Load organizations
    logger.info("Loading organizations data...")
    organizations = load_jsonl(organizations_file)
    logger.info(f"Loaded {len(organizations)} organizations")

    # Analyze models
    if models:
        model_stats = analyze_models(models)
        print_model_stats(model_stats)
    else:
        logger.info("\n⚠️  No model data found")

    # Analyze organizations
    if organizations:
        org_stats = analyze_organizations(organizations)
        print_org_stats(org_stats)
    else:
        logger.info("\n⚠️  No organization data found")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyze_data()
