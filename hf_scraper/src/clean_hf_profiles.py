#!/usr/bin/env python3
"""
Script to parse HuggingFace profile data and export to CSV files.

This script processes HuggingFace profile JSON data and creates three CSV files:
1. hf_profiles.csv - Master list with basic profile info
2. hf_orgs.csv - Detailed organization data
3. hf_users.csv - Detailed user data
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def safe_get(data: Dict[str, Any], *keys, default=None) -> Any:
    """Safely navigate nested dictionary keys."""
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return default
        if result is None:
            return default
    return result


def determine_profile_type(profile_data: Dict[str, Any]) -> str:
    """
    Determine if a profile is an organization, user, or unknown.

    Returns:
        - "org" if data.org field is present
        - "user" if data.u field is present
        - "unknown" if neither present
        - "assumptions broken" if both present
    """
    has_org = safe_get(profile_data, "data", "org") is not None
    has_user = safe_get(profile_data, "data", "u") is not None

    if has_org and has_user:
        return "assumptions broken"
    elif has_org:
        return "org"
    elif has_user:
        return "user"
    else:
        return "unknown"


def determine_org_type(tag_texts: list) -> str:
    """
    Determine organization type based on tags.

    Returns: University, Non-Profit, Company, Community, Government, Classroom, or empty string
    """
    # Convert tags to lowercase for case-insensitive matching
    tags_lower = [tag.lower() for tag in tag_texts]

    # Check in priority order
    if "university" in tags_lower:
        return "University"
    elif "non-profit" in tags_lower:
        return "Non-Profit"
    elif "company" in tags_lower:
        return "Company"
    elif "community" in tags_lower:
        return "Community"
    elif "government" in tags_lower:
        return "Government"
    elif "classroom" in tags_lower:
        return "Classroom"
    else:
        return ""


def parse_org_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse organization profile data."""
    data = profile_data.get("data", {})
    org = data.get("org", {})
    header_metadata = profile_data.get("header_metadata", {})

    # Extract tags
    tags = header_metadata.get("tags", [])
    tag_texts = [tag.get("text", "") for tag in tags if isinstance(tag, dict)]
    tags_str = ", ".join(tag_texts) if tag_texts else ""

    # Determine org type and verification status
    org_type = determine_org_type(tag_texts)
    is_verified = "Verified" in tag_texts

    # Extract links
    links = header_metadata.get("links", [])
    link_urls = [link.get("url", "") for link in links if isinstance(link, dict)]
    links_str = ", ".join(link_urls) if link_urls else ""

    # Check if organization card exists and is not empty
    org_card = data.get("organizationCard")
    has_org_card = org_card is not None and (
        (isinstance(org_card, dict) and len(org_card) > 0) or
        (isinstance(org_card, str) and len(org_card) > 0)
    )

    return {
        "profile_name": profile_data.get("profile", ""),
        "Type": org_type,
        "isVerified": is_verified,
        "type": org.get("type", ""),
        "fullname": org.get("fullname", ""),
        "name": org.get("name", ""),
        "isHf": org.get("isHf", ""),
        "details": org.get("details", ""),
        "isEnterprise": org.get("isEnterprise", ""),
        "plan": org.get("plan", ""),
        "has_org_card": has_org_card,
        "followerCount": data.get("followerCount", ""),
        "userCount": data.get("userCount", ""),
        "numDatasets": data.get("numDatasets", ""),
        "numModels": data.get("numModels", ""),
        "numSpaces": data.get("numSpaces", ""),
        "numPapers": data.get("numPapers", ""),
        "orgEmailDomain": data.get("orgEmailDomain", ""),
        "org_display_name": header_metadata.get("org_display_name", ""),
        "tags": tags_str,
        "links": links_str,
    }


def parse_user_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse user profile data."""
    data = profile_data.get("data", {})
    user = data.get("u", {})
    signup = user.get("signup", {})

    # Calculate number of orgs
    orgs = user.get("orgs", [])
    num_orgs = len(orgs) if isinstance(orgs, list) else ""

    # Check if hardwareItems exists
    has_hardware_items = data.get("hardwareItems") is not None

    return {
        "profile_name": profile_data.get("profile", ""),
        "type": user.get("type", ""),
        "isPro": user.get("isPro", ""),
        "isHf": user.get("isHf", ""),
        "isMod": user.get("isMod", ""),
        "fullname": user.get("fullname", ""),
        "profile_details": signup.get("details", ""),
        "homepage": signup.get("homepage", ""),
        "github": signup.get("github", ""),
        "bluesky": signup.get("bluesky", ""),
        "linkedin": signup.get("linkedin", ""),
        "twitter": signup.get("twitter", ""),
        "numOrgs": num_orgs,
        "totalBlogPosts": data.get("totalBlogPosts", ""),
        "communityScore": data.get("communityScore", ""),
        "numberLikes": data.get("numberLikes", ""),
        "totalPosts": data.get("totalPosts", ""),
        "upvotes": data.get("upvotes", ""),
        "numFollowers": data.get("numFollowers", ""),
        "numFollowingUsers": data.get("numFollowingUsers", ""),
        "numFollowingOrgs": data.get("numFollowingOrgs", ""),
        "numModels": data.get("numModels", ""),
        "numDatasets": data.get("numDatasets", ""),
        "numSpaces": data.get("numSpaces", ""),
        "has_hardware_items": has_hardware_items,
    }


def extract_org_memberships(profile_data: Dict[str, Any]) -> list:
    """
    Extract organization memberships from a user profile.

    Returns: List of dicts with name_user, name_org, userRole
    """
    memberships = []
    profile_name = profile_data.get("profile", "")
    data = profile_data.get("data", {})
    user = data.get("u", {})
    orgs = user.get("orgs", [])

    if isinstance(orgs, list):
        for org in orgs:
            if isinstance(org, dict):
                membership = {
                    "name_user": profile_name,
                    "name_org": org.get("name", ""),
                    "userRole": org.get("userRole", "")
                }
                memberships.append(membership)

    return memberships


def append_to_csv(file_path: Path, data: Dict[str, Any], fieldnames: list):
    """Append a row to a CSV file, creating it with headers if it doesn't exist."""
    file_exists = file_path.exists()

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def parse_hf_profile(profile_data: Dict[str, Any], output_dir: Path = Path(".")):
    """
    Parse a HuggingFace profile and append data to the appropriate CSV files.

    Args:
        profile_data: Dictionary containing the profile data
        output_dir: Directory where CSV files will be created/updated
    """
    # Determine profile type
    profile_name = profile_data.get("profile", "unknown")
    hf_type = determine_profile_type(profile_data)

    # Write to master profiles CSV
    master_csv = output_dir / "hf_profiles.csv"
    master_data = {
        "profile_name": profile_name,
        "hf_type": hf_type
    }
    append_to_csv(master_csv, master_data, ["profile_name", "hf_type"])

    # Write to appropriate detailed CSV based on type
    if hf_type == "org":
        org_data = parse_org_profile(profile_data)
        org_csv = output_dir / "hf_orgs.csv"
        org_fieldnames = [
            "profile_name", "Type", "isVerified", "type", "fullname", "name", "isHf", "details",
            "isEnterprise", "plan", "has_org_card", "followerCount", "userCount",
            "numDatasets", "numModels", "numSpaces", "numPapers", "orgEmailDomain",
            "org_display_name", "tags", "links"
        ]
        append_to_csv(org_csv, org_data, org_fieldnames)
        print(f"✓ Processed organization: {profile_name}")

    elif hf_type == "user":
        user_data = parse_user_profile(profile_data)
        user_csv = output_dir / "hf_users.csv"
        user_fieldnames = [
            "profile_name", "type", "isPro", "isHf", "isMod", "fullname",
            "profile_details", "homepage", "github", "bluesky", "linkedin",
            "twitter", "numOrgs", "totalBlogPosts", "communityScore", "numberLikes",
            "totalPosts", "upvotes", "numFollowers", "numFollowingUsers",
            "numFollowingOrgs", "numModels", "numDatasets", "numSpaces",
            "has_hardware_items"
        ]
        append_to_csv(user_csv, user_data, user_fieldnames)

        # Extract and write org memberships
        memberships = extract_org_memberships(profile_data)
        if memberships:
            members_csv = output_dir / "hf_org_members.csv"
            members_fieldnames = ["name_user", "name_org", "userRole"]
            for membership in memberships:
                append_to_csv(members_csv, membership, members_fieldnames)

        print(f"✓ Processed user: {profile_name}")

    else:
        print(f"⚠ Skipped {profile_name}: type={hf_type}")


def process_file(input_path: Path, output_dir: Path, verbose: bool = True):
    """
    Process a JSON or JSONL file containing HuggingFace profiles.

    Args:
        input_path: Path to the input JSON or JSONL file
        output_dir: Directory where CSV files will be created
        verbose: Whether to print progress messages (default: True)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        # Try to load the entire file as JSON first
        try:
            f.seek(0)
            data = json.load(f)

            # Handle both single objects and arrays
            if isinstance(data, dict):
                # Single JSON object
                parse_hf_profile(data, output_dir)
            elif isinstance(data, list):
                # JSON array
                for profile in data:
                    parse_hf_profile(profile, output_dir)
            else:
                if verbose:
                    print(f"✗ Unexpected JSON format in {input_path}", file=sys.stderr)

        except json.JSONDecodeError:
            # If that fails, try JSONL (one JSON object per line)
            f.seek(0)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    profile = json.loads(line)
                    parse_hf_profile(profile, output_dir)
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"✗ Error parsing line {line_num}: {e}", file=sys.stderr)


def clean_hf_profiles(input_files: list, output_dir: Path):
    """
    Clean HuggingFace profile data from JSON/JSONL files and export to CSV.

    Args:
        input_files: List of paths to JSON or JSONL files
        output_dir: Directory where CSV files will be created

    Returns:
        List of created CSV file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each input file
    for input_path in input_files:
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"✗ Warning: File not found: {input_path}", file=sys.stderr)
            continue

        try:
            process_file(input_path, output_dir, verbose=True)
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}", file=sys.stderr)

    # Return list of created files
    created_files = [
        output_dir / "hf_profiles.csv",
        output_dir / "hf_orgs.csv",
        output_dir / "hf_users.csv",
        output_dir / "hf_org_members.csv"
    ]

    return [f for f in created_files if f.exists()]


def main():
    parser = argparse.ArgumentParser(
        description="Parse HuggingFace profile data and export to CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single JSON file
  python clean_hf_profiles.py data/allenai.json

  # Process a JSONL file with custom output directory
  python clean_hf_profiles.py data/profiles.jsonl --output-dir output/

  # Process multiple files
  python clean_hf_profiles.py data/*.json
        """
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Path(s) to JSON or JSONL file(s) containing profile data"
    )

    # Default output directory relative to script location
    default_output = Path(__file__).parent.parent / "data" / "processed"

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory where CSV files will be created (default: hf_scraper/data/processed)"
    )

    args = parser.parse_args()

    # Process each input file
    total_files = len(args.input_files)
    for i, input_path in enumerate(args.input_files, 1):
        if not input_path.exists():
            print(f"✗ Error: File not found: {input_path}", file=sys.stderr)
            continue

        print(f"\n[{i}/{total_files}] Processing {input_path}...")
        try:
            process_file(input_path, args.output_dir)
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}", file=sys.stderr)

    print(f"\n✓ Done! CSV files created in: {args.output_dir.absolute()}")
    print(f"  - hf_profiles.csv (master list)")
    print(f"  - hf_orgs.csv (organization details)")
    print(f"  - hf_users.csv (user details)")
    print(f"  - hf_org_members.csv (user-org memberships)")


if __name__ == "__main__":
    main()
