"""
Download HuggingFace Hub Stats Data
Downloads the models.parquet file from the cfahlgren1/hub-stats dataset.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm
import shutil
import hashlib
import requests


def _compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _fetch_all_commits_with_pagination(repo_id, repo_type="dataset", days_back=None):
    """
    Fetch all commits from a HuggingFace repository using git commands.

    The HfApi.list_repo_commits() only returns the first 50 commits,
    so we use git log to get the full history.

    Args:
        repo_id: Repository ID (e.g., "cfahlgren1/hub-stats")
        repo_type: Type of repository ("dataset", "model", or "space")
        days_back: If specified, only fetch commits from the past N days

    Returns:
        List of commit dictionaries with keys: commit_id, created_at, title, message
    """
    import subprocess
    import tempfile
    from collections import namedtuple

    # Set cutoff date if days_back is specified
    cutoff_date = None
    if days_back:
        cutoff_date = datetime.now(tz=datetime.now().astimezone().tzinfo) - timedelta(days=days_back)

    # Construct the git URL
    if repo_type == "dataset":
        git_url = f"https://huggingface.co/datasets/{repo_id}"
    elif repo_type == "model":
        git_url = f"https://huggingface.co/{repo_id}"
    else:
        git_url = f"https://huggingface.co/spaces/{repo_id}"

    # Create a temporary directory for the shallow clone
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            logging.info(f"Fetching commit history via git (this may take a moment)...")

            # Do a shallow clone with no files, just git history
            # Use --filter=blob:none to avoid downloading file contents
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", git_url, tmpdir],
                check=True,
                capture_output=True,
                text=True
            )

            # Get commit log with specific format
            # Format: commit_hash|author_date|subject|body
            git_log_cmd = [
                "git", "-C", tmpdir, "log",
                "--pretty=format:%H|%aI|%s|%b",
                "--date=iso-strict"
            ]

            # Add date filter if specified
            if cutoff_date:
                since_date = cutoff_date.strftime("%Y-%m-%d")
                git_log_cmd.append(f"--since={since_date}")

            result = subprocess.run(
                git_log_cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Parse the commits
            CommitInfo = namedtuple('CommitInfo', ['commit_id', 'created_at', 'title', 'message'])
            all_commits = []

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('|', 3)
                if len(parts) >= 3:
                    commit_hash = parts[0]
                    commit_date = datetime.fromisoformat(parts[1])
                    title = parts[2]
                    message = parts[3] if len(parts) > 3 else ""

                    commit_info = CommitInfo(
                        commit_id=commit_hash,
                        created_at=commit_date,
                        title=title,
                        message=message
                    )
                    all_commits.append(commit_info)

            logging.info(f"Fetched {len(all_commits)} commits via git")
            return all_commits

        except subprocess.CalledProcessError as e:
            logging.error(f"Git command failed: {e.stderr}")
            logging.warning("Falling back to API (limited to 50 commits)")

            # Fallback to API
            api = HfApi()
            commits = api.list_repo_commits(repo_id=repo_id, repo_type=repo_type)

            if cutoff_date:
                commits = [c for c in commits if c.created_at >= cutoff_date]

            return commits


def download_models_data(output_dir):
    """
    Download models.parquet from HuggingFace hub-stats dataset.

    Args:
        output_dir: Directory to save the downloaded file (raw data directory)

    Returns:
        Path to the downloaded file
    """
    logging.info("Starting download of models.parquet from HuggingFace")

    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download the file from HuggingFace
        repo_id = "cfahlgren1/hub-stats"
        filename = "models.parquet"

        logging.info(f"Downloading {filename} from {repo_id}...")

        # Download file to cache and get path
        cached_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )

        # Copy to our raw data directory
        output_path = output_dir / filename

        # Read and write to move from cache to our directory
        shutil.copy(cached_file_path, output_path)

        logging.info(f"Successfully downloaded {filename} to {output_path}")
        logging.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

        return output_path

    except Exception as e:
        logging.error(f"Failed to download models data: {str(e)}")
        raise


def download_historical_models_data(output_dir, days_back=None):
    """
    Download all historical versions of model data from HuggingFace hub-stats dataset.

    Iterates through commits that mention models.parquet or models.csv in the commit
    message and downloads the corresponding file. The repository switched from CSV to
    Parquet format at some point, so this function handles both formats.

    Files are saved with naming format: models-{YYYYMMDD}-{7char_commit_sha}.{csv|parquet}

    Only downloads unique versions (deduplicates by file hash).

    Args:
        output_dir: Base directory for raw data. Historical files will be saved to
                   output_dir/historical/
        days_back: Number of days to look back in commit history. If None, fetches all commits.

    Returns:
        List of paths to downloaded files
    """
    logging.info("Starting download of historical models.parquet versions")
    if days_back:
        logging.info(f"Filtering commits from the past {days_back} days")

    try:
        # Setup directories
        output_dir = Path(output_dir)
        historical_dir = output_dir / "historical"
        historical_dir.mkdir(parents=True, exist_ok=True)

        # Initialize HuggingFace API
        api = HfApi()
        repo_id = "cfahlgren1/hub-stats"
        filename = "models.parquet"

        logging.info(f"Fetching commit history for {repo_id}...")

        # Fetch all commits using git (supports full history with pagination)
        commits = _fetch_all_commits_with_pagination(
            repo_id=repo_id,
            repo_type="dataset",
            days_back=days_back
        )

        logging.info(f"Found {len(commits)} commits" + (f" from the past {days_back} days" if days_back else ""))
        date_filtered_commits = commits

        # Filter commits where models.parquet or models.csv changed (by commit message)
        logging.info("Filtering commits by commit message...")
        relevant_commits = []
        for commit in date_filtered_commits:
            # Check both title and message body
            commit_text = f"{commit.title} {commit.message}".lower()
            if "models.parquet" in commit_text or "models.csv" in commit_text:
                relevant_commits.append(commit)

        logging.info(f"Found {len(relevant_commits)} commits with model file changes")

        downloaded_files = []
        successful_downloads = 0
        skipped_existing = 0
        skipped_no_change = 0
        skipped_no_file = 0

        # Track seen file hashes to detect duplicates
        seen_hashes = set()

        # Download each version with progress bar and time estimate
        for commit in tqdm(relevant_commits, desc="Downloading historical versions", unit="commit"):
            try:
                commit_sha = commit.commit_id
                commit_date = commit.created_at

                # Format date as YYYYMMDD
                date_str = commit_date.strftime("%Y%m%d")
                short_sha = commit_sha[:7]

                # Try to determine which file format exists (CSV or Parquet)
                # The repo switched from CSV to Parquet at some point
                file_downloaded = False

                # Try parquet first (more recent format)
                for file_format in ["parquet", "csv"]:
                    try:
                        source_filename = f"models.{file_format}"
                        output_filename = f"models-{date_str}-{short_sha}.{file_format}"
                        output_path = historical_dir / output_filename

                        # Skip if already downloaded
                        if output_path.exists():
                            # Still need to track its hash to avoid duplicates
                            file_hash = _compute_file_hash(output_path)
                            seen_hashes.add(file_hash)
                            logging.debug(f"Skipping {output_filename} (already exists)")
                            skipped_existing += 1
                            file_downloaded = True
                            break

                        # Download the file at this specific commit
                        cached_file_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=source_filename,
                            repo_type="dataset",
                            revision=commit_sha
                        )

                        # Compute hash to check if file changed
                        file_hash = _compute_file_hash(cached_file_path)

                        # Skip if we've already seen this exact file content
                        if file_hash in seen_hashes:
                            logging.debug(f"Skipping commit {short_sha} (no file changes)")
                            skipped_no_change += 1
                            file_downloaded = True
                            break

                        # This is a new version - save it
                        seen_hashes.add(file_hash)
                        shutil.copy(cached_file_path, output_path)

                        downloaded_files.append(output_path)
                        successful_downloads += 1

                        logging.debug(f"Downloaded {output_filename}")
                        file_downloaded = True
                        break

                    except Exception as e:
                        # Try the other format
                        error_msg = str(e)
                        if "404" in error_msg or "not found" in error_msg.lower():
                            continue  # Try next format
                        else:
                            logging.debug(f"Error with {source_filename}: {error_msg}")
                            continue

                if not file_downloaded:
                    logging.debug(f"Skipping commit {short_sha} (neither models.csv nor models.parquet found)")
                    skipped_no_file += 1

            except Exception as e:
                # Unexpected error
                logging.debug(f"Skipping commit {commit.commit_id[:7]}: {str(e)}")
                skipped_no_file += 1
                continue

        logging.info(f"Successfully downloaded {successful_downloads} new historical versions")
        logging.info(f"Skipped {skipped_existing} existing files")
        logging.info(f"Skipped {skipped_no_change} commits (no file changes)")
        logging.info(f"Skipped {skipped_no_file} commits (file doesn't exist)")
        logging.info(f"Historical files saved to {historical_dir}")

        # Calculate total size
        if downloaded_files:
            total_size = sum(f.stat().st_size for f in downloaded_files)
            logging.info(f"Total download size: {total_size / (1024*1024):.2f} MB")
        else:
            logging.info("No new files downloaded")

        return downloaded_files

    except Exception as e:
        logging.error(f"Failed to download historical models data: {str(e)}")
        raise
