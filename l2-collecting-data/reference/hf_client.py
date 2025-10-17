"""
HuggingFace API Client with Rate Limiting

This module provides a rate-limited HTTP client for interacting with the HuggingFace API.
"""

import os
from typing import Optional
from requests_ratelimiter import LimiterSession
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)


class HFClient:
    """Rate-limited HTTP client for HuggingFace API"""

    def __init__(self, requests_per_second: int = 5, api_key: Optional[str] = None):
        """
        Initialize the HuggingFace client with rate limiting.

        Args:
            requests_per_second: Maximum number of requests per second (default: 5)
            api_key: HuggingFace API key. If None, will try to load from environment.
        """
        self.requests_per_second = requests_per_second
        self.api_key = api_key or self._load_api_key()
        self.session = self._create_session()

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or .env file"""
        # Try to load from environment
        api_key = os.getenv('HF_API_KEY')

        if not api_key:
            # Try to load from .env file
            try:
                from pathlib import Path
                env_file = Path(__file__).parent.parent / '.env'
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('HF_API_KEY='):
                                api_key = line.split('=', 1)[1].strip()
                                break
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")

        if not api_key:
            logger.warning("No HF_API_KEY found. Some API calls may be rate-limited more aggressively.")

        return api_key

    def _create_session(self) -> LimiterSession:
        """Create a rate-limited session with retry logic"""
        # Create rate-limited session
        session = LimiterSession(per_second=self.requests_per_second)

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers
        if self.api_key:
            session.headers.update({'Authorization': f'Bearer {self.api_key}'})

        session.headers.update({
            'User-Agent': 'HF-Data-Collection-Script/1.0'
        })

        return session

    def get(self, url: str, **kwargs):
        """
        Make a GET request with rate limiting

        Args:
            url: URL to request
            **kwargs: Additional arguments to pass to requests.get()

        Returns:
            Response object
        """
        logger.debug(f"GET {url}")
        return self.session.get(url, **kwargs)

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_client(requests_per_second: int = 5, api_key: Optional[str] = None) -> HFClient:
    """
    Factory function to create an HFClient instance

    Args:
        requests_per_second: Maximum number of requests per second (default: 5)
        api_key: HuggingFace API key. If None, will try to load from environment.

    Returns:
        HFClient instance
    """
    return HFClient(requests_per_second=requests_per_second, api_key=api_key)
