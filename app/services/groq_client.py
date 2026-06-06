import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def get_groq_client() -> Any | None:
    """Create a Groq client without inheriting ambient proxy settings.

    Some environments expose proxy variables or ship an httpx/Groq
    combination that raises `Client.__init__() got an unexpected keyword
    argument 'proxies'`. Using a dedicated client with `trust_env=False`
    avoids that constructor path while keeping the Groq-backed flow when the
    SDK is available.
    """

    api_key = os.getenv('GROQ_API')
    if not api_key:
        return None

    try:
        from groq import Groq

        return Groq(api_key=api_key, http_client=httpx.Client(trust_env=False))
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning('Unable to create Groq client: %s', exc)
        return None
