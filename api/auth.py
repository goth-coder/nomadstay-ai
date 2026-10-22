import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

# auto_error=False: a missing header returns our 401, not FastAPI's 403
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def requireApiKey(api_key: str | None = Security(api_key_header)):
    """
    Rejects requests without a valid X-API-Key header.
    compare_digest takes the same time whether the first or the last
    character differs, so the key can't be guessed by timing the responses.
    """
    expected = os.environ.get("API_KEY", "")
    if not api_key or not expected or not secrets.compare_digest(
        api_key.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
