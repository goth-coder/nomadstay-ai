import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

from constants import *

load_dotenv()

logger = logging.getLogger(__name__)

AWS_PROFILE = os.environ.get(
    "AWS_PROFILE"
)  # None → cadeia padrão do boto3 (ECS, EC2...)
AWS_REGION = os.environ.get("AWS_REGION")
BUCKET = os.environ.get("BUCKET", "<ARTIFACTS_BUCKET>")

session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)


def s3Key(filename: str, version: int) -> str:
    return f"rag/v{version}/{filename}"


def downloadFromS3(version: int = 1):
    """
    Downloads the FAISS index and the hotels doc from S3 to ARTIFACT_DIR.
    Raises ClientError (code '404') when they are not in S3.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = session.client("s3")
    logger.debug("download s3://%s/%s", BUCKET, s3Key("", version))
    s3.download_file(BUCKET, s3Key(INDEX_PATH.name, version), str(INDEX_PATH))
    s3.download_file(BUCKET, s3Key(DOCS_PATH.name, version), str(DOCS_PATH))


def uploadToS3(version: int = 1):
    """
    Uploads the local FAISS index and hotels doc to S3
    """
    s3 = session.client("s3")
    logger.debug("upload s3://%s/%s", BUCKET, s3Key("", version))
    s3.upload_file(str(INDEX_PATH), BUCKET, s3Key(INDEX_PATH.name, version))
    s3.upload_file(str(DOCS_PATH), BUCKET, s3Key(DOCS_PATH.name, version))


# ---------- v2 artifact sets: rag/v{version}/{city}/{name} ----------

def artifactKey(version: int, city: str, name: str) -> str:
    return f"rag/v{version}/{city}/{name}"


def uploadArtifacts(local_dir, version: int, city: str, names: list[str]) -> list[str]:
    """Uploads the named files of local_dir. Returns the S3 keys written."""
    local_dir = Path(local_dir)
    s3 = session.client("s3")
    keys = []
    for name in names:
        key = artifactKey(version, city, name)
        logger.debug("upload s3://%s/%s", BUCKET, key)
        s3.upload_file(str(local_dir / name), BUCKET, key)
        keys.append(key)
    return keys


def downloadArtifacts(version: int, city: str, local_dir, names: list[str]) -> list[Path]:
    """
    Downloads the named files into local_dir. Raises ClientError (code '404')
    when one is not in S3.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = session.client("s3")
    paths = []
    for name in names:
        logger.debug("download s3://%s/%s", BUCKET, artifactKey(version, city, name))
        s3.download_file(BUCKET, artifactKey(version, city, name), str(local_dir / name))
        paths.append(local_dir / name)
    return paths
