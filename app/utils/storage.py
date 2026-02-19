import hashlib
from typing import Optional
from google.cloud import storage as gcs_storage
from app.config import get_settings

def get_storage_client():
    return gcs_storage.Client(project=get_settings().GCP_PROJECT_ID)

def get_bucket():
    client = get_storage_client()
    return client.bucket(get_settings().GCS_BUCKET_NAME)

def upload_file(path: str, file_bytes: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload file to GCS bucket. Returns the GCS URI."""
    bucket = get_bucket()
    blob = bucket.blob(path)
    blob.upload_from_string(file_bytes, content_type=content_type)
    return f"gs://{bucket.name}/{path}"

def download_file(path: str) -> bytes:
    """Download file from GCS bucket."""
    bucket = get_bucket()
    blob = bucket.blob(path)
    return blob.download_as_bytes()

def generate_signed_url(path: str, expiry_minutes: int = 60) -> str:
    """Generate a signed URL for temporary access to a GCS object."""
    import datetime
    bucket = get_bucket()
    blob = bucket.blob(path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiry_minutes),
        method="GET",
    )
    return url

def delete_file(path: str) -> None:
    """Delete a file from GCS bucket."""
    bucket = get_bucket()
    blob = bucket.blob(path)
    blob.delete()

def download_with_integrity_check(path: str, expected_sha256: Optional[str] = None) -> bytes:
    """Download file from GCS and verify SHA256 integrity."""
    bucket = get_bucket()
    blob = bucket.blob(path)
    blob.reload()  # Get metadata
    data = blob.download_as_bytes()

    actual_hash = hashlib.sha256(data).hexdigest()

    if expected_sha256 is None:
        # Try to get expected hash from blob metadata
        metadata = blob.metadata or {}
        expected_sha256 = metadata.get("sha256")

    if expected_sha256 and actual_hash != expected_sha256:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_hash}"
        )

    return data
