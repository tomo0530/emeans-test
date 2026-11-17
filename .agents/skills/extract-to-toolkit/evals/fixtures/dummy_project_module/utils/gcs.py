"""Thin wrapper around google-cloud-storage for uploading and downloading blobs."""

from google.cloud import storage


def upload_file(bucket_name: str, source_path: str, destination_blob: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_path)


def download_file(bucket_name: str, source_blob: str, destination_path: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(destination_path)
