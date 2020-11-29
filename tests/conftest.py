import warnings

import boto3
import botocore
import pytest

from bodoai_examples import settings


@pytest.fixture(scope="function", autouse=True)
def configure_minio():
    s3cli = boto3.client(
        "s3",
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
        endpoint_url=settings.S3_ENDPOINT,
    )

    try:
        s3cli.create_bucket(Bucket=settings.S3_BUCKET)
    except botocore.exceptions.ClientError as msg:
        warnings.warn(msg)
