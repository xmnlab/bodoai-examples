# S3 Settings
import os

S3_BUCKET = os.environ.get("BODOAI_S3_BUCKET")
S3_SECRET_KEY = os.environ.get("BODOAI_S3_SECRET_KEY")
S3_ACCESS_KEY = os.environ.get("BODOAI_S3_ACCESS_KEY")
S3_REGION = os.environ.get("BODOAI_S3_REGION", None)
S3_ENDPOINT = os.environ.get("BODOAI_S3_ENDPOINT", None)
