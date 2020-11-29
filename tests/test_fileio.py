"""
Some useful commands to check the access to the minio server:

- `aws s3 ls s3://bodoai-bucket --endpoint-url http://localhost:9000`

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv

from bodoai_examples import fileio

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path)


@pytest.fixture
def synthetic_data():
    num_groups = 30
    num_rows = 20_000_000
    df = pd.DataFrame(
        {"A": np.arange(num_rows) % num_groups, "B": np.arange(num_rows)}
    )
    return df


def test_s3_mounted(synthetic_data):
    fileio.parquet.write(
        synthetic_data, str(Path(os.getenv("S3FS_DIR")) / "test_s3.pq")
    )
