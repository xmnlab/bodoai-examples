"""
See Also
--------

- https://aws.amazon.com/es/premiumsupport/knowledge-center/
    s3-could-not-connect-endpoint-url/
"""
import bodo


@bodo.jit(distributed={"df"})
def write(df, filepath):
    """
    Write a pandas dataframe to a parquet file on AWS S3 alike service.

    Parameters
    ----------
    df : pandas.DataFrame
    filepath : str
    """
    df.to_parquet(filepath)


@bodo.jit(distributed={"df"})
def read(df, filepath):
    """
    Read a parquet file from AWS S3 alike service in pandas dataframe format.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Pandas.DataFrame
    """
    return df.read_parquet(filepath)
