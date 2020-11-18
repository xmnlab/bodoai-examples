import bodo


@bodo.jit(distributed={'df'})
def save_to_s3(df, name):
    """
    Save a pandas dataframe to a parquet file on AWS S3 (or minio).

    Parameters
    ----------
    df : pandas.DataFrame
    name : str
    """
    # bodoai doesn't support format string yet
    df.to_parquet(name + 'e.pq')
