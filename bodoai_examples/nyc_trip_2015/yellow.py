"""
This example is based on:
https://matthewrocklin.com/blog/work/2017/01/12/dask-dataframes


"""
import time

import bodo
import pandas as pd

"""
# Load data from CSV files

For an experimental purpose, it is using just the first 3 months of 2015,
(yellow_tripdata_2015-01.csv, yellow_tripdata_2015-03.csv,
yellow_tripdata_2015-02.csv), it is about 6GB in disk and
30 GB in RAM (and it took around 2.5 minutes just to be loaded into memory).
"""


@bodo.jit
def read_csv():
    # return pd.read_csv('/work/bodoai/dataset/nyc-trip-2015/')
    return pd.read_csv(
        '/work/bodoai/dataset/sample.csv',
        parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'],
    )


t0 = time.time()
df = read_csv()
t1 = time.time()

print('read_csv execution time:', t1 - t0, 's')

"""
This data set has the following structure
"""

df.info()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 19 columns):
 #   Column                 Non-Null Count  Dtype
---  ------                 --------------  -----
 0   VendorID               5 non-null      Int64
 1   tpep_pickup_datetime   5 non-null      datetime64[ns]
 2   tpep_dropoff_datetime  5 non-null      datetime64[ns]
 3   passenger_count        5 non-null      Int64
 4   trip_distance          5 non-null      float64
 5   pickup_longitude       5 non-null      float64
 6   pickup_latitude        5 non-null      float64
 7   RateCodeID             5 non-null      Int64
 8   store_and_fwd_flag     5 non-null      object
 9   dropoff_longitude      4 non-null      float64
 10  dropoff_latitude       4 non-null      float64
 11  payment_type           4 non-null      float64
 12  fare_amount            4 non-null      float64
 13  extra                  4 non-null      float64
 14  mta_tax                4 non-null      float64
 15  tip_amount             4 non-null      float64
 16  tolls_amount           4 non-null      float64
 17  improvement_surcharge  4 non-null      float64
 18  total_amount           4 non-null      float64
dtypes: Int64(3), datetime64[ns](2), float64(13), object(1)
"""

"""
## Basic Aggregations and Groupbys

Let's try som basic aggregation on our dataframe:
"""


@bodo.jit
def mean_by_each_passanger_count(df):
    return df.groupby(df.passenger_count).trip_distance.mean()


t0 = time.time()
se = mean_by_each_passanger_count(df)
t1 = time.time()

print('mean_by_each_passanger_count execution time:', t1 - t0, 's')

print(se)
