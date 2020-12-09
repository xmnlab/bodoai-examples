#!/usr/bin/env bash

URL_ROOT=https://s3.amazonaws.com/nyc-tlc/trip+data
DATA_DIR=/work/bodoai/dataset/nyc-trip-2015

for i in {1..12}; do
    month=`printf "%2.0d\n" $i |sed "s/ /0/"`;
    FILENAME=yellow_tripdata_2015-${month}.csv;
    wget -c ${URL_ROOT}/${FILENAME} -O ${DATA_DIR}/${FILENAME};
done
