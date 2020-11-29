#!/usr/bin/env bash

echo ${ACCESS_KEY_ID}:${SECRET_ACCESS_KEY} > ${HOME}/.passwd-s3fs
chmod 600 ${HOME}/.passwd-s3fs
