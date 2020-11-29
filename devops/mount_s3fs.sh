#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -ex

mkdir -p ${S3FS_DIR}

# umount ${S3FS_DIR}
# sudo -E ${CONDA_PREFIX}/bin/

s3fs bodoai-bucket ${S3FS_DIR} \
    -o passwd_file=${HOME}/.passwd-s3fs \
    -o url=${S3_URL} \
    -o use_path_request_style \
    -o dbglevel=info -f -o curldbg

set +ex
