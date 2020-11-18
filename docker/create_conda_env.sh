#!/usr/bin/env bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cp ${CURRENT_DIR}/environment-template.yml ${CURRENT_DIR}/environment-dev.yml

sed -i "s/-\ python$/-\ python=${PYTHON_VERSION:-3.8.*}/" ${CURRENT_DIR}/environment-dev.yml
sed -i "s/{{\s*BODO_CONDA_USERNAME\s*}}/${BODO_CONDA_USERNAME:-test-user.*}/" ${CURRENT_DIR}/environment-dev.yml
sed -i "s/{{\s*BODO_CONDA_TOKEN\s*}}/${BODO_CONDA_TOKEN:-test-token.*}/" ${CURRENT_DIR}/environment-dev.yml

cat ${CURRENT_DIR}/environment-dev.yml

# RUN conda env create --name bodo-env --file ${CURRENT_DIR}/environment-dev.yml \
#   && conda clean -afy
