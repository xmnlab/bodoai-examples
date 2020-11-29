#!/usr/bin/env bash

DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pushd ${DEVOPS_DIR}
source ../.env
popd

cp ${DEVOPS_DIR}/environment-template.yml ${DEVOPS_DIR}/environment-dev.yml

sed -i "s/-\ python$/-\ python ${PYTHON_VERSION:-3.8.*}/" ${DEVOPS_DIR}/environment-dev.yml
sed -i "s/{{\s*BODO_CONDA_USERNAME\s*}}/${BODO_CONDA_USERNAME:-test-user}/" ${DEVOPS_DIR}/environment-dev.yml
sed -i "s/{{\s*BODO_CONDA_TOKEN\s*}}/${BODO_CONDA_TOKEN:-test-token}/" ${DEVOPS_DIR}/environment-dev.yml

# cat ${DOCKER_DIR}/environment-dev.yml

conda env create --name bodoai-examples-dev --file ${DEVOPS_DIR}/environment-dev.yml --force
