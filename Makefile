
TEST_PARAMS:=
FILEPATH:=
NUM_PROCESSES:=1

start-services:
	docker-compose --file docker/docker-compose.yml up -d

run:
	env `cat .env` mpirun -n ${NUM_PROCESSES} python ${FILEPATH}

test:
	env `cat .env` mpirun -n ${NUM_PROCESSES} python -m pytest -v -s ${TEST_PARAMS}

create-conda-env:
	env `cat .env` ./devops/create_conda_env.sh

create-passwd-s3fs:
	env `cat .env` ./devops/create_passwd_s3fs.sh

mount-s3fs: create-passwd-s3fs
	env `cat .env` ./devops/mount_s3fs.sh

develop:
	pip install -e .
	pre-commit install
