
TEST_PARAMS:=

start-services:
	docker-compose --file docker/docker-compose.yml up -d

test:
	env `cat .env` mpirun -n 4 python -m pytest -v -s ${TEST_PARAMS}

create-conda-env:
	env `cat .env` ./devops/create_conda_env.sh

create-passwd-s3fs:
	env `cat .env` ./devops/create_passwd_s3fs.sh

mount-s3fs: create-passwd-s3fs
	env `cat .env` ./devops/mount_s3fs.sh

develop:
	pip install -e .
	pre-commit install
