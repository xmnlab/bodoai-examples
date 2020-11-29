# bodoai-examples

## Environment

First, create a `.env` file. You can use `.env-template` as reference.

The variables `BODO_CONDA_USERNAME` and `BODO_CONDA_TOKEN` are used for creating the `environment-dev.yml` file. This information is given by `Bodo.ai`.

The variables `ACCESS_KEY_ID` and `SECRET_ACCESS_KEY` is used to connect to your `AWS S3` instance. If you want to use the `MinIO` service provided by the local `docker-compose.yml` file, feel free to define any value for these variables, the `MinIO` service will use these information when start up the service.

The variables `S3FS_DIR`, `S3_URL` and `S3_CREDENTIAL_FILE` are used to configure the access to the `AWS S3` like service.

After setting the environment variables at `.env`, the following steps are the creation of a conda environment and the creation of the directory mounting your `AWS S3` like service locally.

For creating a conda environment, run: `make create-conda-environment`.

For creating a s3fs directory mounting your `AWS S3` like service locally, run: `make mount-s3fs`
