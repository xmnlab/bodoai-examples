FROM condaforge/miniforge3

ARG PYTHON_VERSION
ADD docker/environment-template.yml /environment-dev.yml

RUN sed -i "s/-\ python$/-\ python=${PYTHON_VERSION:-3.8.*}/" /environment-dev.yml

# RUN conda update --all --yes --quiet \
#   && conda clean -afy

RUN cat /environment-dev.yml

# RUN conda env create --name bodo-env --file /environment-dev.yml \
#   && conda clean -afy
#
# RUN echo 'source /opt/conda/bin/activate toki-env && exec "$@"' > activate.sh
#
# COPY . /toki
# WORKDIR /toki
#
# RUN bash /activate.sh pip install -e . --no-deps --ignore-installed --no-cache-dir
#
# ENTRYPOINT ["bash", "/activate.sh"]
