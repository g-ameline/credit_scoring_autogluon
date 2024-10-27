FROM mambaorg/micromamba:latest
# repo_root/
# ├ virtual_environment/
# │   ├ environment.yml 
# │   ├ requirements.txt
# │   └ ...
# ├ notebooks/
# │   └ ...
# ├ python_scripts/
# │   └ ...
# :

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG VIRTUAL_ENVIRONMENT_FILEPATH
ARG KERNEL_NAME=notebook_kernel
ARG PORT=1234
ENV PORT=$PORT
RUN export JUPYTER_PORT=$port

RUN micromamba info
RUN eval "$(micromamba shell hook --shell bash)"
COPY --chown=$MAMBA_USER:$MAMBA_USER $VIRTUAL_ENVIRONMENT_FILEPATH /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER ./notebooks/ ./notebooks/
RUN micromamba install --channel conda-forge --yes --name base --file /tmp/env.yaml && \
    micromamba clean --all --yes
RUN micromamba run --name=base python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
RUN touch you_are_in_a_container_btw
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
# RUN micromamba install --channel conda-forge --yes jupyter
# RUN micromamba install --channel conda-forge --yes jupyterlab
# RUN micromamba install --channel conda-forge --yes nb_conda_kernels
# RUN micromamba install --channel conda-forge --yes ipykernel
RUN micromamba run \
  python \
    -m ipykernel \
    install \
      --user \
      --name base_kernel \
      --display-name $KERNEL_NAME
CMD jupyter lab --no-browser --port=$PORT --ip=0.0.0.0

