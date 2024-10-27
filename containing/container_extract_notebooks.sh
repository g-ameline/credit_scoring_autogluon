#!/usr/bin/env bash
echo "extracting notebooks from container"
docker container cp notebook_container:./notbooks ./extracted notebooks
