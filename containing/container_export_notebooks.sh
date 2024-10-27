#!/usr/bin/env bash
echo "extracting notebooks from container"
docker container cp \
  notebooke_container:./notebooks \
  ./extracted_notebooks

