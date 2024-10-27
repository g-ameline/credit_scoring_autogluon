#!/usr/bin/env bash
echo "creating container"
port=$(< port.txt)
echo "port :"
echo "  $port"

docker container create  \
  --publish $port:$port \
  --name notebook_container \
  --memory 3g \
  --interactive \
   notebook_image 
