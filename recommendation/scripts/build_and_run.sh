#!/bin/bash

export STAGE=${1:-'dev'}
echo "selected stage: " $STAGE

envsubst '$STAGE' < Dockerfile >> Dockerfile.temp
docker build --rm -t blogs-reco-system . -f Dockerfile.temp
rm -f Dockerfile.temp

chmod +x $(pwd)/src/serve
chmod +x $(pwd)/src/cache.py

# docker run --rm -p 8080:8080 -it blogs-reco-system serve
docker run --rm -p 8080:8080 -v $(pwd)/src:/opt/ml/code -v $(pwd)/model:/opt/ml/model -e MODEL_SERVER_TIMEOUT=1000 -e MODEL_SERVER_WORKERS=1 -it blogs-reco-system serve
