#!/bin/bash

echo "Loging to ECR"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 256305374409.dkr.ecr.us-west-2.amazonaws.com

echo "Selected stage: " ${1}
export STAGE=${1}
envsubst '$STAGE' < Dockerfile >> Dockerfile.temp
docker build -t blogs-reco-system . -f Dockerfile.temp
docker tag blogs-reco-system:latest 256305374409.dkr.ecr.us-west-2.amazonaws.com/blogs-reco-system:latest

echo "Pushing"
docker push 256305374409.dkr.ecr.us-west-2.amazonaws.com/blogs-reco-system:latest

echo "Cleaning"
rm Dockerfile.temp
