#! /usr/bin/bash
echo "making files executable"

chmod +x serve cache.py

echo "building container"

image_name=koombea_blogs_serve_component
docker-compose build $image_name

echo "cleaning docker cache images"
echo y | docker system prune

echo "executing container"
docker-compose up $image_name

