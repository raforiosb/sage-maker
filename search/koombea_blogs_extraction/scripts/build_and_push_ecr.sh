echo "loging to aws ecr"
aws ecr get-login-password --region ${aws_region} | docker login --username AWS --password-stdin ${account_id}.dkr.ecr.${aws_region}.amazonaws.com

echo "building and tagging docker container"
cd ..
docker-compose build ${docker_compose_service_name}
docker tag ${docker_image_name}:latest \
    ${repository_uri}:latest

echo "pushing container"
docker push ${repository_uri}:latest
    
echo "cleaning dockers cache"
echo y | docker system prune
