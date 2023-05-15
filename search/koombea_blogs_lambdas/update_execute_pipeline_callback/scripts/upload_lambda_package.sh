#!/usr/bin/bash
reinstall=${1:-"reinstall"}
re="reinstall"
if [ "$reinstall" = "$re" ]; then
    echo "installing packages"
    "${CONDA_DIR}/envs/${conda_env}/bin/python" --version
    "${CONDA_DIR}/envs/${conda_env}/bin/pip" install --target ./package --upgrade -r requirements.txt
fi

echo "zipping packages"
cd package

find . -type d -name "tests" -exec rm -rfv {} +
find . -type d -name "__pycache__" -exec rm -rfv {} +

zip -r ../update_execute_pipeline.zip .

echo "adding lambda_function.py"
cd ..
zip -g update_execute_pipeline.zip lambda_function.py vars.*

echo "uploading to s3"
aws s3 cp update_execute_pipeline.zip s3://sagemaker-us-west-2-256305374409/koombea_website_ml/koombea_lambda_execution_zip/update_execute_pipeline.zip