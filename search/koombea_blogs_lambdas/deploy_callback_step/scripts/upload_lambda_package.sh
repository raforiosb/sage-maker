#!/usr/bin/bash
reinstall=${1:-"reinstall"}
re="reinstall"
if [ "$reinstall" = "$re" ]; then
    echo "installing packages"
    pip install --target ./package --upgrade -r requirements.txt
fi

echo "zipping packages"
cd package
zip -r ../deploy_callback_step.zip .

echo "adding lambda_function.py"
cd ..
zip -g deploy_callback_step.zip lambda_function.py

echo "uploading to s3"
aws s3 cp deploy_callback_step.zip s3://sagemaker-us-west-2-256305374409/koombea_website_ml/koombea_lambda_execution_zip/deploy_callback_step.zip