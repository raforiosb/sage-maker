# BLOG SEARCH

## Part 1

With the assigned user within the AWS account ID `256305374409` log in and change to the `us-west-2` region

<br>

Make sure you have access to the GITHUB repository [Koombea-website-ml](https://github.com/koombea/koombea-website-ml)

<br>

In `SageMaker` under the `Notebook` option select ***`Notebook instances`***. Here you must validate that the instance called `deploy-stg` is active and working with the state ***`InService`***. If the instance is off, it must be initialized and go to `Part 3`. If the instance cannot be initialized or does not exist, go to `Part 2`.

-------------    
## Part 2

Create a `Notebook` in `SageMaker`, this is equivalent to a specialized instance for the execution of `Jupyter` and `Python` type files where the entire configuration and deployment process of the applications will be done.

<br>

It is recommended when creating the `Notebook` to connect to the `GITHUB` repository of the application to have the code available and the `GIT` utilities installed.

-------------
<br>

## Part 3

Once the `Jupyter Lab` instance has been initialized in a terminal, the following steps must be performed.

<br>

`Step 1: Setting environment variables and SSH files`

- Copy the PEM file into **`/home/ec2-user/SageMaker/search/koombea_blogs_extraction`**
- Copy _vars.env, vars.prod.env and vars.staging.env_ files into **`/home/ec2-user/SageMaker/search/koombea_blogs_extraction`**
- Copy the PEM file into **`/home/ec2-user/SageMaker/search/koombea_blogs_serve`**
- Copy _vars.env, vars.prod.env and vars.staging.env_ files into **`/home/ec2-user/SageMaker/search/koombea_blogs_serve`**
- Copy _vars.env, vars.prod.env and vars.staging.env_ files into **`/home/ec2-user/SageMaker/search/koombea_blogs_train`**

<br>

`Step 2: Building the Docker image and uploading to the ECR container. Use "dev" for development or "prod" for production`

```shell
docker-compose up
```
```shell
docker-compose build
```
```shell
bash ./scripts/build_and_push_ecr.sh 
```

-------------

## Part 4

In each folder you must run the following ***`notebooks`***.

<br>

> /home/ec2-user/SageMaker/search/koombea_blogs_extraction/notebooks/sagemaker_processor.ipynb

> /home/ec2-user/SageMaker/search/koombea_blogs_train/notebooks/sagemaker_trainer.ipynb

> /home/ec2-user/SageMaker/search/koombea_blogs_serve/notebooks/sagemaker_inference.ipynb

> /home/ec2-user/SageMaker/search/koombea_blogs_lambdas/update_execute_pipeline_callback/DeployLambda.ipynb

-------------

## Part 5

Once you finish executing the process described in `Part 4` in `LAMBDA SERVICES` check the service called `koombea-website-search-prod`. Go to the `Test` tab, and make a test. Also you can go to the follow [URL link](https://tn02epjb2d.execute-api.us-west-2.amazonaws.com/prod/search?per_page=12&page=1&s=custom&lang=en) and test it

<br>

For the `staging` environment you should go to the `LAMBDA` service called `koombea-search-dev` and test it. Additionally you can try the following [URL link](https://tn02epjb2d.execute-api.us-west-2.amazonaws.com/staging/search?per_page=12&page=1&s=custom&lang=en)

-------------
