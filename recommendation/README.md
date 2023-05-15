# BLOG RECOMENDATION

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

`Step 1: Creating environment (Kernel for Notebooks)`

```shell
./scripts/set_up_kernel.sh

```
<br>

`Step 2: Setting environment variables and SSH files`

Copy the PEM file into **src/app/config**

<br>

Edit the ***config_variables.py*** file put the name of the PEM file where the ***CONFIG_DIRECTORY*** variable is located

<br>

`Step 3: Building the Docker image and uploading to the ECR container. Use "dev" for development or "prod" for production`

```shell
./scripts/build_and_run.sh dev
```
```shell
./scripts/clean_test.sh 
```
```shell
./scripts/build_and_push.sh dev
```

-------------
## Part 4

In the navigation bar select the ***`notebooks`*** folder and run from the `Jupyter` console the notebooks in the following order.

<br>

> Popular-Blogs.ipynb

> Prepare-Data.ipynb

> RecoSystem-ItemContentBased.ipynb

> Sagemaker-Deploy.ipynb
}

-------------

## Part 5

Once you finish executing the process described in `Part 4` in `API GATEWAY` check the API service called `koombea-blogs-recommendation`. Go to the `stage variables` tab, check if the `sm_recommendation_endpoint` variable is defined with the value `blogsreco-stage-dev-2020-07-27-19-34-26` for the `dev` environment and `blogsreco-stage-prod-2020-07-27-20-19-34` for the `prod` environment.

<br>

You should also review the `Lambda service` called `lambda-blogs-recommendation` in order to verify that it is active.

-------------