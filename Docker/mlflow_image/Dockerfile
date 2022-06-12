# docker build -f Docker/mlflow_image/Dockerfile -t mlflow_server .
FROM python:3.10.4

# to correct SSL error
RUN pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org raw.githubusercontent.com github.com"
# install python libs for mlflow
RUN pip install mlflow boto3 psycopg2
