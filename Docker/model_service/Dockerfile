FROM python:3.10.4

# Envs
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_ENV=windmill_power_prediction

WORKDIR /code

# Solve SSL problem
RUN pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org raw.githubusercontent.com github.com"

# Package manager, could be poetry if use poetry
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate\
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Copy files for package manager, e.g. for poetry is pyproject.toml and poetry.lock
COPY ./Docker/model_service/conda.yaml /code/

# Copy code
COPY ./src/app /code/app
# COPY ./Docker/model_service/.env /code/app/.env

# Disable SSL and install dependencies
RUN conda config --set ssl_verify no
RUN conda env create -f /code/conda.yaml
# RUN poetry config virtualenvs.create false \ 
#     && poetry install $(test "$YOUR_ENV" -- production && echo "--no-dev") \
#     --no-inerastion --no-ansi

CMD ["sh", "-c", "conda run --no-capture-output -n ${CONDA_ENV} \
    python -m uvicorn app.inference:app --host 0.0.0.0 --port 80 --reload"]
