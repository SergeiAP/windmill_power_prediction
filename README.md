windmill_power_prediction
==============================

Data Science case for windmill power prediction based on weather. Based on Data Challenge of Air Liquide and TotalEnergies companies in 2021. The link of the competition is https://datascience.total.com/fr/challenge/19/details#.

Docker commands:
```bash
docker-compose up -d --build
docker build -f Docker/model_service/Dockerfile -t wpp_model_service .
docker-compose up -d --build app
docker cp ./inference.py wpp_model_service:/code/app/inference.py
```

To connect to database use:
```bash
docker ps
# find the image postgres / container wpp_postgres, copy `CONTAINER ID`
docker inspect `CONTAINER ID`
# copy "IPAddress" in the end of the file, use it for database connection in `pgadmin`
```

To add S3-like (not AWS S3) as dvc remote use the following commands:
```bash
dvc remote add -d remote s3://wind-power-prediciton/dvc
dvc remote modify remote endpointurl http://127.0.0.1:5441
```
then add `access_key_id` and `secret_access_key` in `./dvc/config`.

Check wheather port is busy in Windows/cmd (e.g. 5443)
```bash
netstat -a -n -o | find "5443"`
```

Save conda environment
```bash
conda env export > conda.yml
```

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
