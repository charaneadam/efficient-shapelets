# Efficient Shapelets
This repo contains the code for the paper [Efficient Shapelets Selection
](https://github.com/charaneadam/shapelts-docs).

## Code

Use ***container.sh*** to run experiments, open the web application or launch a
Jupyter instance.

To open the web application, run:
> ./container run webapp

To open an instance of Jupyter, run:
> ./container run jupyter

To get access to bash with all priviliges, run:
> ./container run bash

To stop the container, run:
> ./container stop

To build the image (this is only needed if you change the Containerfile), run:
> ./container build

When you run Jupyter or the web application, you have to browse to:
- Jupyter: [localhost:8888](http://localhost:8888)
- Web app: [localhost:8501](http://localhost:8501)

## Short documentation:
When you first use the ***container.sh***, a Postgres image with Miniconda will
be built. Metadata of the UCR archive will be added to the database.
A [volume](https://docs.podman.io/en/latest/markdown/podman-volume.1.html) will
be created to store the database content outside the container, and the src
directory will be mounted on the container (in /code), i.e., you can change 
the code in src directory without the need to rebuild/rerun the container.

- storage: This module deals with the data, and the database
    - data: Provides classes to get raw time series and generate sliding,
    and hopping windows, as well as other necessary utilities.
    - database: This module offer ORM models to be used in order to store and
    fetch the results from the database,
    - init_db: this script fetchs and inserts metadata about UCR into the
    database, create the necessary tables, and optionally check for any missing
    data in the time series from UCR.

- benchmarks: This module contains scripts to benchmark the methods:
    - Compare all methods with a set of default parameters
    - Compare different combinations of Kmeans method

- methods: This module contains methods to select shapelets. Each method is in
its own class. Each class can have different arguments/hyperparameters, however,
every class should have at least two methods:
    - fit: which takes train data (X, y) and select shapelets.
    - transform: which takes a list of time series (X) and return the shapelet
    transform of the data.

- classifiers.py: This module contains the list of classifiers with their 
parameters as well as a Classifier class that takes a classifier name as
argument and contains two methods: fit and predict (same api as sklean).

- exceptions.py: This module contains exceptions to be raised and used for logging.
