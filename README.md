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

- data: This module contains class Data, which takes as argument a dataset name,
and loads train and test data.

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

- main.py: This module is used to transform and classify datasets. It has two
main methods:
    - transform_dataset: takes an instance of Data, method_name, params, then 
    transform the data given using the method specified using params.
    It return train and test data (X_train, y_train, X_test, y_test).
    - classify_dataset: This function takes (X_train, y_train, X_test, y_test)
    and classify the dataset using the different classifiers in 
    ***classifiers*** module.

- experiments: This module contains scripts (should be run directly through main
function as modules).
    - methods_comparison: Transform the specified datasets with all shapelet
    selection methods, and classify using all classifiers.
    - Kmeans_params: This script run different combinations of window lengths 
    and number of top shapelets to be selected, on all the selected datasets.
