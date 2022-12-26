#!/usr/bin/env python
# coding: utf-8

"""
Mini project - MNIST digits - Hyperparameter tuning
"""

import logging
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from mnist import MNIST
from swampy import structshape as ss
import pandas as pd
import numpy as np
import joblib
from joblib import parallel_backend
import matplotlib.pyplot as plt
from feature_engine.wrappers import SklearnTransformerWrapper
import json
from datetime import datetime


def ingest(NUM_TRAIN_INSTANCES=1000):
    """ Load the MNIST data into dataframes """
    data_dir = "./emnist_data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(data_dir + "does not exist.")

    mndata = MNIST('./emnist_data')
    mndata.select_emnist()

    X_test, y_test = mndata.load_testing()
    X_train, y_train = mndata.load_training()

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    # Reduce size of data
    X_train, y_train = X_train[:NUM_TRAIN_INSTANCES], y_train[0:NUM_TRAIN_INSTANCES]
    return X_train, y_train, X_test, y_test


def remove_low_variance_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """ Remove features with very little information content """
    low_variance_selector = SklearnTransformerWrapper(
        VarianceThreshold(threshold=0.01))
    low_variance_selector.fit(X_train)
    assert len(X_train.columns == 784) and len(X_test.columns == 784)

    X_train = low_variance_selector.transform(X_train)
    X_test = low_variance_selector.transform(X_test)
    assert len(X_train.columns) == len(X_test.columns)
    return X_train, X_test


def save_metadata(out_file, estimator: GridSearchCV,
                  dfs: List[pd.DataFrame], META_DIR):
    """ Save metadata to file """
    meta_file = f"{META_DIR}/meta_{out_file}_.json"
    meta = {}
    if not os.path.exists(meta_file):
        meta['date_created'] = str(datetime.now())
        meta['results_file'] = out_file
        for idx, df in enumerate(dfs):
            meta['df_' + str(idx) + "_shape"] = df.shape
        meta['final_estimator'] = str(
            type(estimator.estimator._final_estimator))
        meta['params'] = estimator.estimator._final_estimator.get_params()
        meta['cv'] = estimator.get_params()['cv']
        meta['pipeline_steps'] = str(estimator.estimator.named_steps.keys())
        with open(meta_file, "w") as f:
            f.write(json.dumps(meta))
    else:
        logging.warning(f"meta_file {meta_file} exists; It was left as-is")


def tune_model(X_train, y_train, pipe, param_grid,
               joblib_file_name, MODEL_DIR):
    """ Tune model to find best # of nearest neighbors """
    logging.warning("tune_model() started")
    grid_file = joblib_file_name
    grid = 0
    if not os.path.exists(f"{MODEL_DIR}/{grid_file}"):
        grid = GridSearchCV(estimator=pipe, param_grid=param_grid,
                            scoring="accuracy", cv=12, n_jobs=2)
        grid.fit(X_train, y_train)
        save_metadata(grid_file, grid, [X_train, y_train], META_DIR=MODEL_DIR)
        joblib.dump(grid, filename=f"{MODEL_DIR}/{grid_file}", compress=5)
    else:
        grid = joblib.load(filename=f"{MODEL_DIR}/{grid_file}")
    return grid


def plot_cv_results(grid, param_name, title_add_on):
    """ Plot GridSearchCV results"""
    grid_results = grid.cv_results_
    n_neighbors = pd.Series(grid_results[param_name])
    mean_test = pd.Series(grid_results['mean_test_score'])
    std_test = pd.Series(grid_results['std_test_score'])
    fig = plt.figure()
    sns.set_palette('tab10')
    sns.pointplot(x=n_neighbors,
                  y=mean_test, label='mean test score')
    sns.pointplot(x=n_neighbors,
                  y=mean_test - std_test, color='green', join=True, scale=0.8, linestyles=':', markers='+', label='+/- std dev')
    sns.pointplot(x=n_neighbors,
                  y=mean_test + std_test, color='green', join=True, scale=0.8, linestyles=':', markers='+')
    plt.xlabel(param_name)
    plt.ylabel("test score")
    plt.title("Hypertuning cv=" + str(grid.get_params()['cv']) + title_add_on)
    plt.grid(visible=True)
    plt.legend()
    return fig


if __name__ == "__main__":

    ###########################################################
    # NOTE Config Params Start
    ##########################################################
    RUN_NAME = "gridsearch4"
    LOG_FILE = f"log_{RUN_NAME}.md"
    TRAIN_DATA_SIZE = 150_000
    # MODEL_ABBREV = "rf"
    MODEL_ABBREV = "mlp"
    MODEL = MLPClassifier(warm_start=True, solver="adam", hidden_layer_sizes=(150,100), random_state=42, early_stopping=True, validation_fraction=0.1,max_iter=500)
    # MODEL = RandomForestClassifier(n_estimators=(500), random_state=42)
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    FIGURE_DIR = "./figures"
    VARIED_PARAM = "learning_rate_init"
    # VARIED_PARAM_LIST = [(100,), (100, 100), (150, 100), (150, 100, 100),]
    # VARIED_PARAM_LIST = list(range(2, 10, 2))
    VARIED_PARAM_LIST = 0.001, 0.009, 0.02
    GRIDSEARCH_PLOT_TITLE_NOTES = f" instances={TRAIN_DATA_SIZE}"
    ###########################################################
    # NOTE Config Params End
    ###########################################################

    FORMAT = "%(asctime)s %(message)s"
    logging.basicConfig(
        format=FORMAT,
        level=logging.WARN,
        filename=f"{LOG_DIR}/{LOG_FILE}")

    X_train, y_train, X_test, y_test = ingest(
        NUM_TRAIN_INSTANCES=TRAIN_DATA_SIZE)
    logging.warning("Ingestion completed")
    logging.warning(f"y_train value counts=\n{y_train.value_counts()}")
    X_train, X_test = remove_low_variance_features(X_train, X_test)
    logging.warning("Low variance features removed")

    my_scaler = SklearnTransformerWrapper(MinMaxScaler())
    pipe = Pipeline(
        steps=[("scale", my_scaler), (MODEL_ABBREV, MODEL)])

    param_grid = {f"{MODEL_ABBREV}__{VARIED_PARAM}": VARIED_PARAM_LIST}

    grid = {}
    with parallel_backend('multiprocessing'):
        grid = tune_model(X_train=X_train, y_train=y_train,
                          pipe=pipe, param_grid=param_grid, joblib_file_name=f"{RUN_NAME}.pkl", MODEL_DIR=MODEL_DIR)

    logging.warning(f"best parameters = {grid.best_params_}")

    fig = plot_cv_results(grid, f"param_{MODEL_ABBREV}__{VARIED_PARAM}",
                          f" instances={TRAIN_DATA_SIZE}")
    fig.savefig(f"{FIGURE_DIR}/{RUN_NAME}.png")
    logging.warning("Grid search complete")
