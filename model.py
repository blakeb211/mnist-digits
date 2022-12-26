#!/usr/bin/env python
# coding: utf-8

"""
Mini project - MNIST digits - ML Modeling
"""

from typing import List
import inspect
import logging
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import f1_score

import seaborn as sns
from mnist import MNIST
from swampy import structshape as ss
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from feature_engine.wrappers import SklearnTransformerWrapper
import json
from datetime import datetime


def ingest():
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
    X_train, y_train = X_train, y_train
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


def save_model_metadata(out_file, estimator,
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
            type(estimator._final_estimator))
        meta['params'] = estimator._final_estimator.get_params()
        meta['pipeline_steps'] = str(estimator.named_steps.keys())
        with open(meta_file, "w") as f:
            f.write(json.dumps(meta))
    else:
        logging.info(f"meta_file {meta_file} exists; It was left as-is")


def train_model(X_train, y_train, pipe: Pipeline, joblib_file_name, MODEL_DIR):
    """ Tune model to find best # of nearest neighbors """
    logging.warning(f"{inspect.stack()[0][3]} started")
    grid_file = joblib_file_name
    grid = Pipeline(steps=[])
    if not os.path.exists(f"{MODEL_DIR}/{grid_file}"):
        grid = pipe
        grid.fit(X_train, y_train)
        save_model_metadata(
            grid_file, grid, [X_train, y_train], META_DIR=MODEL_DIR)
        joblib.dump(grid, filename=f"{MODEL_DIR}/{grid_file}", compress=5)
    else:
        logging.warning(
            f"grid file {grid_file} found in {MODEL_DIR}. Loading...")
        grid = joblib.load(filename=f"{MODEL_DIR}/{grid_file}")
    return grid


if __name__ == "__main__":
    ###########################################################
    # NOTE Config Params Start
    ###########################################################
    RUN_NAME = "mlp1"
    LOG_FILE = f"log_{RUN_NAME}.md"
    MODEL_DIR = "./models"
    LOG_DIR = "./logs"
    FIGURE_DIR = "./figures"
    ###########################################################
    # NOTE Config Params End
    ###########################################################

    FORMAT = "%(asctime)s %(message)s"
    logging.basicConfig(
        format=FORMAT,
        level=logging.WARN,
        filename=f"{LOG_DIR}/{LOG_FILE}")

    # Ingest
    X_train, y_train, X_test, y_test = ingest()

    # Preprocess
    X_train, X_test = remove_low_variance_features(X_train, X_test)

    # Model
    model = MLPClassifier(hidden_layer_sizes=(
        150,100), learning_rate_init=0.001, random_state=42, 
        early_stopping=False, max_iter=500)
    my_scaler = SklearnTransformerWrapper(MinMaxScaler())
    pipe = Pipeline(steps=[("scale", my_scaler), ("model", model)])

    train_model(X_train, y_train, pipe=pipe,
                joblib_file_name=f"{RUN_NAME}.pkl", MODEL_DIR=MODEL_DIR)

    cm = ConfusionMatrix(estimator=pipe, percent=True)
    cm.fit(X_train, y_train)
    cm.score(X_test, y_test)
    plt.xticks(rotation=90)
    cm.show(outpath=f"{FIGURE_DIR}/model_{RUN_NAME}_confusion_matrix.png")
    scoring_metric = f1_score(y_test, cm.estimator.predict(X_test), average="weighted")
    logging.warning(f"F1_SCORE = {scoring_metric}")
