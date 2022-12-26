#!/usr/bin/env python
# coding: utf-8
"""
Mini project - MNIST digits - EDA
"""
from mnist import MNIST
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.spatial as sp
import logging
import pdb
###############################################
# Setup logging
FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(
    format=FORMAT,
    level=logging.WARN)
###############################################
# Constants
FIGURE_DIR = "./figures"
###############################################


def plot_digit_cdf(df):
    # Distribution of each number as we move from top to bottom in the dataset
    plt.figure()
    tab10 = {}
    tab10.update(mpl.colors.TABLEAU_COLORS)
    tab10 = list(tab10.values())
    assert isinstance(tab10, list)

    for curr_num in range(10):
        def selector(x): return 1 if x == curr_num else 0
        x = df.index
        y = df.tgt.map(arg=selector).cumsum()
        x, y = x[:], y[:]

        # getting data of the histogram
        count, bins_count = np.histogram(y, bins=500)
        # finding the PDF of the histogram using count values
        pdf = count / sum(count)

        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        cdf = np.cumsum(pdf)

        # plotting CDF
        plt.plot(bins_count[1:], cdf, color=tab10[curr_num],
                 label=f'CDF digit {curr_num}')
    plt.ylabel("Cumulative Distribution Function")
    plt.xlabel("Bin Number")
    plt.legend()
    plt.savefig(f"{FIGURE_DIR}/eda_cdf_digits.png")
    plt.show()


def plot_mean_digit(df, digit, ax):
    df = df[df.tgt == digit].sum()
    df.pop('tgt')
    df = df.to_numpy().reshape((28, 28))
    avg = (df - df.min()) / (df.max() - df.min())
    ax.set_title(str(digit))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(avg, interpolation='none')
    # return the average digit
    return (digit, avg)


def plot_grid_of_mean_digits(df):
    average_digits = {}  # dictionary holding average digits
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
    for i in range(0, 2):
        for j in range(0, 5):
            digit, avg = plot_mean_digit(df, digit=(i)*5+j, ax=ax[i][j])
            # save the average digit picture to a dictionary
            average_digits[digit] = avg
    plt.tight_layout(h_pad=0.1, pad=0.1)
    plt.tight_layout()
    fig.suptitle(f"Averaged (mean) digits", fontsize=22)
    plt.savefig(f"{FIGURE_DIR}/eda_mean_digit_grid.png")
    return fig, average_digits


def plot_oddball_digit(df, digit, average_digit, distance_metric, ax):
    """ plot the digit with largest distance metric from average_digit """
    df = df[df.tgt == digit]
    df.pop('tgt')
    # find the oddball digit
    oddball = np.ndarray(shape=(28, 28))
    max_dist = -99
    oddball_row_id = -99
    for idx, row in df.iterrows():
        row = row.to_numpy().reshape((28, 28))
        row = (row - row.min()) / (row.max() - row.min())
        distance = np.absolute(sp.distance.cdist(row, average_digit, metric=distance_metric)).sum()
        assert np.isscalar(distance)
        if np.isnan(distance): 
            pdb.set_trace()
        if np.isinf(distance):
            pdb.set_trace()
        if distance > max_dist:
            max_dist = distance
            oddball = row 
            oddball_row_id = idx
    print(oddball_row_id)
    ax.set_title(str(digit))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(oddball, interpolation='none')
    # return oddball index to validate anomalies found 
    return oddball_row_id
    # return the average digit


def plot_grid_of_oddball_digits(df, average_digits, distance_metric):
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
    oddball_row_ids = []
    for i in range(0, 2):
        for j in range(0, 5):
            linear_index = i*5+j
            average_digit = average_digits[linear_index]
            oddball_row_id = plot_oddball_digit(df, digit=linear_index, average_digit=average_digit,
                               distance_metric=distance_metric, ax=ax[i][j])
            oddball_row_ids.append(oddball_row_id)
    plt.tight_layout(h_pad=0.1, pad=0.1)
    plt.tight_layout()
    fig.suptitle(f"Outliers found with {distance_metric} metric", fontsize=22)
    plt.savefig(f"{FIGURE_DIR}/eda_oddball_{distance_metric}_digit_grid.png")
    return fig, oddball_row_ids

if __name__ == "__main__":
    # Load data 
    loader = MNIST("./emnist_data")
    loader.select_emnist()
    X, y = loader.load_testing()
    X, y = pd.DataFrame(X), pd.Series(y)
    df = pd.concat([X, y], axis=1)
    names = df.columns.tolist()
    names[-1] = 'tgt'
    df.columns = names

    plt.rcParams['savefig.facecolor'] = "0.8"
    fig = plot_digit_cdf(df)
    fig, average_digits = plot_grid_of_mean_digits(df)
    distance_metrics = 'canberra', 'chebyshev', 'cityblock', 'euclidean', 'hamming', 'jaccard'
    # correlation, cosine, dice, kulczynski1 - metrics removed because failed 
    oddball_row_ids = {}
    for distance_metric in distance_metrics:
        fig, oddball_row_ids = plot_grid_of_oddball_digits(
            df, average_digits=average_digits, distance_metric=distance_metric)
