"""
original code available at:
    https://github.com/udacity/deep-learning-v2-pytorch/
        tree/master/intro-neural-networks/gradient-descent


Implementing the Gradient Descent Algorithm

In this lab, we'll implement the basic functions of the Gradient Descent
algorithm to find the boundary in a small dataset. First, we'll start with
some functions that will help us plot and visualize the data.


"""
import time

import bodo
import numpy as np
import pandas as pd

# local
from bodoai_examples.utils import bd_zip


def setup():
    url = (
        'https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/'
        'master/intro-neural-networks/gradient-descent/data.csv'
    )
    data = pd.read_csv(url, header=None)
    data.to_csv('/tmp/data.csv', header=None, index=None)

    data_check = pd.read_csv(url, header=None)
    pd.testing.assert_frame_equal(data, data_check)

    # duplicate data just for benchmark propose
    dfs = []
    for i in range(100):
        dfs.append(data)
    data = pd.concat(dfs).reset_index(drop=True)

    data.to_csv('/tmp/data_10k.csv', header=None, index=None)


# NON BODO AI FUNCTIONS


def read_data():
    """
    Return X (features) and y (target) data.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
    """
    data = pd.read_csv('/tmp/data_10k.csv', header=None)

    X = np.array(data[[0, 1]])
    y = np.array(data[2])
    return X, y


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)


def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias


def train(features, targets, epochs, learnrate, graph_lines=False):
    """
    Training function

    This function will help us iterate the gradient descent algorithm through
    all the data, for a number of epochs. It will also plot the data, and
    some of the boundary lines obtained as we run the algorithm.


    Parameters
    ----------
    features : numpy.ndarray
    targets : numpy.ndarray
    epochs : int
    learnrate : float
    graph_lines : bool, optional
        by default False
    """
    errors = []
    n_records, n_features = features.shape
    last_loss = 99999999999.9
    weights = np.random.normal(0.0, 1 / n_features ** 0.5, n_features)
    bias = 0

    for e in range(epochs):
        for x, y in zip(features, targets):
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)  # noqa: F841
            weights, bias = update_weights(x, y, weights, bias, learnrate)

        # Printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean((predictions == targets).astype(int))
            print("Accuracy: ", accuracy)
        # if graph_lines and e % (epochs / 100) == 0:
        #     print(-weights[0]/weights[1], -bias/weights[1])


# BODO AI FUNCTIONS


@bodo.jit
def bd_read_data():
    """
    Return X (features) and y (target) data.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
    """
    data = pd.read_csv('/tmp/data_10k.csv', header=None)

    X = data[['0', '1']].values
    y = data['2'].values
    return X, y


# Activation (sigmoid) function
@bodo.jit
def bd_sigmoid(x):
    return 1 / (1 + np.exp(-x))


@bodo.jit
def bd_output_formula(features, weights, bias):
    return bd_sigmoid(np.dot(features, weights) + bias)


@bodo.jit
def bd_error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


@bodo.jit
def bd_update_weights(x, y, weights, bias, learnrate):
    output = bd_output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias


@bodo.jit
def bd_train(features, targets, epochs, learnrate, graph_lines=False):
    """
    Training function

    This function will help us iterate the gradient descent algorithm through
    all the data, for a number of epochs. It will also plot the data, and
    some of the boundary lines obtained as we run the algorithm.


    Parameters
    ----------
    features : numpy.ndarray
    targets : numpy.ndarray
    epochs : int
    learnrate : float
    graph_lines : bool, optional
        by default False
    """
    errors = []
    n_records, n_features = features.shape
    last_loss = 99999999999.9
    weights = np.random.normal(0.0, 1 / n_features ** 0.5, n_features)
    bias = 0

    for e in range(epochs):
        for x, y in bd_zip(features, targets):
            output = bd_output_formula(x, weights, bias)
            error = bd_error_formula(y, output)  # noqa: F841
            weights, bias = bd_update_weights(x, y, weights, bias, learnrate)

        # Printing out the log-loss error on the training set
        out = bd_output_formula(features, weights, bias)
        loss = np.mean(bd_error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean((predictions == targets).astype(int))
            print("Accuracy: ", accuracy)
        # if graph_lines and e % (epochs / 100) == 0:
        #     print(-weights[0]/weights[1], -bias/weights[1])


def main():
    setup()

    # benchmark for NON bodo ai training
    if bodo.get_rank() == 0:
        t0 = time.time()

        np.random.seed(44)
        epochs = 100
        learnrate = 0.01

        X, y = read_data()
        print(
            '\n\nNON bodoai training, X.shape:', X.shape, ', y.shape:', y.shape
        )

        train(X, y, epochs, learnrate, True)

        print('\n\nTime for NON bodoai training:', time.time() - t0, 's\n\n')

        print('=' * 80, '\n\n')

    # benchmark for bodo ai training
    t0 = time.time()

    np.random.seed(44)
    epochs = 100
    learnrate = 0.01

    bd_X, bd_y = bd_read_data()

    print(
        '\n\nbodoai training, X.shape:', bd_X.shape, ', y.shape:', bd_y.shape
    )

    bd_train(bd_X, bd_y, epochs, learnrate, True)

    print('\n\nTime for bodoai training:', time.time() - t0, 's\n\n')


if __name__ == '__main__':
    main()
