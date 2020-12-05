"""
Original code available at:
    https://github.com/udacity/deep-learning-v2-pytorch/tree/master/
        intro-neural-networks/student-admissions

# Predicting Student Admissions with Neural Networks

In this notebook, we predict student admissions to graduate school at UCLA
based on three pieces of data:

- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset originally came from here: http://www.ats.ucla.edu/

## Loading the data

To load the data and format it nicely, we will use two very useful packages
called Pandas and Numpy. You can read on the documentation here:

- https://pandas.pydata.org/pandas-docs/stable/
- https://docs.scipy.org/

admit, gre, gpa,   rank
int,   int, float, int
"""
import time
from pathlib import Path

import bodo
import numpy as np
import pandas as pd

# local
from bodoai_examples.utils import bd_zip


def setup():
    # bodo doesn't work well with pd.read_csv with compound path,
    # constant path works well.
    filename = 'student_data.csv'
    filepath = str(Path(__file__).parent / filename)

    data = pd.read_csv(filepath)

    print('Original data shape:', data.shape)

    # duplicate data just for benchmark propose
    dfs = []
    for i in range(100):
        dfs.append(data)

    data = pd.concat(dfs).reset_index(drop=True)
    data.to_csv(str(Path('/tmp/') / filename), index=None)

    print('Replicated data shape:', data.shape)


# ======================================================
# Pure python/numpy
# ======================================================


def read_data():
    # Reading the csv file into a pandas DataFrame
    return pd.read_csv('/tmp/student_data.csv')


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


# Backpropagate the error
# Now it's your turn to shine. Write the error term. Remember that
# this is given by the equation (𝑦−𝑦̂ )𝜎′(𝑥)
def error_term_formula(x, y, output):
    return (y - output) * sigmoid_prime(x)


# Training function
def train_nn(features, targets, epochs, learnrate):

    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(
        0.0, scale=1 / n_features ** 0.5, size=n_features
    )

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            # error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times
            # the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch:", e, end=', ')
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
    print("Finished training!")
    print("=" * 50)
    return weights


def main():
    data = read_data()

    # TODO:  Make dummy variables for rank
    one_hot_data = pd.get_dummies(data, columns=['rank'])

    # Scaling the data
    #
    # The next step is to scale the data. We notice that the range for
    # grades is 1.0-4.0, whereas the range for test scores is roughly
    # 200-800, which is much larger. This means our data is skewed, and
    # that makes it hard for a neural network to handle. Let's fit our
    # two features into a range of 0-1, by dividing the grades by 4.0,
    # and the test score by 800.

    # Making a copy of our data
    processed_data = one_hot_data[:]

    # TODO: Scale the columns
    processed_data['gre'] /= processed_data['gre'].max()
    processed_data['gpa'] /= processed_data['gpa'].max()

    # Splitting the data into Training and Testing
    #
    # In order to test our algorithm, we'll split the data into a Training
    # and a Testing set. The size of the testing set will be 10% of the
    # total data.

    sample = np.random.choice(
        processed_data.index,
        size=int(len(processed_data) * 0.9),
        replace=False,
    )
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(
        sample
    )

    print("Number of training samples is", len(train_data))
    print("Number of testing samples is", len(test_data))

    # Splitting the data into features and targets (labels)
    #
    # Now, as a final step before the training, we'll split the data into
    # features (X) and targets (y).

    features = train_data.drop('admit', axis=1)
    targets = train_data['admit']
    features_test = test_data.drop('admit', axis=1)
    targets_test = test_data['admit']

    # Training the 2-layer Neural Network
    #
    # The following function trains the 2-layer neural network. First,
    # we'll write some helper functions.

    # Neural Network hyperparameters
    epochs = 1000
    learnrate = 0.5

    weights = train_nn(features, targets, epochs, learnrate)

    # Calculate accuracy on test data
    test_out = sigmoid(np.dot(features_test, weights))
    predictions = test_out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))


# ======================================================
# with bodo
# ======================================================


@bodo.jit
def bd_read_data():
    # Reading the csv file into a pandas DataFrame
    # admit, gre, gpa,   rank
    # int,   int, float, int
    return pd.read_csv(
        '/tmp/student_data.csv',
        dtype={
            'admit': np.float64,
            'gre': np.float64,
            'gpa': np.float64,
            'rank': np.int64,
        },
    )


# Activation (sigmoid) function
@bodo.jit
def bd_sigmoid(x):
    return 1 / (1 + np.exp(-x))


@bodo.jit
def bd_sigmoid_prime(x):
    return bd_sigmoid(x) * (1 - bd_sigmoid(x))


@bodo.jit
def bd_error_formula(y, output):
    return -y * np.log(output) - (1 - y) * np.log(1 - output)


# Backpropagate the error
# Now it's your turn to shine. Write the error term. Remember that
# this is given by the equation (𝑦−𝑦̂ )𝜎′(𝑥)
@bodo.jit
def bd_error_term_formula(x, y, output):
    return (y - output) * bd_sigmoid_prime(x)


# Training function
@bodo.jit
def bd_train_nn(features, targets, epochs, learnrate):

    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = -9999999999.9

    # Initialize weights
    weights = np.random.normal(0.0, 1 / n_features ** 0.5, n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in bd_zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            #   Notice we multiply the inputs and the weights here
            #   rather than storing h as a separate variable
            output = bd_sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            # error = bd_error_formula(y, output)

            # The error term
            error_term = bd_error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times
            # the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = bd_sigmoid(np.dot(features.values, weights))
            loss = np.mean((out - targets) ** 2)
            msg = "Epoch:" + str(e) + ','
            if last_loss and last_loss < loss:
                print(msg, "Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print(msg, "Train loss: ", loss)
            last_loss = loss
    print("Finished training!")
    print("=" * 50)
    return weights


def bd_main():
    # str(Path(__file__).parent / 'student_data.csv')
    data = bd_read_data()

    # TODO:  Make dummy variables for rank
    one_hot_data = pd.get_dummies(data, columns=['rank']).astype(np.float64)

    # Scaling the data
    #
    # The next step is to scale the data. We notice that the range for
    # grades is 1.0-4.0, whereas the range for test scores is roughly
    # 200-800, which is much larger. This means our data is skewed, and
    # that makes it hard for a neural network to handle. Let's fit our
    # two features into a range of 0-1, by dividing the grades by 4.0,
    # and the test score by 800.

    # Making a copy of our data
    processed_data = one_hot_data[:]

    # TODO: Scale the columns
    processed_data['gre'] /= processed_data['gre'].max()
    processed_data['gpa'] /= processed_data['gpa'].max()

    # Splitting the data into Training and Testing
    #
    # In order to test our algorithm, we'll split the data into a Training
    # and a Testing set. The size of the testing set will be 10% of the
    # total data.

    sample = np.random.choice(
        processed_data.index,
        size=int(len(processed_data) * 0.9),
        replace=False,
    )
    train_data = processed_data.iloc[sample].reset_index(drop=True)
    test_data = processed_data.drop(sample).reset_index(drop=True)

    print("Number of training samples is", len(train_data))
    print("Number of testing samples is", len(test_data))

    # Splitting the data into features and targets (labels)
    #
    # Now, as a final step before the training, we'll split the data into
    # features (X) and targets (y).

    features = train_data.drop('admit', axis=1)
    targets = train_data['admit']
    features_test = test_data.drop('admit', axis=1)
    targets_test = test_data['admit']

    # Training the 2-layer Neural Network
    #
    # The following function trains the 2-layer neural network. First,
    # we'll write some helper functions.

    # Neural Network hyperparameters
    epochs = 1000
    learnrate = 0.5

    weights = bd_train_nn(features, targets, epochs, learnrate)

    # Calculate accuracy on test data
    test_out = bd_sigmoid(np.dot(features_test, weights))
    predictions = test_out > 0.5
    accuracy = np.mean(predictions == targets_test)
    print("Prediction accuracy: {:.3f}".format(accuracy))


if __name__ == '__main__':
    setup()

    t0 = time.time()
    main()
    t1 = time.time()

    print('=' * 50)
    print('Training time using pure python/numpy:', t1 - t0, 's')
    print('=' * 50)

    t0 = time.time()
    bd_main()
    t1 = time.time()

    print('=' * 50)
    print('Training time using bodo:', t1 - t0, 's')
    print('=' * 50)
