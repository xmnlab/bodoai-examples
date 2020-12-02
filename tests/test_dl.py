import os

import bodo
import numpy as np
from torchvision import datasets

from bodoai_examples.dl import deep_learning


def generate_train_test_data():
    # https://github.com/Bodo-inc/Bodo-examples/blob/master/
    #   deep_learning/generate_mnist_data.py
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    train = datasets.MNIST("data", train=True, download=True)
    test = datasets.MNIST("data", train=False)

    train.data.numpy().tofile(f"{data_dir}/train_data.dat")
    train.targets.numpy().tofile(f"{data_dir}/train_targets.dat")

    test.data.numpy().tofile(f"{data_dir}/test_data.dat")
    test.targets.numpy().tofile(f"{data_dir}/test_targets.dat")


generate_train_test_data()


@bodo.jit
def test_dl():
    X_train = np.fromfile("data/train_data.dat", np.uint8)
    X_train = X_train.reshape(60000, 28, 28)
    y_train = np.fromfile("data/train_targets.dat", np.int64)

    X_test = np.fromfile("data/test_data.dat", np.uint8)
    X_test = X_test.reshape(10000, 28, 28)
    y_test = np.fromfile("data/test_targets.dat", np.int64)

    # preprocessing: do image normalization in Bodo
    # https://pytorch.org/docs/stable/torchvision/transforms.html#
    #   torchvision.transforms.ToTensor
    # https://pytorch.org/docs/stable/torchvision/transforms.html#
    #   torchvision.transforms.Normalize
    # using mean=0.1307, std=0.3081
    X_train = ((X_train / 255) - 0.1307) / 0.3081
    X_train = X_train.astype(np.float32)
    X_test = ((X_test / 255) - 0.1307) / 0.3081
    X_test = X_test.astype(np.float32)

    bodo.dl.start("torch")

    X_train = bodo.dl.prepare_data(X_train)
    y_train = bodo.dl.prepare_data(y_train)
    X_test = bodo.dl.prepare_data(X_test)
    y_test = bodo.dl.prepare_data(y_test)

    with bodo.objmode:
        deep_learning(X_train, y_train, X_test, y_test)
    bodo.dl.end()
