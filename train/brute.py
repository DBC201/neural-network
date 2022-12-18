import sys

import nnfs
from nnfs.datasets import vertical_data, spiral_data
from layers.denseLayer import NumpyDenseLayer as DenseLayer
from activation_functions.softmax import numpy_softmax_activation_function as softmax
from activation_functions.relu import numpy_relu_activation_function as relu
from loss_functions.caterogical_cross_entropy import numpy_categorical_cross_entropy as loss
import numpy

nnfs.init()
def iterate(X, y):
    dense1 = DenseLayer(2, 3, relu)

    dense2 = DenseLayer(3, 3, softmax)

    best_w1 = dense1.weights.copy()
    best_b1 = dense1.bias.copy()

    best_w2 = dense2.weights.copy()
    best_b2 = dense2.bias.copy()

    best_loss = sys.maxsize

    for i in range(10000):
        dense1.weights += numpy.random.randn(2, 3) * 0.05
        dense1.bias += numpy.random.randn(1, 3) * 0.05

        dense2.weights += numpy.random.randn(3, 3) * 0.05
        dense2.bias += numpy.random.randn(1, 3) * 0.05

        r1 = dense1.impulse(X)

        r2 = dense2.impulse(r1)

        l = loss(r2, y)

        if l < best_loss:
            #p = numpy.argmax(r2, axis=1)
            #print(numpy.mean(p == y), l)
            best_loss = l
            best_w1 = dense1.weights.copy()
            best_w2 = dense2.weights.copy()
            best_b1 = dense1.bias.copy()
            best_b2 = dense2.bias.copy()
        else:
            dense1.weights = best_w1.copy()
            dense1.bias = best_b1.copy()
            dense2.weights = best_w2.copy()
            dense2.bias = best_b2.copy()
    res = dense2.impulse(dense1.impulse(X))
    l = loss(res, y)
    accuracy = numpy.mean(numpy.argmax(res, axis=1) == y)
    return l, accuracy


if __name__ == '__main__':
    X, y = vertical_data(samples=100, classes=3)
    print(iterate(X, y))
    X, y = spiral_data(samples=100, classes=3)
    print(iterate(X, y))

