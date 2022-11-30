import random
from math_utils.operations import naive_dot_product
from math_utils.type import naive_is_num
import numpy


class NaiveDenseLayer:
    def __init__(self, input_size, neurons, activation=None):
        self.input_size = input_size
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(neurons)] for _ in range(input_size)]
        self.bias = [0 for _ in range(neurons)]
        self.activation = activation

    def impulse(self, x):
        products = naive_dot_product(x, self.weights)
        if naive_is_num(products):
            products += self.bias[0]
        else:
            for e in products:
                for i, _ in enumerate(e):
                    e[i] += self.bias[i]
        if self.activation:
            return self.activation(products)
        else:
            return products


class NumpyDenseLayer:
    def __init__(self, input_size, neurons, activation=None):
        self.input_size = input_size
        self.weights = 0.1 * numpy.random.randn(input_size, neurons)
        self.bias = numpy.zeros((1, neurons))
        self.activation = activation

    def impulse(self, x):
        products = numpy.dot(x, self.weights) + self.bias
        if self.activation:
            return self.activation(products)
        else:
            return products
