import random
from math_utils.operations import dot_product
from math_utils.type import is_num


class DenseLayer:
    def __init__(self, input_size, neurons, activation=None):
        self.input_size = input_size
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(neurons)] for _ in range(input_size)]
        self.bias = [0 for _ in range(neurons)]
        self.activation = activation

    def impulse(self, x):
        products = dot_product(x, self.weights)
        if is_num(products):
            products += self.bias[0]
        else:
            for e in products:
                for i, _ in enumerate(e):
                    e[i] += self.bias[i]
        if self.activation:
            return self.activation(products)
        else:
            return products
