import nnfs
from nnfs.datasets import spiral_data
from layers.denseLayer import DenseLayer
from activation_functions.softmax import softmax_activation_function

nnfs.init()

if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)
    X = X.tolist()

    dense1 = DenseLayer(2, 3, softmax_activation_function)

    dense2 = DenseLayer(3, 3, softmax_activation_function)

    r1 = dense1.impulse(X)
    # print(r1)

    r2 = dense2.impulse(r1)
    print(r2[:5])
