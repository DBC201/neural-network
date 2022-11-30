from nnfs.datasets import spiral_data
from layers.denseLayer import NaiveDenseLayer
from activation_functions.softmax import naive_softmax_activation_function

if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)
    X = X.tolist()

    dense1 = NaiveDenseLayer(2, 3, naive_softmax_activation_function)

    dense2 = NaiveDenseLayer(3, 3, naive_softmax_activation_function)

    r1 = dense1.impulse(X)
    # print(r1)

    r2 = dense2.impulse(r1)
    print(r2[:5])
