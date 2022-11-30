import nnfs
from nnfs.datasets import spiral_data
from layers.denseLayer import NumpyDenseLayer as DenseLayer
from activation_functions.softmax import numpy_softmax_activation_function as softmax
from activation_functions.relu import numpy_relu_activation_function as relu


nnfs.init()


X, y = spiral_data(samples=100, classes=3)

dense1 = DenseLayer(2, 3, relu)

dense2 = DenseLayer(3, 3, softmax)

r1 = dense1.impulse(X)

r2 = dense2.impulse(r1)
print(r2[:5])
