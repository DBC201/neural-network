from loss_functions.caterogical_cross_entropy import naive_categorical_cross_entropy, numpy_categorical_cross_entropy
from activation_functions.relu import numpy_relu_activation_function
from activation_functions.softmax import numpy_softmax_activation_function
from layers.denseLayer import NumpyDenseLayer
from nnfs.datasets import spiral_data
from nnfs import init as nnfs_init

nnfs_init()

X, y = spiral_data(samples=100, classes=3)

l1 = NumpyDenseLayer(2, 3, numpy_relu_activation_function)

l2 = NumpyDenseLayer(3, 3, numpy_softmax_activation_function)

res = l2.impulse(l1.impulse(X))

naive_loss = naive_categorical_cross_entropy(res[:5].tolist(), list(map(int, y.tolist()[5:])))
numpy_loss = numpy_categorical_cross_entropy(res[:5], y[:5])
print(naive_loss)
print(numpy_loss)
