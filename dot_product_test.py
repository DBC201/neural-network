import numpy
from layer import Layer

if __name__ == '__main__':
    i0 = [1, 2]
    w0 = [3, 4]

    print(numpy.dot(w0, i0))
    print(Layer.dot_product(w0, i0))
    print("---")

    i1 = [[1, 2], [3, 4]]
    w1 = [1, 2]

    print(numpy.dot(w1, i1))
    print(Layer.dot_product(w1, i1))
    print("---")

    i2 = [1, 2]
    w2 = [[1, 2], [3, 4]]
    print(numpy.dot(w2, i2))
    print(Layer.dot_product(w2, i2))
    print("---")

    i3 = [[1, 2, 3], [3, 4, 5]]
    w3 = [1, 2]
    print(numpy.dot(w3, i3))
    print(Layer.dot_product(w3, i3))
    print("---")

    i4 = [[[1, 2, 3], [3, 4, 5]], [[-1, -2, -3], [-3, -4, -5]], [[-1, -2, -3], [-3, -4, -5]]]
    w4 = [1, 2]
    print(numpy.dot(w4, i4))
    print(Layer.dot_product(w4, i4))
    print("---")

    i5 = [1, 2, 3, 4]
    w5 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 3, 1]
    ]
    print(numpy.dot(w5, i5))
    print(Layer.dot_product(w5, i5))
    print("---")
