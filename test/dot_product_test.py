import numpy
from math_utils.operations import naive_dot_product
from math_utils.operations import naive_transpose

if __name__ == '__main__':
    i0 = [1, 2]
    w0 = [3, 4]

    print(numpy.dot(w0, i0))
    print(naive_dot_product(w0, i0))
    print("---")

    i1 = [[1, 2], [3, 4]]
    w1 = [1, 2]

    print(numpy.dot(w1, i1))
    print(naive_dot_product(w1, i1))
    print("---")

    i2 = [1, 2]
    w2 = [[1, 2], [3, 4]]
    print(numpy.dot(w2, i2))
    print(naive_dot_product(w2, i2))
    print("---")

    i3 = [[1, 2, 3], [3, 4, 5]]
    w3 = [1, 2]
    print(numpy.dot(w3, i3))
    print(naive_dot_product(w3, i3))
    print("---")

    i4 = [[[1, 2, 3], [3, 4, 5]], [[-1, -2, -3], [-3, -4, -5]], [[-1, -2, -3], [-3, -4, -5]]]
    w4 = [1, 2]
    print(numpy.dot(w4, i4))
    print(naive_dot_product(w4, i4))
    print("---")

    i5 = [1, 2, 3, 4]
    w5 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 2, 3, 1]
    ]
    print(numpy.dot(w5, i5))
    print(naive_dot_product(w5, i5))
    print("---")

    i6 = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

    w6 = [[0.2, 0.5, -0.26],
          [0.8, -0.91, -0.27],
          [-0.5, 0.26, 0.17],
          [1.0, -0.5, 0.87]]

    print(numpy.dot(i6, w6))
    print(naive_dot_product(i6, w6))
    print("----")

    x = [[1, 2, 3], [4, 5, 6]]
    print(numpy.array(x).T)
    print(naive_transpose(x))
    print("---")

    print(numpy.array(w6).T)
    print(naive_transpose(w6))
