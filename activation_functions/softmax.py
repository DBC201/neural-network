from math import e
from math_utils.type import naive_is_num
import numpy


def naive_normalization(a):
    if naive_is_num(a):
        return 0
    elif naive_is_num(a[0]):
        m = max(a)
        res = []
        for i in a:
            res.append(i-m)
        return res
    else:
        res = []
        for i in a:
            res.append(naive_normalization(i))
        return res


def naive_exponentiation(a):
    if naive_is_num(a):
        return e**a
    elif naive_is_num(a[0]):
        res = []
        for i in a:
            res.append(e**i)
        return res
    else:
        res = []
        for i in a:
            res.append(naive_exponentiation(i))
        return res


def naive_softmax_activation_function(x):
    if naive_is_num(x):
        return 1/x
    elif naive_is_num(x[0]):
        x = naive_exponentiation(naive_normalization(x))
        res = []
        s = sum(x)
        for i in x:
            res.append(i/s)
        return res
    else:
        res = []
        for i in x:
            res.append(naive_softmax_activation_function(i))
        return res


def numpy_softmax_activation_function(x):
    exp_values = numpy.exp(x - numpy.max(x, axis=1, keepdims=True))
    return exp_values / numpy.sum(exp_values, axis=1, keepdims=True)
