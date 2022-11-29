from math import e
from math_utils.type import is_num


def sigmoid(a):
    return 1/(1+(e**(a*-1)))


def sigmoid_activation_function(x):
    if is_num(x):
        return sigmoid(x)
    elif is_num(x[0]):
        res = []
        for i in x:
            res.append(sigmoid(i))
        return res
    else:
        res = []
        for i in x:
            res.append(sigmoid_activation_function(i))
        return res
